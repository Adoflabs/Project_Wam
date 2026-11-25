"""Automated experience collection over seeded booking scenarios.

Runs many episodes through the real observer, executor and recorder with
three policies:

  * goal: the goal-directed decider (successes, natural slot picks,
    deliberate race conflicts and invalid emails).
  * noisy: the same decider plus detours: invalid text later corrected,
    changing a selection, Back, unticking consent, clicking Next too early,
    and recovering from a race conflict by choosing another time.
  * explore: random available actions for a bounded number of steps.

Some goal episodes become counterfactual branch groups: the backend is
reset, the page reloaded and the same prefix replayed, the starting state
is verified (visible state and full backend state), and then a different
action is recorded in each branch.

Records go to `<out>/<split>/transitions.jsonl`, so splits never mix; a
manifest lists every episode with its status. `validate_raw()` re-checks
every run from the records themselves, without trusting stored flags.

Policies use privileged knowledge (the goal, the scenario's wording) to
steer collection; none of it is model input.

Usage:

  python -m browser_world.collection.collector --transitions 50 --out data/raw/e04/pilot
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import json
import pathlib
import random
try:
    import resource
except ImportError:
    resource = None
import socket
import sys
import time
import traceback
import uuid
from typing import Any

from playwright import sync_api

from browser_world.environment import booking
from browser_world.interaction import executor
from browser_world.collection import goal_policy
from browser_world.diagnostics import live_demo
from browser_world.interaction import observer
from browser_world.collection import recorder
from browser_world.environment import scenarios
from browser_world.environment import server

GOAL = 'goal'
NOISY = 'noisy'
EXPLORE = 'explore'
POLICY_WEIGHTS = {GOAL: 0.4, NOISY: 0.3, EXPLORE: 0.3}
SPLIT_WEIGHTS = {'train': 0.7, 'validation': 0.15, 'test': 0.15}
# Interleaved schedules with exactly those shares, so even a small run
# covers every policy and split. Episode i takes its policy from
# POLICY_SCHEDULE; each policy then steps through SPLIT_SCHEDULE on its own
# counter, so every policy (and so every split) gets the 70/15/15 shares.
POLICY_SCHEDULE = (GOAL, NOISY, EXPLORE, GOAL, EXPLORE, NOISY, GOAL, NOISY,
                   EXPLORE, GOAL)
SPLIT_SCHEDULE = ('train', 'validation', 'train', 'test', 'train', 'train',
                  'train', 'train', 'validation', 'train', 'train', 'test',
                  'train', 'train', 'train', 'validation', 'train', 'test',
                  'train', 'train')
# Goal outcomes, cycled separately within each split so every split gets
# each one: 40% success, 20% natural slot choice, 20% deliberate race,
# 20% invalid email. Rarer outcomes come first, so even the small
# validation and test splits see them early.
GOAL_OUTCOME_SCHEDULE = (scenarios.INVALID_EMAIL, scenarios.SUCCESS,
                         scenarios.CONFLICT, scenarios.NATURAL,
                         scenarios.SUCCESS)
NOISY_OUTCOME_SCHEDULE = (scenarios.SUCCESS, scenarios.CONFLICT)
# Collection seeds start here, apart from the seeds used while building.
SEED_BASE = 10_000
BRANCH_EVERY = 4  # Every fourth goal episode is a branch group.
MANIFEST_FILE = 'manifest.json'

COMPLETE = 'complete'
FAILED = 'failed'
TIMED_OUT = 'timed_out'
REFUSED = 'refused'
ABANDONED = 'abandoned'

_GOAL_MAX_STEPS = 45
_EXPLORE_STEPS = (8, 16)
_DETOUR_CHANCE = 0.3
_BRANCHES = 3
_BRANCH_PREFIX = (1, 6)


@dataclasses.dataclass
class Episode:
    """One manifest entry: a run (or a branch) and how it ended.

    Attributes:
      episode_id: Unique id; branch runs share their group's prefix.
      run_id: The recorder run, or None if nothing was recorded.
      split: 'train', 'validation' or 'test'.
      family: Scenario family.
      scenario_id: Scenario id.
      seed: Scenario seed.
      policy: GOAL, NOISY or EXPLORE ('goal' for branches).
      goal_outcome: The goal the policy pursued, or None for EXPLORE.
      status: COMPLETE (reached its end; for EXPLORE, its step budget),
        FAILED (an error), TIMED_OUT (a goal episode hit the step limit),
        REFUSED (the executor refused an action) or ABANDONED (nothing
        useful offered, or a branch point could not be restored).
      ended: How the page ended ('booked', 'conflict', ...), 'step_budget'
        for exploration, or None.
      steps: Transitions recorded.
      tags: Count of each behaviour tag (detours, recoveries, ...).
      branch: Branch metadata for counterfactual runs, else None.
      baseline_artifacts: Whether screenshots and HTML were saved.
      duration_s: Wall time.
      error: Failure detail, if any.
    """

    episode_id: str
    run_id: str | None
    split: str
    family: str
    scenario_id: str
    seed: int
    policy: str
    goal_outcome: str | None
    status: str = TIMED_OUT
    ended: str | None = None
    steps: int = 0
    tags: dict[str, int] = dataclasses.field(default_factory=dict)
    branch: dict[str, Any] | None = None
    baseline_artifacts: bool = False
    duration_s: float = 0.0
    error: str | None = None


def _weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _corrupt(value: str, rng: random.Random) -> str:
    """Returns an invalid variant of a name, email or phone value."""
    if '@' in value:
        return rng.choice((value.split('@')[0], value.replace('.com', ''),
                           value.replace('@', ' at ')))
    if any(c.isdigit() for c in value):
        return value[:rng.randint(3, 7)]
    return value[:1]


def _offer_key(offer: executor.Affordance,
               text: str | None) -> tuple[Any, ...]:
    return (offer.kind, offer.name, offer.checked, offer.scroll_y, text)


class Collector:
    """Collects episodes in one browser page and records them by split."""

    def __init__(self, page: sync_api.Page, watcher: observer.Observer,
                 site_url: str, assignments: dict[str, booking.Scenario],
                 out_dir: pathlib.Path, speed: executor.Speed,
                 baseline_episodes: int = 0):
        """Creates a collector.

        Args:
          page: The page to collect in.
          watcher: The observer attached to `page`.
          site_url: The booking server's URL.
          assignments: The server's session-to-scenario map, filled here.
          out_dir: Root directory for raw records and the manifest.
          speed: Executor speed.
          baseline_episodes: The first this many episodes also save
            screenshots and HTML, as a comparison subset.
        """
        self._page = page
        self._watcher = watcher
        self._site = site_url
        self._assignments = assignments
        self._out = out_dir
        self._hand = executor.Executor(page, watcher, speed)
        self._baseline_episodes = baseline_episodes
        self._session_id = ''
        self.episodes: list[Episode] = []

    def _recorder(self, split: str, baselines: bool) -> recorder.Recorder:
        return recorder.Recorder(self._page, self._watcher, self._hand,
                                 self._out / split, truth_fn=self._truth,
                                 capture_baselines=baselines)

    def _truth(self) -> Any:
        return self._page.request.get(
            f'{self._site}/api/truth/state',
            headers={server.SESSION_HEADER: self._session_id}).json()

    def _open(self, scenario: booking.Scenario) -> None:
        """Starts a fresh session of `scenario` in the page."""
        self._session_id = uuid.uuid4().hex
        self._assignments[self._session_id] = scenario
        self._page.goto(f'{self._site}/?session={self._session_id}')
        self._ready()

    def _reset(self) -> None:
        """Resets the session's backend and reloads the page."""
        self._page.request.post(
            f'{self._site}/api/reset',
            headers={server.SESSION_HEADER: self._session_id})
        self._page.reload()
        self._ready()

    def _ready(self) -> None:
        self._page.wait_for_selector('body[data-ready=true]')
        self._watcher.wait_until_settled()
        self._watcher.observe()

    def _observe(self) -> observer.Observation:
        self._watcher.wait_until_settled()
        return self._watcher.observe()

    def collect(self, transitions: int, seed_base: int = SEED_BASE,
                policy_weights: Mapping[str, float] | None = None,
                branch_every: int = BRANCH_EVERY) -> None:
        """Runs episodes until at least `transitions` are recorded.

        The manifest is rewritten after every episode.

        Args:
          transitions: How many transitions to record at least.
          seed_base: First scenario seed; episode i uses seed_base + i.
          policy_weights: Random share of episodes per policy instead of
            POLICY_SCHEDULE (used by tests to force one policy).
          branch_every: The first goal episode and every this many after
            it become counterfactual branch groups; 0 means never.
        """
        rng = random.Random(f'collect:{seed_base}')
        recorded = 0
        index = 0
        per_policy: collections.Counter[str] = collections.Counter()
        per_split: collections.Counter[tuple[str, str]] = (
            collections.Counter())
        while recorded < transitions:
            policy = (_weighted(rng, policy_weights) if policy_weights
                      else POLICY_SCHEDULE[index % len(POLICY_SCHEDULE)])
            split = SPLIT_SCHEDULE[per_policy[policy] % len(SPLIT_SCHEDULE)]
            per_policy[policy] += 1
            family = rng.choice(scenarios.SPLITS[split])
            seed = seed_base + index
            baselines = index < self._baseline_episodes
            scenario = scenarios.generate(family, seed)
            if (policy == GOAL and branch_every
                    and per_policy[GOAL] % branch_every == 1 % branch_every):
                new = self._branch_group(scenario, split, seed)
            else:
                schedule = {GOAL: GOAL_OUTCOME_SCHEDULE,
                            NOISY: NOISY_OUTCOME_SCHEDULE}.get(policy)
                outcome = None
                if schedule is not None:
                    turn = per_split[(policy, split)]
                    per_split[(policy, split)] += 1
                    outcome = schedule[turn % len(schedule)]
                new = [self._episode(policy, scenario, split, seed, outcome,
                                     baselines)]
            for episode in new:
                recorded += episode.steps
            self.episodes.extend(new)
            self._write_manifest()
            index += 1

    def _metadata(self, scenario: booking.Scenario, split: str,
                  seed: int, policy: str) -> dict[str, Any]:
        return {'scenario_id': scenario.scenario_id,
                'family': scenario.scenario_id.rsplit('-', 1)[0],
                'split': split, 'seed': seed, 'policy': policy}

    def _episode(self, policy: str, scenario: booking.Scenario, split: str,
                 seed: int, outcome: str | None, baselines: bool) -> Episode:
        """Runs one episode with `policy` toward `outcome` (None for
        exploration); never raises."""
        rng = random.Random(f'episode:{scenario.scenario_id}:{policy}')
        goal = scenarios.goal(scenario, seed, outcome or scenarios.SUCCESS)
        episode = Episode(
            episode_id=f'{policy}-{scenario.scenario_id}', run_id=None,
            split=split, family=scenario.scenario_id.rsplit('-', 1)[0],
            scenario_id=scenario.scenario_id, seed=seed, policy=policy,
            goal_outcome=outcome, baseline_artifacts=baselines)
        started = time.monotonic()
        try:
            self._open(scenario)
            rec = self._recorder(split, baselines)
            episode.run_id = rec.new_run(
                episode.episode_id, 'start',
                scenario=self._metadata(scenario, split, seed, policy))
            self._drive(episode, rec, scenario, goal, rng)
        except Exception:  # pylint: disable=broad-exception-caught
            # Isolation point: one failed episode is recorded in the
            # manifest and collection continues with a fresh session.
            episode.status = FAILED
            episode.error = traceback.format_exc(limit=3)
        episode.duration_s = round(time.monotonic() - started, 2)
        return episode

    def _drive(self, episode: Episode, rec: recorder.Recorder,
               scenario: booking.Scenario, goal: scenarios.Goal,
               rng: random.Random) -> None:
        """Steps the episode until it ends; fills in status and counts."""
        text = scenario.presentation.text
        wanted = list(goal_policy.intents(goal, text))
        max_steps = (rng.randint(*_EXPLORE_STEPS)
                     if episode.policy == EXPLORE else _GOAL_MAX_STEPS)
        detours = rng.randint(1, 3) if episode.policy == NOISY else 0
        tags: collections.Counter[str] = collections.Counter()
        for _ in range(max_steps):
            observation = self._observe()
            ended = goal_policy.outcome_of(observation, text)
            if ended == goal_policy.CONFLICT and episode.policy == NOISY and (
                    not tags['conflict_recovery']):
                choice = self._recover(observation, text)
                if choice is not None:
                    replacement = scenarios.goal(scenario, episode.seed + 1,
                                                 scenarios.SUCCESS)
                    wanted = list(goal_policy.intents(replacement, text))
            elif _terminal(episode.policy, ended):
                episode.status, episode.ended = COMPLETE, ended
                break
            else:
                choice = self._choose(episode.policy, observation, wanted,
                                      goal, text, rng, detours > 0)
            if choice is None:
                episode.status = ABANDONED
                episode.error = 'No useful action was offered.'
                break
            action, tag = choice
            if tag not in ('goal', 'explore'):
                detours -= 1
            outcome = rec.record_step(episode.run_id or '', observation,
                                      action)
            if not isinstance(outcome, recorder.Transition):
                episode.status = REFUSED
                episode.error = f'{outcome.refusal}: {outcome.detail}'
                break
            episode.steps += 1
            tags[tag] += 1
        else:
            if episode.policy == EXPLORE:  # Used its step budget, as meant.
                episode.status, episode.ended = COMPLETE, 'step_budget'
        episode.tags = dict(tags)

    def _recover(self, observation: observer.Observation,
                 text: Mapping[str, str]
                 ) -> tuple[executor.Action, str] | None:
        for offer in executor.available_actions(observation):
            if offer.kind == executor.CLICK and offer.name == text[
                    'choose_another']:
                return offer.action(), 'conflict_recovery'
        return None

    def _choose(self, policy: str, observation: observer.Observation,
                wanted: Sequence[goal_policy.Intent], goal: scenarios.Goal,
                text: Mapping[str, str], rng: random.Random,
                may_detour: bool) -> tuple[executor.Action, str] | None:
        """Returns the policy's next action and a behaviour tag."""
        if policy == EXPLORE:
            return _explore(observation, goal, rng)
        if policy == NOISY and may_detour and rng.random() < _DETOUR_CHANCE:
            detour = _detour(observation, wanted, goal, text, rng)
            if detour is not None:
                return detour
        choice = goal_policy.choose(
            observation, goal_policy.pending(observation, wanted), text)
        return None if choice is None else (choice[0], 'goal')

    def _branch_group(self, scenario: booking.Scenario, split: str,
                      seed: int) -> list[Episode]:
        """Records one alternative action per branch from a verified,
        identical starting state."""
        rng = random.Random(f'branch:{scenario.scenario_id}')
        goal = scenarios.goal(scenario, seed, scenarios.SUCCESS)
        text = scenario.presentation.text
        wanted = goal_policy.intents(goal, text)
        prefix_length = rng.randint(*_BRANCH_PREFIX)
        group = f'branch-{scenario.scenario_id}'
        base = Episode(episode_id=group, run_id=None, split=split,
                       family=scenario.scenario_id.rsplit('-', 1)[0],
                       scenario_id=scenario.scenario_id, seed=seed,
                       policy=GOAL, goal_outcome=scenarios.SUCCESS)
        started = time.monotonic()
        try:
            self._open(scenario)
            prefix = self._replay(wanted, text, prefix_length)
            point = self._observe()
            reference = recorder.canonical(point)
            truth = self._truth()
            fingerprint = hashlib.sha256(json.dumps(
                reference, sort_keys=True).encode()).hexdigest()[:16]
            alternatives = _alternatives(point, goal, text, rng)
        except Exception:  # pylint: disable=broad-exception-caught
            # Isolation point: see _episode.
            base.status, base.error = FAILED, traceback.format_exc(limit=3)
            base.duration_s = round(time.monotonic() - started, 2)
            return [base]
        branches = []
        for index, key in enumerate(alternatives):
            branch = dataclasses.replace(
                base, episode_id=f'{group}/b{index}', tags={},
                branch={'group': group, 'index': index,
                        'prefix': prefix, 'prefix_length': len(prefix),
                        'point_fingerprint': fingerprint,
                        'action': list(key)})
            self._run_branch(branch, scenario, wanted, text, len(prefix),
                             index, reference, truth, key)
            branches.append(branch)
        return branches

    def _run_branch(self, branch: Episode, scenario: booking.Scenario,
                    wanted: Sequence[goal_policy.Intent],
                    text: Mapping[str, str], prefix_length: int, index: int,
                    reference: dict[str, Any], truth: Any,
                    key: tuple[Any, ...]) -> None:
        """Restores the branch point (from the second branch on), checks
        it, and records the alternative action."""
        started = time.monotonic()
        try:
            if index:
                self._reset()
                self._replay(wanted, text, prefix_length)
            point = self._observe()
            found = recorder.differences(reference, recorder.canonical(point))
            if found or self._truth() != truth:
                branch.status = ABANDONED
                branch.error = ('Branch point not restored: '
                                f'{found[:3] or "backend state differs"}')
                return
            offer = next((o for o in executor.available_actions(point)
                          if _offer_key(o, key[-1])[:4] == key[:4]), None)
            if offer is None:
                branch.status, branch.error = ABANDONED, 'Action not offered.'
                return
            rec = self._recorder(branch.split, False)
            metadata = self._metadata(scenario, branch.split, branch.seed,
                                      GOAL)
            metadata['branch'] = branch.branch
            branch.run_id = rec.new_run(branch.episode_id, 'branch',
                                        scenario=metadata)
            outcome = rec.record_step(branch.run_id, point,
                                      offer.action(key[-1]))
            if not isinstance(outcome, recorder.Transition):
                branch.status = REFUSED
                branch.error = f'{outcome.refusal}: {outcome.detail}'
                return
            branch.steps, branch.status = 1, COMPLETE
            branch.tags = {'branch': 1}
        except Exception:  # pylint: disable=broad-exception-caught
            # Isolation point: see _episode.
            branch.status, branch.error = FAILED, traceback.format_exc(limit=3)
        finally:
            branch.duration_s = round(time.monotonic() - started, 2)

    def _replay(self, wanted: Sequence[goal_policy.Intent],
                text: Mapping[str, str], length: int) -> list[str]:
        """Executes the goal decider's first `length` actions without
        recording them; returns their descriptions."""
        done = []
        for _ in range(length):
            observation = self._observe()
            if goal_policy.outcome_of(observation, text) is not None:
                break
            choice = goal_policy.choose(
                observation, goal_policy.pending(observation, wanted), text)
            if choice is None:
                break
            result = self._hand.execute(observation, choice[0])
            if not result.executed:
                raise RuntimeError(f'Prefix refused: {result.refusal}')
            action = choice[0]
            done.append(f'{action.kind}:{action.text or action.checked or ""}'
                        f':{action.scroll_y or ""}:'
                        f'{_name_of(observation, action.target)}')
        return done

    def _write_manifest(self) -> None:
        self._out.mkdir(parents=True, exist_ok=True)
        data = {'schema_version': recorder.SCHEMA_VERSION,
                'splits': scenarios.SPLITS,
                'episodes': [dataclasses.asdict(e) for e in self.episodes]}
        (self._out / MANIFEST_FILE).write_text(json.dumps(data, indent=1))


def _terminal(policy: str, ended: str | None) -> bool:
    if policy == GOAL:
        return ended is not None
    return ended == goal_policy.DONE


def _name_of(observation: observer.Observation, node_id: int | None) -> str:
    for element in observation.elements:
        if element.node_id == node_id:
            return element.name
    return 'page'


def _text_pool(goal: scenarios.Goal, rng: random.Random) -> list[str]:
    contact = goal.contact
    values = [contact.name, contact.email, contact.phone]
    return values + [_corrupt(v, rng) for v in values] + ['']


def _explore(observation: observer.Observation, goal: scenarios.Goal,
             rng: random.Random) -> tuple[executor.Action, str] | None:
    offers = executor.available_actions(observation)
    if not offers:
        return None
    offer = rng.choice(offers)
    text = rng.choice(_text_pool(goal, rng)) if offer.kind == executor.TYPE \
        else None
    return offer.action(text), 'explore'


def _detour(observation: observer.Observation,
            wanted: Sequence[goal_policy.Intent], goal: scenarios.Goal,
            text: Mapping[str, str], rng: random.Random
            ) -> tuple[executor.Action, str] | None:
    """Returns a noisy deviation from the goal, tagged by kind."""
    offers = executor.available_actions(observation)
    goal_names = {goal.service, goal.doctor, goal.date}
    placeholders = {text['pick_service'], text['pick_doctor'],
                    text['pick_date']}
    fields = {text['name']: goal.contact.name,
              text['email']: goal.contact.email,
              text['phone']: goal.contact.phone}
    visible_unfinished = any(goal_policy.satisfied(observation, i) is False
                             for i in wanted)
    candidates: dict[str, list[executor.Action]] = collections.defaultdict(
        list)
    for offer in offers:
        if offer.kind == executor.TYPE and offer.name in fields:
            candidates['invalid_text'].append(
                offer.action(_corrupt(fields[offer.name], rng)))
        elif (offer.kind == executor.SELECT
              and offer.name not in goal_names | placeholders
              and goal_policy.satisfied(observation, goal_policy.Intent(
                  executor.SELECT, offer.name)) is False):
            candidates['change_selection'].append(offer.action())
        elif offer.kind == executor.CLICK and offer.name == text['back']:
            candidates['back'].append(offer.action())
        elif (offer.kind == executor.CHECK and offer.name == text['consent']
              and offer.checked is False):
            candidates['consent_toggle'].append(offer.action())
        elif (offer.kind == executor.CLICK and offer.name == text['next']
              and visible_unfinished):
            candidates['premature_next'].append(offer.action())
    if not candidates:
        return None
    tag = rng.choice(sorted(candidates))
    return rng.choice(candidates[tag]), tag


def _alternatives(point: observer.Observation, goal: scenarios.Goal,
                  text: Mapping[str, str],
                  rng: random.Random) -> list[tuple[Any, ...]]:
    """Picks up to `_BRANCHES` different actions from the branch point:
    the goal's own next action first, then others of varied kinds."""
    offers = executor.available_actions(point)
    choice = goal_policy.choose(point, goal_policy.pending(
        point, goal_policy.intents(goal, text)), text)
    keys = []
    if choice is not None:
        action = choice[0]
        keys.append((action.kind, _name_of(point, action.target),
                     action.checked, action.scroll_y, action.text))
    rng.shuffle(offers)
    for offer in sorted(offers, key=lambda o: o.kind in {k[0] for k in keys}):
        text_value = (rng.choice(_text_pool(goal, rng))
                      if offer.kind == executor.TYPE else None)
        key = _offer_key(offer, text_value)
        if key[:4] not in {k[:4] for k in keys}:
            keys.append(key)
        if len(keys) == _BRANCHES:
            break
    return keys


def validate_raw(out_dir: pathlib.Path) -> dict[str, Any]:
    """Re-checks every recorded run from the records alone.

    For each split file: schema version 3, records in the split they claim,
    steps 0..n-1 per run, and at every boundary between consecutive steps
    the same visible content, page epoch, page version and backend truth.
    Stored `continuity` flags are not trusted. Branch groups must stay
    within one split.

    Returns:
      Counts and a list of problems; no problems means valid.
    """
    problems = []
    runs_total = 0
    records_total = 0
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for split in scenarios.SPLITS:
        path = out_dir / split / recorder.TRANSITIONS_FILE
        if not path.exists():
            continue
        runs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for record in recorder.read_jsonl(path):
            records_total += 1
            runs[record['run_id']].append(record)
            if record['schema_version'] != recorder.SCHEMA_VERSION:
                problems.append(f"{record['run_id']}: schema "
                                f"{record['schema_version']}")
            if record['scenario']['split'] != split:
                problems.append(f"{record['run_id']}: in {split} but claims "
                                f"{record['scenario']['split']}")
            branch = record['scenario'].get('branch')
            if branch:
                group_splits[branch['group']].add(split)
        runs_total += len(runs)
        for run_id, records in runs.items():
            problems.extend(_run_problems(run_id, records))
    for group, splits in group_splits.items():
        if len(splits) > 1:
            problems.append(f'{group}: branches in {sorted(splits)}')
    return {'records': records_total, 'runs': runs_total,
            'problems': problems}


def _run_problems(run_id: str,
                  records: Sequence[Mapping[str, Any]]) -> list[str]:
    problems = []
    steps = [r['step'] for r in records]
    if steps != list(range(len(records))):
        problems.append(f'{run_id}: steps {steps}')
    for record in records:
        for issue in recorder.sequence_problems(record):
            problems.append(f"{run_id} step {record['step']}: {issue}")
    for previous, current in zip(records, records[1:]):
        after = observer.observation_from_dict(previous['after'])
        before = observer.observation_from_dict(current['before'])
        found = recorder.differences(recorder.canonical(after),
                                     recorder.canonical(before))
        if (after.page_epoch, after.page_version) != (
                before.page_epoch, before.page_version):
            found.append('page epoch or version changed')
        if previous['truth']['after'] != current['truth']['before']:
            found.append('backend truth changed')
        if found:
            problems.append(f"{run_id} step {current['step']}: boundary "
                            f'{found[:2]}')
    return problems


_BACK_WORDS = {w['back'] for w in scenarios.WORDINGS.values()}
_PLACEHOLDERS = {w[k] for w in scenarios.WORDINGS.values()
                 for k in ('pick_service', 'pick_doctor', 'pick_date')}


def _peak_process_rss_mib() -> float | None:
    """Returns peak Python process RSS on Linux, if available."""
    if sys.platform != 'linux' or resource is None:
        return None
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak_rss_kib / 1024, 1)


def coverage(out_dir: pathlib.Path,
             episodes: Sequence[Episode]) -> dict[str, Any]:
    """Counts behaviours, outcomes and data quality across the dataset."""
    counts: collections.Counter[str] = collections.Counter()
    per_split: collections.Counter[str] = collections.Counter()
    actions: collections.Counter[str] = collections.Counter()
    refusals = 0
    size = 0
    scenarios_seen = set()
    for split in scenarios.SPLITS:
        for name in (recorder.TRANSITIONS_FILE, recorder.REFUSALS_FILE):
            path = out_dir / split / name
            if not path.exists():
                continue
            size += path.stat().st_size
            if name == recorder.REFUSALS_FILE:
                refusals += len(recorder.read_jsonl(path))
                continue
            for record in recorder.read_jsonl(path):
                per_split[split] += 1
                actions[record['action']['kind']] += 1
                scenarios_seen.add(record['scenario']['scenario_id'])
                counts.update(_behaviours(record))
    outcomes = collections.Counter(
        f'{e.policy}:{e.goal_outcome}->{e.ended}' for e in episodes
        if e.status == COMPLETE)
    return {
        'episodes': len(episodes),
        'episodes_by_status': dict(collections.Counter(
            e.status for e in episodes)),
        'episodes_by_policy': dict(collections.Counter(
            e.policy for e in episodes)),
        'transitions': sum(per_split.values()),
        'transitions_by_split': dict(per_split),
        'transitions_by_policy': dict(sum(
            (collections.Counter({e.policy: e.steps}) for e in episodes),
            collections.Counter())),
        'unique_scenarios': len(scenarios_seen),
        'actions': dict(actions),
        'behaviours': dict(counts),
        'outcomes': dict(outcomes),
        'step_tags': dict(sum((collections.Counter(e.tags)
                                 for e in episodes), collections.Counter())),
        'branch_groups': len({e.branch['group'] for e in episodes
                              if e.branch}),
        'refused_attempts': refusals,
        'raw_bytes': size,
        # MiB of peak Python process RSS on Linux (not total browser memory).
        'max_rss_mb': _peak_process_rss_mib(),
    }


def _behaviours(record: Mapping[str, Any]) -> list[str]:
    """Tags one transition with the behaviours it shows."""
    tags = []
    before = observer.observation_from_dict(record['before'])
    after = observer.observation_from_dict(record['after'])
    action, target = record['action'], record['target']
    if not recorder.differences(recorder.canonical(before),
                                recorder.canonical(after)):
        tags.append('no_change')
    if not before.consistent or not after.consistent:
        tags.append('inconsistent')
    if not record['events_status']['complete']:
        tags.append('incomplete_events')
    target_key = target['match_key'] if target else None
    old = {e.match_key: e for e in before.elements if e.match_key}
    for element in after.elements:
        if element.match_key is None:
            continue
        prior = old.get(element.match_key)
        if (prior is not None and element.match_key != target_key
                and element.role == 'combobox'
                and prior.states['disabled'] != element.states['disabled']):
            tags.append('dependency')
            break
    if action['kind'] == executor.TYPE and target:
        prior_value = before.elements[target['index']].value
        if prior_value and prior_value != action['text']:
            tags.append('correction')
    if action['kind'] == executor.SELECT and target:
        dropdown = _dropdown_value(before, target['index'])
        if dropdown and dropdown not in _PLACEHOLDERS:
            tags.append('selection_change')
    if action['kind'] == executor.CLICK and target and (
            target['name'] in _BACK_WORDS):
        tags.append('back')
    if (action['kind'] == executor.CHECK and target
            and target['role'] == 'checkbox'):
        tags.append('consent_toggle_off' if action['checked'] is False
                    else 'consent_on')
    if action['kind'] == executor.SCROLL:
        tags.append('scroll')
    return tags


def _dropdown_value(observation: observer.Observation,
                    option_index: int) -> str | None:
    by_id = {e.node_id: e for e in observation.elements}
    node = observation.elements[option_index]
    parent = by_id.get(node.parent_id) if node.parent_id else None
    while parent is not None and parent.role != 'combobox':
        parent = by_id.get(parent.parent_id) if parent.parent_id else None
    return parent.value if parent is not None else None


def run(transitions: int, out_dir: pathlib.Path, seed_base: int,
        speed: executor.Speed, baseline_episodes: int,
        report: Callable[[str], None] = print) -> dict[str, Any]:
    """Collects into `out_dir` and returns coverage, validation and timing."""
    assignments: dict[str, booking.Scenario] = {}
    app = server.create_app(
        lambda sid: assignments.get(sid, booking.default_scenario()))
    site = live_demo.start_server(app, _free_port())
    port = site.config.port
    started = time.monotonic()
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 700})
            watcher = observer.attach(page)
            collector = Collector(page, watcher, f'http://127.0.0.1:{port}',
                                  assignments, out_dir, speed,
                                  baseline_episodes)
            collector.collect(transitions, seed_base)
            browser.close()
    finally:
        site.should_exit = True
    elapsed = time.monotonic() - started
    summary = coverage(out_dir, collector.episodes)
    summary['elapsed_s'] = round(elapsed, 1)
    summary['transitions_per_s'] = round(summary['transitions'] / elapsed, 2)
    summary['validation'] = validate_raw(out_dir)
    (out_dir / 'coverage.json').write_text(json.dumps(summary, indent=1))
    report(json.dumps(summary, indent=1))
    return summary


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--transitions', type=int, default=50)
    parser.add_argument('--out', default='data/raw/e04/pilot')
    parser.add_argument('--seed-base', type=int, default=SEED_BASE)
    parser.add_argument('--speed', choices=('fast', 'normal'), default='fast')
    parser.add_argument('--baseline-episodes', type=int, default=0)
    args = parser.parse_args()
    speed = (executor.FAST_SPEED if args.speed == 'fast'
             else executor.NORMAL_SPEED)
    run(args.transitions, pathlib.Path(args.out), args.seed_base, speed,
        args.baseline_episodes)


if __name__ == '__main__':
    main()
