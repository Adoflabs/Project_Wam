"""Live experience-learning runtime connecting WAM2 to real Chromium."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import pathlib
import time
from typing import Any, Protocol
import uuid

from playwright import sync_api
import torch

from browser_world.collection import recorder
from browser_world.diagnostics import watch_agent
from browser_world.interaction import executor
from browser_world.interaction import observer
from browser_world.wam2 import content
from browser_world.wam2 import live_learning
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import planning


class RuntimeMode(enum.Enum):
    """Operating mode for the live WAM runtime."""

    TEACH = 'teach'
    AUTONOMOUS = 'autonomous'


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Initial engineering limits for the live WAM runtime."""

    max_actions: int = 30
    max_candidates_per_batch: int = 16
    max_ranked_candidates: int = 32
    max_copy_spans: int = 4
    max_copy_tokens: int = 32
    max_instruction_tokens: int = 256
    decision_repeats_limit: int = 2


class RuntimeHost(Protocol):
    """Thread-safe host interface implemented by WorkspaceState or tests."""

    def is_stop_requested(self) -> bool:
        """Returns True if the user requested stopping the current task."""

    def wait_if_paused(self) -> bool:
        """Blocks while paused; returns False if stopped or reset."""

    def get_mode(self) -> str:
        """Returns current runtime mode string ('teach' or 'autonomous')."""

    def request_feedback(
        self,
        decision_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publishes a decision for feedback and waits; None if interrupted."""

    def publish_event(self, event: dict[str, Any]) -> None:
        """Records a structured runtime event."""

    def add_message(self, role: str, kind: str, text: str) -> None:
        """Appends a human-readable message to the transcript."""

    def set_observation_summary(self, obs: observer.Observation) -> None:
        """Updates current observation telemetry."""

    def update_timings(self, timings: dict[str, float]) -> None:
        """Updates timing telemetry in milliseconds."""


def _append_jsonl(path: pathlib.Path, record: dict[str, Any]) -> None:
    """Appends one JSON line to `path` using explicit UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')


def _summarize_changes(
    before: observer.Observation,
    after: observer.Observation,
) -> list[str]:
    """Summarizes observable differences between two real observations."""
    changes: list[str] = []
    if before.url != after.url:
        changes.append(f'url: {before.url} -> {after.url}')
    if len(before.elements) != len(after.elements):
        changes.append(
            f'elements: {len(before.elements)} -> {len(after.elements)}'
        )
    before_vals = {
        (e.role, e.name): e.value
        for e in before.elements
        if e.value is not None
    }
    for elem in after.elements:
        if elem.value is not None:
            prev = before_vals.get((elem.role, elem.name))
            if prev != elem.value:
                changes.append(
                    f'{elem.role} "{elem.name}": {prev!r} -> {elem.value!r}'
                )
    return changes[:8]


def _detach_belief(belief: model.Belief) -> model.Belief:
    """Returns a detached copy of `belief`."""
    return model.Belief(
        latent=belief.latent.detach(),
        present=belief.present.detach(),
        states=belief.states.detach(),
        roles=belief.roles.detach(),
        flags=belief.flags.detach(),
        boxes=belief.boxes.detach(),
        affordances=belief.affordances.detach(),
        parents=belief.parents.detach(),
        context=belief.context.detach(),
    )


def _format_candidate_summary(
    pred: planning.CandidatePrediction,
    obs: observer.Observation,
) -> dict[str, Any]:
    """Formats a CandidatePrediction as a JSON-ready UI summary."""
    cand = pred.candidate
    target_name = 'page'
    if cand.action.target is not None:
        for elem in obs.elements:
            if elem.node_id == cand.action.target:
                target_name = f'{elem.role}: {elem.name}'
                break
    return {
        'candidate_id': cand.candidate_id,
        'kind': cand.action.kind,
        'target_node_id': cand.action.target,
        'target_slot': cand.target_slot,
        'target_name': target_name,
        'text': cand.action.text,
        'checked': cand.action.checked,
        'scroll_y': cand.action.scroll_y,
        'instruction_span': (
            list(cand.instruction_span)
            if cand.instruction_span is not None
            else None
        ),
        'proposal_logit': cand.proposal_logit,
        'satisfaction_score': pred.satisfaction_score,
        'return_score': pred.return_score,
        'disagreement': pred.disagreement,
        'decision_score': pred.decision_score,
    }


def _decode_top_copy_spans(
    instruction: str,
    copy_logits: torch.Tensor,
    present: torch.Tensor,
    token_offsets: torch.Tensor,
    token_mask: torch.Tensor,
    max_tokens: int,
    top_k: int,
) -> tuple[tuple[int, int], ...]:
    """Decodes ranked unique copy spans across present entity slots."""
    seq_len = copy_logits.shape[-1]
    if seq_len == 0 or not bool(token_mask.any().item()):
        return ()

    present_slots = [
        i for i in range(present.shape[1]) if float(present[0, i].item()) > 0.5
    ]
    if not present_slots:
        return ()

    # Pool max start/end logits across present slots to surface candidate spans
    # without inventing any text.
    slot_idx = torch.tensor(
        present_slots, dtype=torch.long, device=copy_logits.device
    )
    pooled_start = copy_logits[0, slot_idx, 0, :].max(dim=0).values
    pooled_end = copy_logits[0, slot_idx, 1, :].max(dim=0).values
    return planning.decode_copy_spans(
        instruction=instruction,
        start_logits=pooled_start,
        end_logits=pooled_end,
        token_offsets=token_offsets,
        token_mask=token_mask[0],
        max_tokens=max_tokens,
        top_k=top_k,
    )


class WamRuntime:
    """Runs the existing WAM against a real browser environment."""

    def __init__(
        self,
        *,
        world: model.WorldActionModel,
        encoder: observation.TextEncoder,
        role_to_index: dict[str, int],
        config: RuntimeConfig,
        checkpoint_metadata: dict[str, Any],
        learner: live_learning.LiveLearner,
        experience_dir: pathlib.Path,
        checkpoint_dir: pathlib.Path,
    ) -> None:
        self.world = world
        self.encoder = encoder
        self.role_to_index = dict(role_to_index)
        self.config = config
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.learner = learner
        self.experience_dir = experience_dir
        self.checkpoint_dir = checkpoint_dir

        self.current_obs: observer.Observation | None = None
        self.current_aligned: observation.AlignedObservation | None = None
        self.current_belief: model.Belief | None = None
        self.exact_content: tuple[content.EntityContent, ...] = ()
        self.current_instruction: str | None = None
        self._cached_tokens: torch.Tensor | None = None
        self._cached_mask: torch.Tensor | None = None
        self._cached_offsets: torch.Tensor | None = None
        self.pointer: tuple[float, float] = (0.0, 0.0)

        self._recorder: recorder.Recorder | None = None
        self._run_id: str | None = None
        self._last_page_ref: sync_api.Page | None = None

    def reset_session(self) -> None:
        """Clears browser-tied belief and recorder state on reset."""
        self.current_obs = None
        self.current_aligned = None
        self.current_belief = None
        self.exact_content = ()
        self.pointer = (0.0, 0.0)
        self._recorder = None
        self._run_id = None

    def encode_instruction(
        self,
        instruction: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes `instruction` with mmBERT, caching unchanged instructions."""
        if (
            self.current_instruction == instruction
            and self._cached_tokens is not None
            and self._cached_mask is not None
            and self._cached_offsets is not None
        ):
            return (
                self._cached_tokens,
                self._cached_mask,
                self._cached_offsets,
            )

        tokens, mask, offsets = self.encoder.encode(
            (instruction,), max_tokens=self.config.max_instruction_tokens
        )
        self.current_instruction = instruction
        self._cached_tokens = tokens
        self._cached_mask = mask
        self._cached_offsets = offsets[0]
        return tokens, mask, offsets[0]

    def _ensure_recorder(
        self,
        page: sync_api.Page,
        watcher: observer.Observer,
        hand: executor.Executor,
        episode_id: str,
    ) -> tuple[recorder.Recorder, str]:
        """Initializes Recorder and run_id for the active page."""
        if self._recorder is None or self._last_page_ref is not page:
            self._recorder = recorder.Recorder(
                page, watcher, hand, self.experience_dir
            )
            self._last_page_ref = page
            self._run_id = None
        if self._run_id is None:
            self._run_id = self._recorder.new_run(episode_id, origin='start')
        return self._recorder, self._run_id

    def _record_event(
        self,
        host: RuntimeHost,
        event: dict[str, Any],
    ) -> None:
        """Persists and publishes a structured runtime event."""
        enriched = dict(event)
        enriched.setdefault('timestamp_ms', round(time.time() * 1000, 2))
        _append_jsonl(self.experience_dir / 'events.jsonl', enriched)
        host.publish_event(enriched)

    def run_task(
        self,
        instruction: str,
        mode: RuntimeMode,
        *,
        page: sync_api.Page,
        watcher: observer.Observer,
        hand: executor.Executor,
        overlay: watch_agent.Overlay | None,
        host: RuntimeHost,
    ) -> str:
        """Runs `instruction` via observation, prediction, and learning."""
        try:
            inst_tokens, inst_mask, tok_offsets = self.encode_instruction(
                instruction
            )
        except ValueError as exc:
            host.add_message('wam', 'error', str(exc))
            return 'unsupported_instruction'

        episode_id = f'live-{uuid.uuid4().hex[:8]}'
        rec, run_id = self._ensure_recorder(page, watcher, hand, episode_id)
        episode_samples: list[live_learning.LearningSample] = []
        last_signature: tuple[str, int | None, str | None, str] | None = None
        repeat_count = 0
        self.pointer = hand.pointer

        for _ in range(self.config.max_actions):
            if host.is_stop_requested() or not host.wait_if_paused():
                return 'stopped'

            obs_start = time.perf_counter()
            obs = watcher.observe()
            obs_ms = round((time.perf_counter() - obs_start) * 1000, 2)
            host.set_observation_summary(obs)

            try:
                aligned = observation.encode_observation(
                    obs,
                    self.encoder,
                    self.role_to_index,
                    slot_count=self.world.slot_count,
                    previous=self.current_aligned,
                )
            except ValueError as exc:
                host.add_message('wam', 'error', f'Slot capacity error: {exc}')
                return 'stopped'

            self.world.eval()
            with torch.inference_mode():
                belief = observation.observe(
                    self.world, aligned, previous=self.current_belief
                )
                belief = _detach_belief(belief)
            self.current_obs = obs
            self.current_aligned = aligned
            self.current_belief = belief
            self.exact_content = aligned.exact_content

            inf_start = time.perf_counter()
            with torch.inference_mode():
                goal_logits, proposal_logits, copy_logits = (
                    self.world.evaluate_goal(belief, inst_tokens, inst_mask)
                )
            if not (
                torch.isfinite(goal_logits).all().item()
                and torch.isfinite(proposal_logits).all().item()
                and torch.isfinite(copy_logits).all().item()
            ):
                host.add_message(
                    'wam', 'error', 'Non-finite model outputs encountered.'
                )
                return 'failed'

            copy_spans = _decode_top_copy_spans(
                instruction,
                copy_logits,
                aligned.inputs.present,
                tok_offsets,
                inst_mask,
                max_tokens=self.config.max_copy_tokens,
                top_k=self.config.max_copy_spans,
            )
            affordances = tuple(executor.available_actions(obs))
            candidates = planning.enumerate_candidates(
                aligned,
                instruction,
                copy_spans,
                affordances,
                proposal_logits=proposal_logits,
            )
            if not candidates and not affordances:
                host.add_message('wam', 'error', 'No legal candidates found.')
                return 'stopped'

            predictions = planning.predict_candidates(
                self.world,
                belief,
                candidates[:self.config.max_ranked_candidates],
                aligned,
                self.encoder,
                batch_size=self.config.max_candidates_per_batch,
                pointer=self.pointer,
            )
            cap = self.learner.capability_status()
            predictions = planning.score_predictions(
                self.world,
                predictions,
                inst_tokens,
                inst_mask,
                satisfaction_trained=bool(cap['satisfaction_supervised']),
                return_trained=bool(cap['return_supervised']),
                batch_size=self.config.max_candidates_per_batch,
            )
            inf_ms = round((time.perf_counter() - inf_start) * 1000, 2)
            host.update_timings({'observe_ms': obs_ms, 'inference_ms': inf_ms})

            model_pick = planning.select_candidate(
                predictions,
                allow_proposal_only=bool(cap['proposal_supervised']),
            )
            # In TEACH mode, surface the highest raw proposal_logit as an
            # explicitly untrained recommendation when no supervision exists.
            display_pick = model_pick or planning.select_candidate(
                predictions, allow_proposal_only=True
            )

            decision_id = f'decision-{uuid.uuid4().hex[:8]}'
            self._record_event(
                host,
                {
                    'event_type': 'prediction',
                    'decision_id': decision_id,
                    'model_revision': self.learner.model_revision,
                    'observation_digest': obs.state_digest,
                    'candidate_count': len(predictions),
                    'model_selected_candidate': (
                        model_pick.candidate.candidate_id
                        if model_pick is not None
                        else None
                    ),
                    'recommended_candidate': (
                        display_pick.candidate.candidate_id
                        if display_pick is not None
                        else None
                    ),
                    'prediction_completed_before_execution': True,
                    'capabilities': cap,
                },
            )

            active_mode = RuntimeMode(host.get_mode())
            feedback: live_learning.ActionFeedback | None = None
            selected_pred: planning.CandidatePrediction | None = None
            selection_source = 'model_selected'

            if active_mode == RuntimeMode.TEACH:
                payload = {
                    'decision_id': decision_id,
                    'instruction': instruction,
                    'model_revision': self.learner.model_revision,
                    'observation_digest': obs.state_digest,
                    'recommended_candidate_id': (
                        display_pick.candidate.candidate_id
                        if display_pick is not None
                        else None
                    ),
                    'capabilities': cap,
                    'candidates': [
                        _format_candidate_summary(p, obs) for p in predictions
                    ],
                    'affordances': [
                        {
                            'index': idx,
                            'kind': aff.kind,
                            'target': aff.target,
                            'role': aff.role,
                            'name': aff.name,
                            'checked': aff.checked,
                            'scroll_y': aff.scroll_y,
                        }
                        for idx, aff in enumerate(affordances)
                    ],
                }
                raw_fb = host.request_feedback(payload)
                if raw_fb is None or host.is_stop_requested():
                    return 'stopped'

                feedback = live_learning.ActionFeedback(
                    decision_id=decision_id,
                    approved_candidate_id=raw_fb.get('approved_candidate_id'),
                    rejected_candidate_ids=tuple(
                        raw_fb.get('rejected_candidate_ids') or ()
                    ),
                    literal_span=(
                        tuple(raw_fb['literal_span'])
                        if raw_fb.get('literal_span') is not None
                        else None
                    ),
                    goal_satisfied=raw_fb.get('goal_satisfied'),
                    constraint_violated=raw_fb.get('constraint_violated'),
                    terminal_outcome=raw_fb.get('terminal_outcome'),
                )
                valid_ids = {p.candidate.candidate_id for p in predictions}
                live_learning.validate_feedback(
                    feedback,
                    valid_candidate_ids=valid_ids,
                    instruction_length=len(instruction),
                )

                _append_jsonl(
                    self.experience_dir / 'feedback.jsonl',
                    {
                        'decision_id': decision_id,
                        'episode_id': episode_id,
                        'model_revision': self.learner.model_revision,
                        'feedback': dataclasses.asdict(feedback),
                    },
                )

                if feedback.approved_candidate_id is None:
                    if feedback.terminal_outcome in ('success', 'failure'):
                        ret_rep = self.learner.learn_terminal_returns(
                            episode_samples, feedback.terminal_outcome
                        )
                        if ret_rep is not None:
                            self.learner.save_checkpoint(self.checkpoint_dir)
                            self._record_event(
                                host,
                                {
                                    'event_type': 'weight_update',
                                    **dataclasses.asdict(ret_rep),
                                },
                            )
                        return (
                            'goal_achieved'
                            if feedback.terminal_outcome == 'success'
                            else 'goal_not_achieved'
                        )
                    return 'stopped'

                chosen = next(
                    p
                    for p in predictions
                    if p.candidate.candidate_id
                    == feedback.approved_candidate_id
                )
                if (
                    feedback.literal_span is not None
                    and chosen.candidate.action.kind == executor.TYPE
                    and chosen.candidate.instruction_span
                    != feedback.literal_span
                ):
                    s, e = feedback.literal_span
                    corrected_cand = dataclasses.replace(
                        chosen.candidate,
                        action=dataclasses.replace(
                            chosen.candidate.action, text=instruction[s:e]
                        ),
                        instruction_span=feedback.literal_span,
                    )
                    selected_pred = planning.predict_candidates(
                        self.world,
                        belief,
                        (corrected_cand,),
                        aligned,
                        self.encoder,
                        batch_size=1,
                        pointer=self.pointer,
                    )[0]
                    selection_source = 'human_corrected_literal'
                elif (
                    display_pick is not None
                    and chosen.candidate.candidate_id
                    == display_pick.candidate.candidate_id
                ):
                    selected_pred = chosen
                    selection_source = 'human_approved_model_proposal'
                else:
                    selected_pred = chosen
                    selection_source = 'human_corrected_target'
            else:
                block_reason = self.learner.autonomous_block_reason()
                if block_reason is not None:
                    host.add_message('wam', 'error', block_reason)
                    return 'stopped'
                if model_pick is None:
                    host.add_message(
                        'wam',
                        'error',
                        'No supported autonomous candidate available.',
                    )
                    return 'stopped'
                selected_pred = model_pick
                selection_source = (
                    'autonomous_predictive'
                    if selected_pred.decision_score is not None
                    else 'autonomous_proposal_only'
                )

            if not watcher.is_current(obs):
                host.add_message(
                    'wam',
                    'status',
                    'Observation became stale before execution; re-observing.',
                )
                continue

            exec_action = planning.resolve_candidate_action(
                selected_pred.candidate, aligned
            )
            if exec_action is None:
                host.add_message(
                    'wam',
                    'error',
                    'Selected target slot could not be grounded to real page.',
                )
                return 'stopped'

            sig = (
                exec_action.kind,
                selected_pred.candidate.target_slot,
                exec_action.text,
                obs.state_digest,
            )
            if sig == last_signature:
                repeat_count += 1
                if repeat_count >= self.config.decision_repeats_limit:
                    host.add_message(
                        'wam',
                        'status',
                        'Stopped: repeated decision without progress.',
                    )
                    return 'stopped'
            else:
                last_signature = sig
                repeat_count = 0

            if overlay is not None and exec_action.target is not None:
                overlay.target(exec_action.target)

            exec_start = time.perf_counter()
            step_outcome = rec.record_step(run_id, obs, exec_action)
            exec_ms = round((time.perf_counter() - exec_start) * 1000, 2)
            host.update_timings({'execute_ms': exec_ms})
            self.pointer = hand.pointer

            if not isinstance(step_outcome, recorder.Transition):
                self._record_event(
                    host,
                    {
                        'event_type': 'actual_outcome',
                        'decision_id': decision_id,
                        'executed_candidate_id': (
                            selected_pred.candidate.candidate_id
                        ),
                        'selection_source': selection_source,
                        'execution_succeeded': False,
                        'refusal': str(step_outcome.refusal),
                    },
                )
                host.add_message(
                    'wam',
                    'error',
                    f'Executor refused action: {step_outcome.detail}',
                )
                continue

            after_obs = step_outcome.after
            host.set_observation_summary(after_obs)
            after_aligned = observation.encode_observation(
                after_obs,
                self.encoder,
                self.role_to_index,
                slot_count=self.world.slot_count,
                previous=aligned,
            )
            self.world.eval()
            with torch.inference_mode():
                actual_after_belief = observation.observe(
                    self.world, after_aligned, previous=belief
                )
                act_in = observation.encode_action(
                    exec_action, aligned, self.encoder, self.pointer
                )
                act_lat = self.world.encode_action(
                    belief,
                    act_in.kind,
                    act_in.target,
                    act_in.text,
                    act_in.checked,
                    act_in.scroll,
                    act_in.pointer,
                )
                known_mask = (
                    after_aligned.inputs.existence_known
                    & (
                        after_aligned.inputs.matched
                        | (
                            (aligned.inputs.present <= 0.5)
                            & (after_aligned.inputs.present <= 0.5)
                        )
                    )
                )
                adapted_belief = self.world.correct_context(
                    actual_after_belief,
                    selected_pred.prediction.belief,
                    act_lat,
                    known_mask,
                )
                self.current_belief = _detach_belief(adapted_belief)
            self.current_obs = after_obs
            self.current_aligned = after_aligned
            self.exact_content = after_aligned.exact_content

            changes = _summarize_changes(obs, after_obs)
            transition_id = f'{step_outcome.run_id}#step{step_outcome.step}'
            self._record_event(
                host,
                {
                    'event_type': 'actual_outcome',
                    'decision_id': decision_id,
                    'transition_id': transition_id,
                    'executed_candidate_id': (
                        selected_pred.candidate.candidate_id
                    ),
                    'selection_source': selection_source,
                    'execution_succeeded': True,
                    'observed_changes': changes,
                    'context_updated': True,
                },
            )

            sample = live_learning.LearningSample(
                sample_id=transition_id,
                decision_id=decision_id,
                before_aligned=aligned,
                after_aligned=after_aligned,
                executed_candidate=selected_pred.candidate,
                candidates=tuple(p.candidate for p in predictions),
                pointer=self.pointer,
                instruction=instruction,
                instruction_tokens=inst_tokens.detach(),
                instruction_mask=inst_mask.detach(),
                token_offsets=tok_offsets.detach(),
                feedback=feedback,
                previous_belief=belief,
                context=belief.context.detach(),
            )
            episode_samples.append(sample)

            update_report = self.learner.learn_from_sample(sample)
            if update_report.status == 'weight_update_accepted':
                self.learner.save_checkpoint(self.checkpoint_dir)
            self._record_event(
                host,
                {
                    'event_type': 'weight_update',
                    **dataclasses.asdict(update_report),
                },
            )

            if feedback is not None and feedback.terminal_outcome in (
                'success',
                'failure',
            ):
                ret_rep = self.learner.learn_terminal_returns(
                    episode_samples, feedback.terminal_outcome
                )
                if ret_rep is not None:
                    self.learner.save_checkpoint(self.checkpoint_dir)
                    self._record_event(
                        host,
                        {
                            'event_type': 'weight_update',
                            **dataclasses.asdict(ret_rep),
                        },
                    )
                return (
                    'goal_achieved'
                    if feedback.terminal_outcome == 'success'
                    else 'goal_not_achieved'
                )

        host.add_message(
            'wam',
            'status',
            f'Action budget ({self.config.max_actions}) reached.',
        )
        return 'stopped'
