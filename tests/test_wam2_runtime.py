"""Tests for WAM2 live runtime ordering, controls, and belief integrity."""

from __future__ import annotations

import pathlib
from typing import Any

import torch

from browser_world.interaction import executor
from browser_world.interaction import observer
from browser_world.wam2 import live_learning
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import runtime
from tests import wam2_helpers


class _FakeWatcher:
    def __init__(self, obs_seq: list[observer.Observation]) -> None:
        self._obs_seq = obs_seq
        self._idx = 0
        self.events: list[str] = []

    def observe(self) -> observer.Observation:
        self.events.append('observe')
        return self._obs_seq[min(self._idx, len(self._obs_seq) - 1)]

    def is_current(self, obs: observer.Observation) -> bool:
        del obs
        return True

    def start_events(self) -> None:
        pass

    def wait_until_settled(self) -> bool:
        return True

    def take_events(self) -> observer.EventTrace:
        return observer.EventTrace((), True, 0, ())


class _FakeHand:
    def __init__(self, watcher: _FakeWatcher, call_log: list[str]) -> None:
        self.pointer = (10.0, 20.0)
        self._watcher = watcher
        self._call_log = call_log

    def execute(
        self, obs: observer.Observation, action: executor.Action
    ) -> executor.ActionResult:
        self._call_log.append('execute')
        self._watcher._idx += 1
        return executor.ActionResult(
            action=action,
            executed=True,
            refusal=None,
            detail='',
            observation_sequence=obs.sequence,
            pointer_path=(),
            approach_changed_page=False,
            started_at_ms=0.0,
            duration_ms=1.0,
        )


class _FakeHost:
    def __init__(
        self, call_log: list[str], *, stop_on_fb: bool = False
    ) -> None:
        self.call_log = call_log
        self.stop_on_fb = stop_on_fb
        self.events: list[dict[str, Any]] = []
        self.messages: list[tuple[str, str, str]] = []
        self._stop = False
        self._fb_count = 0

    def is_stop_requested(self) -> bool:
        return self._stop

    def wait_if_paused(self) -> bool:
        return not self._stop

    def get_mode(self) -> str:
        return 'teach'

    def request_feedback(
        self, decision_payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        self.call_log.append('feedback')
        if self.stop_on_fb:
            self._stop = True
            return None
        self._fb_count += 1
        cands = decision_payload['candidates']
        return {
            'decision_id': decision_payload['decision_id'],
            'approved_candidate_id': cands[0]['candidate_id'],
            'rejected_candidate_ids': [],
            'literal_span': None,
            'goal_satisfied': True,
            'constraint_violated': False,
            'terminal_outcome': 'success',
        }

    def publish_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.call_log.append(f"event:{event['event_type']}")

    def add_message(self, role: str, kind: str, text: str) -> None:
        self.messages.append((role, kind, text))

    def set_observation_summary(self, obs: observer.Observation) -> None:
        del obs

    def update_timings(self, timings: dict[str, float]) -> None:
        del timings


def _build_runtime(tmp_path: pathlib.Path) -> runtime.WamRuntime:
    torch.manual_seed(0)
    role_to_idx, _ = observation.load_role_vocab()
    world = model.WorldActionModel(role_count=len(role_to_idx), slot_count=32)
    encoder = wam2_helpers.FakeSemanticEncoder()
    learner = live_learning.LiveLearner(
        world, encoder, role_to_idx, live_learning.LearningConfig()
    )
    return runtime.WamRuntime(
        world=world,
        encoder=encoder,
        role_to_index=role_to_idx,
        config=runtime.RuntimeConfig(max_actions=4),
        checkpoint_metadata={},
        learner=learner,
        experience_dir=tmp_path / 'exp',
        checkpoint_dir=tmp_path / 'ckpt',
    )


def test_runtime_order_observe_predict_execute_record_learn(
    tmp_path: pathlib.Path,
) -> None:
    rt = _build_runtime(tmp_path)
    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(10, 'textbox', 'Name', value=''),
    ])
    obs1 = wam2_helpers.make_observation([
        wam2_helpers.make_element(10, 'textbox', 'Name', value='Rahul'),
    ])
    call_log: list[str] = []
    watcher = _FakeWatcher([obs0, obs1])
    hand = _FakeHand(watcher, call_log)
    host = _FakeHost(call_log)

    outcome = rt.run_task(
        'Enter Rahul in Name',
        runtime.RuntimeMode.TEACH,
        page=object(),  # type: ignore[arg-type]
        watcher=watcher,  # type: ignore[arg-type]
        hand=hand,  # type: ignore[arg-type]
        overlay=None,
        host=host,
    )
    assert outcome == 'goal_achieved'
    pred_idx = call_log.index('event:prediction')
    fb_idx = call_log.index('feedback')
    exec_idx = call_log.index('execute')
    out_idx = call_log.index('event:actual_outcome')
    upd_idx = call_log.index('event:weight_update')
    assert pred_idx < fb_idx < exec_idx < out_idx < upd_idx
    assert rt.learner.model_revision >= 1
    assert (tmp_path / 'exp' / 'transitions.jsonl').exists()
    assert (tmp_path / 'exp' / 'feedback.jsonl').exists()
    assert (tmp_path / 'exp' / 'events.jsonl').exists()


def test_stop_cancels_pending_feedback(tmp_path: pathlib.Path) -> None:
    rt = _build_runtime(tmp_path)
    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(10, 'textbox', 'Name', value=''),
    ])
    call_log: list[str] = []
    watcher = _FakeWatcher([obs0])
    hand = _FakeHand(watcher, call_log)
    host = _FakeHost(call_log, stop_on_fb=True)

    outcome = rt.run_task(
        'Enter Rahul in Name',
        runtime.RuntimeMode.TEACH,
        page=object(),  # type: ignore[arg-type]
        watcher=watcher,  # type: ignore[arg-type]
        hand=hand,  # type: ignore[arg-type]
        overlay=None,
        host=host,
    )
    assert outcome == 'stopped'
    assert 'execute' not in call_log
