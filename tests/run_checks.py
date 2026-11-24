"""Shared assertions for recorded runs."""

from __future__ import annotations

import pathlib
from typing import Any

from browser_world import recorder


def assert_one_continuous_run(raw_dir: pathlib.Path,
                              count: int) -> list[dict[str, Any]]:
    """Checks that `raw_dir` holds exactly `count` transitions forming one
    run with consecutive steps that may all enter training, and returns
    them."""
    saved = recorder.read_jsonl(raw_dir / recorder.TRANSITIONS_FILE)
    assert len(saved) == count
    assert len({r['run_id'] for r in saved}) == 1
    assert [r['step'] for r in saved] == list(range(len(saved)))
    problems = [recorder.sequence_problems(r) for r in saved]
    assert not any(problems), problems
    return saved
