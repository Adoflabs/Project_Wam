"""Split-screen WAM research workspace and visible Chromium launcher.

Left window: WAM conversation transcript and natural-language composer.
Right window: Real Chromium browser running the booking application.

Launch from the repository root:
  .\\.venv\\Scripts\\python.exe -m browser_world.diagnostics.workspace
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import ctypes
from ctypes import wintypes
import hashlib
import pathlib
import threading
import time
from typing import Any
import uuid

from playwright import sync_api
from starlette import applications
from starlette import requests
from starlette import responses
from starlette import routing
import torch

from browser_world.diagnostics import live_demo
from browser_world.diagnostics import watch_agent
from browser_world.environment import booking
from browser_world.environment import scenarios
from browser_world.environment import server
from browser_world.interaction import executor
from browser_world.interaction import observer
from browser_world.wam2 import content
from browser_world.wam2 import live_learning
from browser_world.wam2 import observation
from browser_world.wam2 import runtime
from browser_world.wam2 import semantic
from browser_world.wam2 import training_data

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WORKSPACE_HTML = pathlib.Path(__file__).parent / 'static' / 'workspace.html'
DEFAULT_BASE_CKPT = (
    _REPO_ROOT / 'artifacts' / 'wam2' / 'dynamics_v2_corrected' / 'best.pt'
)
DEFAULT_LIVE_DIR = _REPO_ROOT / 'artifacts' / 'wam2' / 'live_learning_v1'
DEFAULT_EXPERIENCE_DIR = _REPO_ROOT / 'data' / 'raw' / 'wam2_live_v1'
MAX_INSTRUCTION_BYTES = 1024
UNINITIALIZED_REASON = 'New WAM runtime not initialized'


def _desktop_work_area() -> tuple[int, int, int, int]:
    """Returns (left, top, width, height) of the usable desktop work area."""
    try:
        rect = wintypes.RECT()
        ok = ctypes.windll.user32.SystemParametersInfoW(
            48, 0, ctypes.byref(rect), 0
        )
        if ok:
            w = max(800, rect.right - rect.left)
            h = max(600, rect.bottom - rect.top)
            return int(rect.left), int(rect.top), int(w), int(h)
    except (AttributeError, OSError):
        pass
    return 0, 0, 1920, 1040


def place_window_cdp(
    page: sync_api.Page,
    left: int,
    top: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Places a visible Chromium window via CDP and returns target & bounds."""
    cdp = page.context.new_cdp_session(page)
    win = cdp.send('Browser.getWindowForTarget')
    window_id = win['windowId']
    cdp.send(
        'Browser.setWindowBounds',
        {
            'windowId': window_id,
            'bounds': {
                'left': int(left),
                'top': int(top),
                'width': int(width),
                'height': int(height),
                'windowState': 'normal',
            },
        },
    )
    actual = cdp.send('Browser.getWindowBounds', {'windowId': window_id})
    target = cdp.send('Target.getTargetInfo')
    cdp.detach()
    return {
        'windowId': window_id,
        'targetId': target.get('targetInfo', {}).get('targetId', ''),
        'bounds': actual.get('bounds', {}),
    }


def check_model_readiness(
    *,
    task_runner: Callable[..., Any] | None = None,
    wam_runtime: runtime.WamRuntime | None = None,
) -> dict[str, Any]:
    """Checks whether the new WAM runtime is connected."""
    nl_ready = callable(task_runner)
    reason = 'All runtimes ready' if nl_ready else UNINITIALIZED_REASON
    if wam_runtime is None:
        return {
            'ready': nl_ready,
            'nl_ready': nl_ready,
            'ui_available': True,
            'reason': reason,
            'dynamics_loaded': False,
            'teaching_ready': False,
            'decision_supervision_available': False,
            'experimental_autonomous_enabled': False,
        }
    cap = wam_runtime.learner.capability_status()
    param_count = sum(p.numel() for p in wam_runtime.world.parameters())
    return {
        'ready': nl_ready,
        'nl_ready': nl_ready,
        'ui_available': True,
        'reason': reason,
        'dynamics_loaded': True,
        'teaching_ready': True,
        'decision_supervision_available': bool(cap['proposal_supervised']),
        'experimental_autonomous_enabled': False,
        'param_count': param_count,
        'capabilities': cap,
    }


def validate_instruction(payload: Any) -> tuple[str | None, str | None]:
    """Validates POST /api/run instruction without modifying or truncating."""
    if not isinstance(payload, Mapping):
        return None, 'Request body must be a JSON object.'
    instruction = payload.get('instruction')
    if not isinstance(instruction, str):
        return None, 'Field "instruction" must be a string.'
    if not instruction.strip():
        return None, 'Instruction cannot be empty.'
    if len(instruction.encode('utf-8')) > MAX_INSTRUCTION_BYTES:
        return None, (
            f'Instruction exceeds {MAX_INSTRUCTION_BYTES} UTF-8 bytes.'
        )
    return instruction, None


class WorkspaceState:
    """Thread-safe shared state between the Starlette API and browser thread."""

    def __init__(
        self,
        readiness: dict[str, Any],
        workspace_url: str,
        environment_url: str,
        *,
        autonomous_check: Callable[[], str | None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._autonomous_check = autonomous_check
        self.readiness = readiness
        self.workspace_url = workspace_url
        self.environment_url = environment_url
        self.execution_status = 'idle'
        self.mode = runtime.RuntimeMode.TEACH.value
        self.current_instruction: str | None = None
        self.runtime_messages: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.pending_decision: dict[str, Any] | None = None
        self._pending_feedback: dict[str, Any] | None = None
        self.observation_id = ''
        self.observable_entities: list[dict[str, Any]] = []
        self.error: str | None = None
        self.timings: dict[str, float] = {}
        self.run_requested = False
        self.paused = False
        self.stop_requested = False
        self.reset_requested = False

    def snapshot(self) -> dict[str, Any]:
        """Returns a JSON-serializable copy of workspace state."""
        with self._lock:
            return {
                'readiness': dict(self.readiness),
                'workspace_url': self.workspace_url,
                'environment_url': self.environment_url,
                'execution_status': self.execution_status,
                'mode': self.mode,
                'current_instruction': self.current_instruction,
                'runtime_messages': list(self.runtime_messages),
                'events': list(self.events[-50:]),
                'pending_decision': (
                    dict(self.pending_decision)
                    if self.pending_decision is not None
                    else None
                ),
                'observation_id': self.observation_id,
                'observable_entities': list(self.observable_entities),
                'error': self.error,
                'timings': dict(self.timings),
            }

    def set_readiness(self, readiness: dict[str, Any]) -> None:
        """Updates readiness telemetry."""
        with self._lock:
            self.readiness = dict(readiness)

    def set_observation_summary(self, obs: observer.Observation) -> None:
        """Updates current observation ID and entity list for telemetry."""
        entities = [
            {
                'index': idx,
                'role': elem.role,
                'name': elem.name,
                'states': dict(elem.states),
            }
            for idx, elem in enumerate(obs.elements)
        ]
        with self._lock:
            self.observation_id = obs.state_digest
            self.observable_entities = entities

    def update_timings(self, timings: dict[str, float]) -> None:
        """Merges timing telemetry in milliseconds."""
        with self._lock:
            self.timings.update(timings)

    def add_message(self, role: str, kind: str, text: str) -> None:
        """Appends a status or error message to the transcript."""
        with self._lock:
            self.runtime_messages.append({
                'id': uuid.uuid4().hex[:10],
                'role': role,
                'kind': kind,
                'text': text,
                'timestamp': time.time(),
            })

    def publish_event(self, event: dict[str, Any]) -> None:
        """Records a structured runtime event."""
        with self._lock:
            self.events.append(dict(event))

    def is_stop_requested(self) -> bool:
        """Returns True if the user requested stopping the current task."""
        with self._lock:
            return self.stop_requested or self.reset_requested

    def wait_if_paused(self) -> bool:
        """Blocks while paused; returns False if stopped or reset."""
        with self._cond:
            while self.paused and not (
                self.stop_requested or self.reset_requested
            ):
                self._cond.wait(timeout=0.1)
            return not (self.stop_requested or self.reset_requested)

    def get_mode(self) -> str:
        """Returns current runtime mode string."""
        with self._lock:
            return self.mode

    def set_mode(self, mode: str) -> tuple[bool, int, str | None]:
        """Switches between teach and experimental autonomous mode."""
        if mode not in (
            runtime.RuntimeMode.TEACH.value,
            runtime.RuntimeMode.AUTONOMOUS.value,
        ):
            return False, 400, f'Unsupported mode {mode!r}.'
        if mode == runtime.RuntimeMode.AUTONOMOUS.value:
            if self._autonomous_check is not None:
                reason = self._autonomous_check()
                if reason is not None:
                    return False, 409, reason
        with self._lock:
            self.mode = mode
            self.readiness['experimental_autonomous_enabled'] = (
                mode == runtime.RuntimeMode.AUTONOMOUS.value
            )
            return True, 200, None

    def request_feedback(
        self,
        decision_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Publishes a pending decision and waits for human supervision."""
        with self._cond:
            if self.stop_requested or self.reset_requested:
                return None
            self.pending_decision = dict(decision_payload)
            self._pending_feedback = None
            while (
                self._pending_feedback is None
                and not self.stop_requested
                and not self.reset_requested
            ):
                self._cond.wait(timeout=0.1)
            result = self._pending_feedback
            self.pending_decision = None
            self._pending_feedback = None
            if self.stop_requested or self.reset_requested:
                return None
            return result

    def submit_feedback(
        self,
        payload: Any,
    ) -> tuple[bool, int, str | None]:
        """Validates and delivers human feedback for the pending decision."""
        if not isinstance(payload, Mapping):
            return False, 400, 'Feedback payload must be a JSON object.'
        decision_id = payload.get('decision_id')
        if not isinstance(decision_id, str) or not decision_id:
            return False, 400, 'Field "decision_id" must be a string.'

        with self._cond:
            pending = self.pending_decision
            if pending is None:
                return False, 409, 'No pending decision awaits feedback.'
            if pending.get('decision_id') != decision_id:
                return False, 409, 'Stale decision_id.'

            valid_ids = {
                int(c['candidate_id']) for c in pending.get('candidates', [])
            }
            approved = payload.get('approved_candidate_id')
            if approved is not None and (
                not isinstance(approved, int)
                or isinstance(approved, bool)
                or approved not in valid_ids
            ):
                return False, 400, 'Invalid approved_candidate_id.'

            raw_rejected = payload.get('rejected_candidate_ids', [])
            if not isinstance(raw_rejected, list):
                return False, 400, 'rejected_candidate_ids must be a list.'
            rejected: list[int] = []
            for cid in raw_rejected:
                if (
                    not isinstance(cid, int)
                    or isinstance(cid, bool)
                    or cid not in valid_ids
                ):
                    return False, 400, 'Invalid rejected_candidate_id.'
                rejected.append(cid)

            raw_span = payload.get('literal_span')
            span: list[int] | None = None
            if raw_span is not None:
                inst = pending.get('instruction', '')
                if (
                    not isinstance(raw_span, list)
                    or len(raw_span) != 2
                    or not all(
                        isinstance(v, int) and not isinstance(v, bool)
                        for v in raw_span
                    )
                    or raw_span[0] < 0
                    or raw_span[1] <= raw_span[0]
                    or raw_span[1] > len(inst)
                ):
                    return False, 400, 'Invalid literal_span.'
                span = [raw_span[0], raw_span[1]]

            goal_sat = payload.get('goal_satisfied')
            if goal_sat is not None and not isinstance(goal_sat, bool):
                return False, 400, 'goal_satisfied must be bool or null.'
            const_viol = payload.get('constraint_violated')
            if const_viol is not None and not isinstance(const_viol, bool):
                return False, 400, 'constraint_violated must be bool or null.'

            term = payload.get('terminal_outcome')
            if (
                term is not None
                and term not in live_learning.TERMINAL_OUTCOMES
            ):
                return False, 400, 'Invalid terminal_outcome.'

            if approved is not None and approved in rejected:
                return (
                    False,
                    400,
                    'approved_candidate_id overlaps rejected_candidate_ids.',
                )

            self._pending_feedback = {
                'decision_id': decision_id,
                'approved_candidate_id': approved,
                'rejected_candidate_ids': rejected,
                'literal_span': span,
                'goal_satisfied': goal_sat,
                'constraint_violated': const_viol,
                'terminal_outcome': term,
            }
            self._cond.notify_all()
            return True, 200, None

    def enqueue_run(self, instruction: str) -> tuple[bool, int, str | None]:
        """Queues a natural-language task if the WAM runtime is ready."""
        with self._cond:
            if not self.readiness.get('nl_ready'):
                reason = self.readiness.get('reason', UNINITIALIZED_REASON)
                self.execution_status = 'unsupported_instruction'
                self.error = reason
                return False, 503, reason
            active = (
                self.execution_status in ('running', 'paused')
                or self.run_requested
            )
            if active:
                return (
                    False,
                    409,
                    'A task is running. Stop it before starting a new task.',
                )
            self.current_instruction = instruction
            self.error = None
            self.paused = False
            self.stop_requested = False
            self.run_requested = True
            self.runtime_messages.append({
                'id': uuid.uuid4().hex[:10],
                'role': 'user',
                'kind': 'instruction',
                'text': instruction,
                'timestamp': time.time(),
            })
            self._cond.notify_all()
            return True, 200, None

    def enqueue_pause(self) -> tuple[bool, str | None]:
        """Pauses an active task."""
        with self._cond:
            if self.execution_status != 'running':
                return (
                    False,
                    f'Cannot pause while status is {self.execution_status}.',
                )
            self.paused = True
            self.execution_status = 'paused'
            self._cond.notify_all()
            return True, None

    def enqueue_resume(self) -> tuple[bool, str | None]:
        """Resumes a paused task."""
        with self._cond:
            if self.execution_status != 'paused':
                return (
                    False,
                    f'Cannot resume while status is {self.execution_status}.',
                )
            self.paused = False
            self.execution_status = 'running'
            self._cond.notify_all()
            return True, None

    def enqueue_stop(self) -> tuple[bool, str | None]:
        """Stops a running or queued task."""
        with self._cond:
            active = (
                self.execution_status in ('running', 'paused')
                or self.run_requested
            )
            if not active:
                return (
                    False,
                    f'No active task to stop (status={self.execution_status}).',
                )
            self.stop_requested = True
            self.paused = False
            self.run_requested = False
            self.pending_decision = None
            self._pending_feedback = None
            self.execution_status = 'stopped'
            self._cond.notify_all()
            return True, None

    def enqueue_reset(self) -> tuple[bool, str | None]:
        """Resets conversation state and browser environment."""
        with self._cond:
            active = (
                self.execution_status in ('running', 'paused')
                or self.run_requested
            )
            if active:
                return False, 'Stop the active task before resetting.'
            self.stop_requested = False
            self.paused = False
            self.run_requested = False
            self.reset_requested = True
            self.pending_decision = None
            self._pending_feedback = None
            self.current_instruction = None
            self.runtime_messages.clear()
            self.events.clear()
            self.error = None
            self.timings = {}
            self.execution_status = 'idle'
            self._cond.notify_all()
            return True, None


def create_workspace_app(state: WorkspaceState) -> applications.Starlette:
    """Builds Starlette routes for the WAM research workspace."""

    async def index(request: requests.Request) -> responses.Response:
        del request
        html = _WORKSPACE_HTML.read_text(encoding='utf-8')
        return responses.HTMLResponse(html)

    async def api_state(request: requests.Request) -> responses.Response:
        del request
        return responses.JSONResponse(state.snapshot())

    async def api_run(request: requests.Request) -> responses.Response:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return responses.JSONResponse(
                {'error': 'Invalid JSON payload.'}, status_code=400
            )
        instruction, err = validate_instruction(body)
        if err is not None or instruction is None:
            return responses.JSONResponse({'error': err}, status_code=400)
        ok, status_code, run_err = state.enqueue_run(instruction)
        if not ok:
            return responses.JSONResponse(
                {'error': run_err}, status_code=status_code
            )
        return responses.JSONResponse({'ok': True, 'instruction': instruction})

    async def api_feedback(request: requests.Request) -> responses.Response:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return responses.JSONResponse(
                {'error': 'Invalid JSON payload.'}, status_code=400
            )
        ok, status_code, err = state.submit_feedback(body)
        if not ok:
            return responses.JSONResponse(
                {'error': err}, status_code=status_code
            )
        return responses.JSONResponse({'ok': True})

    async def api_mode(request: requests.Request) -> responses.Response:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return responses.JSONResponse(
                {'error': 'Invalid JSON payload.'}, status_code=400
            )
        if not isinstance(body, Mapping):
            return responses.JSONResponse(
                {'error': 'Request body must be a JSON object.'},
                status_code=400,
            )
        ok, status_code, err = state.set_mode(str(body.get('mode', '')))
        if not ok:
            return responses.JSONResponse(
                {'error': err}, status_code=status_code
            )
        return responses.JSONResponse({'ok': True, 'mode': state.get_mode()})

    async def api_pause(request: requests.Request) -> responses.Response:
        del request
        ok, err = state.enqueue_pause()
        if not ok:
            return responses.JSONResponse({'error': err}, status_code=409)
        return responses.JSONResponse({'ok': True})

    async def api_resume(request: requests.Request) -> responses.Response:
        del request
        ok, err = state.enqueue_resume()
        if not ok:
            return responses.JSONResponse({'error': err}, status_code=409)
        return responses.JSONResponse({'ok': True})

    async def api_stop(request: requests.Request) -> responses.Response:
        del request
        ok, err = state.enqueue_stop()
        if not ok:
            return responses.JSONResponse({'error': err}, status_code=409)
        return responses.JSONResponse({'ok': True})

    async def api_reset(request: requests.Request) -> responses.Response:
        del request
        ok, err = state.enqueue_reset()
        if not ok:
            return responses.JSONResponse({'error': err}, status_code=409)
        return responses.JSONResponse({'ok': True})

    return applications.Starlette(routes=[
        routing.Route('/', index),
        routing.Route('/api/state', api_state),
        routing.Route('/api/run', api_run, methods=['POST']),
        routing.Route('/api/feedback', api_feedback, methods=['POST']),
        routing.Route('/api/mode', api_mode, methods=['POST']),
        routing.Route('/api/pause', api_pause, methods=['POST']),
        routing.Route('/api/resume', api_resume, methods=['POST']),
        routing.Route('/api/stop', api_stop, methods=['POST']),
        routing.Route('/api/reset', api_reset, methods=['POST']),
    ])


def build_default_runtime(
    *,
    base_checkpoint: pathlib.Path = DEFAULT_BASE_CKPT,
    live_dir: pathlib.Path = DEFAULT_LIVE_DIR,
    experience_dir: pathlib.Path = DEFAULT_EXPERIENCE_DIR,
    resume_live: bool = False,
) -> runtime.WamRuntime:
    """Initializes WamRuntime from the verified dynamics checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = semantic.SemanticEncoder(device)
    base_sha = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()

    live_ckpt_path = live_dir / 'latest.pt'
    if resume_live and live_ckpt_path.exists():
        world, ckpt = live_learning.load_from_live_checkpoint(
            live_ckpt_path, device
        )
        model_revision = int(ckpt.get('model_revision', 0))
        sup_counts = dict(ckpt.get('supervision_counts', {}))
    else:
        world, ckpt = content.load_from_dynamics_checkpoint(
            base_checkpoint, device
        )
        model_revision = 0
        sup_counts = None

    role_vocab = tuple(ckpt['role_vocabulary'])
    role_to_index = {role: idx for idx, role in enumerate(role_vocab)}

    loaded_train = training_data.load_experience_windows(
        split='train', max_horizon=1
    )
    all_wins = loaded_train.windows[1]
    ref_wins = all_wins[:4]
    replay_wins = all_wins[4:]

    learner = live_learning.LiveLearner(
        world,
        encoder,
        role_to_index,
        live_learning.LearningConfig(),
        base_checkpoint_sha256=base_sha,
        replay_windows=replay_wins,
        reference_windows=ref_wins,
        model_revision=model_revision,
        supervision_counts=sup_counts,
    )
    if (
        resume_live
        and live_ckpt_path.exists()
        and 'optimizer_state_dict' in ckpt
    ):
        learner.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    return runtime.WamRuntime(
        world=world,
        encoder=encoder,
        role_to_index=role_to_index,
        config=runtime.RuntimeConfig(),
        checkpoint_metadata={
            'base_checkpoint': str(base_checkpoint),
            'base_checkpoint_sha256': base_sha,
            'resumed_live': bool(resume_live and live_ckpt_path.exists()),
        },
        learner=learner,
        experience_dir=experience_dir,
        checkpoint_dir=live_dir,
    )


def _load_fresh_env(
    page: sync_api.Page,
    watcher: observer.Observer,
    assignments: dict[str, booking.Scenario],
    site_port: int,
    family: str,
    seed: int,
) -> str:
    scenario = scenarios.generate(family, seed)
    sid = f'wam-{uuid.uuid4().hex[:8]}'
    assignments[sid] = scenario
    url = f'http://127.0.0.1:{site_port}/?session={sid}'
    page.goto(url)
    page.wait_for_selector('body[data-ready=true]')
    watcher.wait_until_settled()
    return url


def run_workspace(
    family: str = 'riverside-wizard',
    seed: int = 90000,
    site_port: int = 7800,
    workspace_port: int = 7801,
    headless: bool = False,
    task_runner: Callable[..., Any] | None = None,
    wam_runtime: runtime.WamRuntime | None = None,
    verify_windows_only: bool = False,
) -> dict[str, Any]:
    """Starts servers and launches two independent Chromium browsers."""
    assignments: dict[str, booking.Scenario] = {}
    site_app = server.create_app(
        lambda sid: assignments.get(sid, booking.default_scenario())
    )
    site_srv = live_demo.start_server(site_app, site_port)

    if task_runner is None and wam_runtime is not None:
        def _run_with_wam(
            ws_state_arg: WorkspaceState,
            page: sync_api.Page,
            watcher: observer.Observer,
            hand: executor.Executor,
            overlay: watch_agent.Overlay,
            instruction: str,
        ) -> str:
            mode = runtime.RuntimeMode(ws_state_arg.get_mode())
            return wam_runtime.run_task(
                instruction,
                mode,
                page=page,
                watcher=watcher,
                hand=hand,
                overlay=overlay,
                host=ws_state_arg,
            )

        task_runner = _run_with_wam

    workspace_url = f'http://127.0.0.1:{workspace_port}/'
    readiness = check_model_readiness(
        task_runner=task_runner, wam_runtime=wam_runtime
    )
    ws_state = WorkspaceState(
        readiness=readiness,
        workspace_url=workspace_url,
        environment_url=f'http://127.0.0.1:{site_port}/',
        autonomous_check=(
            wam_runtime.learner.autonomous_block_reason
            if wam_runtime is not None
            else None
        ),
    )
    ws_app = create_workspace_app(ws_state)
    ws_srv = live_demo.start_server(ws_app, workspace_port)
    info: dict[str, Any] = {
        'workspace_url': workspace_url,
        'environment_url': ws_state.environment_url,
    }

    try:
        with sync_api.sync_playwright() as pw:
            launch_args = ['--force-device-scale-factor=1']
            left_browser = pw.chromium.launch(
                headless=headless, args=launch_args
            )
            right_browser = pw.chromium.launch(
                headless=headless, args=launch_args
            )
            ctx_kwargs = {'no_viewport': True} if not headless else {
                'viewport': {'width': 960, 'height': 900}
            }
            left_ctx = left_browser.new_context(**ctx_kwargs)
            right_ctx = right_browser.new_context(**ctx_kwargs)

            left_page = left_ctx.new_page()
            right_page = right_ctx.new_page()
            watcher = observer.attach(right_page)

            left_page.goto(workspace_url, wait_until='domcontentloaded')
            env_url = _load_fresh_env(
                right_page, watcher, assignments, site_port, family, seed
            )
            with ws_state._lock:
                ws_state.environment_url = env_url
            ws_state.set_observation_summary(watcher.observe())
            info['environment_url'] = env_url

            if not headless:
                dx, dy, dw, dh = _desktop_work_area()
                half_w = dw // 2
                left_meta = place_window_cdp(left_page, dx, dy, half_w, dh)
                right_meta = place_window_cdp(
                    right_page, dx + half_w, dy, dw - half_w, dh
                )
                info['left_window'] = left_meta
                info['right_window'] = right_meta
                print(f'Workspace URL : {workspace_url}')
                print(f'Env URL       : {env_url}')

            if verify_windows_only:
                left_page.wait_for_selector('#instruction')
                info['textarea_visible'] = left_page.is_visible('#instruction')
                info['send_visible'] = left_page.is_visible('#send')
                left_browser.close()
                right_browser.close()
                return info

            overlay = watch_agent.Overlay(right_page)
            hand = executor.Executor(
                right_page,
                watcher,
                executor.FAST_SPEED if headless else executor.NORMAL_SPEED,
                on_move=overlay.pointer,
            )

            while not left_page.is_closed() and not right_page.is_closed():
                with ws_state._lock:
                    should_reset = ws_state.reset_requested
                    if should_reset:
                        ws_state.reset_requested = False
                if should_reset:
                    if wam_runtime is not None:
                        wam_runtime.reset_session()
                    env_url = _load_fresh_env(
                        right_page, watcher, assignments, site_port,
                        family, seed,
                    )
                    with ws_state._lock:
                        ws_state.environment_url = env_url
                    ws_state.set_observation_summary(watcher.observe())
                    continue

                with ws_state._lock:
                    should_run = ws_state.run_requested
                    if should_run:
                        ws_state.run_requested = False
                        instruction = ws_state.current_instruction or ''
                        nl_ready = bool(ws_state.readiness.get('nl_ready'))
                        reason = ws_state.readiness.get(
                            'reason', UNINITIALIZED_REASON
                        )
                    else:
                        instruction = ''
                        nl_ready = False
                        reason = ''

                if not should_run:
                    time.sleep(0.1)
                    continue

                if not nl_ready or not callable(task_runner):
                    with ws_state._lock:
                        ws_state.execution_status = 'unsupported_instruction'
                        ws_state.error = reason
                    ws_state.add_message(
                        'wam', 'error', f'Cannot run: {reason}'
                    )
                    continue

                with ws_state._lock:
                    ws_state.execution_status = 'running'
                try:
                    outcome = task_runner(
                        ws_state, right_page, watcher, hand, overlay,
                        instruction,
                    )
                    if wam_runtime is not None:
                        ws_state.set_readiness(
                            check_model_readiness(
                                task_runner=task_runner,
                                wam_runtime=wam_runtime,
                            )
                        )
                    with ws_state._lock:
                        if ws_state.execution_status == 'running':
                            ws_state.execution_status = (
                                outcome
                                if isinstance(outcome, str) and outcome
                                else 'completed'
                            )
                except Exception as exc:  # Isolate runner failure from UI loop.
                    with ws_state._lock:
                        ws_state.execution_status = 'failed'
                        ws_state.error = str(exc)
                    ws_state.add_message(
                        'wam', 'error', f'Execution failed: {exc}'
                    )

            left_browser.close()
            right_browser.close()
    except KeyboardInterrupt:
        pass
    finally:
        site_srv.should_exit = True
        ws_srv.should_exit = True

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--family', default='riverside-wizard')
    parser.add_argument('--seed', type=int, default=90000)
    parser.add_argument('--site-port', type=int, default=7800)
    parser.add_argument('--workspace-port', type=int, default=7801)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--resume-live', action='store_true')
    args = parser.parse_args()
    wam_rt = build_default_runtime(resume_live=args.resume_live)
    run_workspace(
        family=args.family,
        seed=args.seed,
        site_port=args.site_port,
        workspace_port=args.workspace_port,
        headless=args.headless,
        wam_runtime=wam_rt,
    )


if __name__ == '__main__':
    main()
