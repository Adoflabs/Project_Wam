"""Tests for the clean WAM research workspace runtime and UI."""

from __future__ import annotations

import threading

from playwright import sync_api
from starlette import testclient

from browser_world.diagnostics import live_demo
from browser_world.diagnostics import workspace
from tests import conftest


def _make_state(*, nl_ready: bool = False) -> workspace.WorkspaceState:
    readiness = {
        'ready': nl_ready,
        'nl_ready': nl_ready,
        'ui_available': True,
        'reason': (
            'All runtimes ready'
            if nl_ready
            else workspace.UNINITIALIZED_REASON
        ),
        'dynamics_loaded': nl_ready,
        'teaching_ready': nl_ready,
        'decision_supervision_available': False,
        'experimental_autonomous_enabled': False,
    }
    return workspace.WorkspaceState(
        readiness=readiness,
        workspace_url='http://127.0.0.1/',
        environment_url='http://127.0.0.1/',
        autonomous_check=(
            lambda: 'Autonomous mode disabled: no supervised proposals.'
        ),
    )


def test_readiness_reporting() -> None:
    uninit = workspace.check_model_readiness(task_runner=None)
    assert uninit['ready'] is False
    assert uninit['nl_ready'] is False
    assert uninit['ui_available'] is True
    assert uninit['reason'] == 'New WAM runtime not initialized'

    ready = workspace.check_model_readiness(task_runner=lambda *a: None)
    assert ready['ready'] is True
    assert ready['nl_ready'] is True
    assert ready['ui_available'] is True


def test_http_contract_and_controls() -> None:
    uninit_state = _make_state(nl_ready=False)
    uninit_client = testclient.TestClient(
        workspace.create_workspace_app(uninit_state)
    )
    res_503 = uninit_client.post('/api/run', json={'instruction': 'Book skin'})
    assert res_503.status_code == 503
    assert res_503.json()['error'] == 'New WAM runtime not initialized'

    state = _make_state(nl_ready=True)
    client = testclient.TestClient(workspace.create_workspace_app(state))
    assert client.post('/api/run', json={}).status_code == 400
    assert (
        client.post('/api/run', json={'instruction': '   '}).status_code == 400
    )
    assert (
        client.post('/api/run', json={'instruction': 'Book'}).status_code == 200
    )
    assert (
        client.post('/api/run', json={'instruction': 'Again'}).status_code
        == 409
    )
    assert client.post('/api/stop', json={}).status_code == 200
    assert client.post('/api/reset', json={}).status_code == 200
    snap = client.get('/api/state').json()
    assert snap['execution_status'] == 'idle'
    assert snap['runtime_messages'] == []


def test_feedback_and_mode_endpoints() -> None:
    state = _make_state(nl_ready=True)
    client = testclient.TestClient(workspace.create_workspace_app(state))

    # Autonomous blocked without supervision.
    mode_res = client.post('/api/mode', json={'mode': 'autonomous'})
    assert mode_res.status_code == 409

    # Feedback rejected when no decision is pending.
    fb_empty = client.post('/api/feedback', json={'decision_id': 'dec-1'})
    assert fb_empty.status_code == 409

    received: dict[str, object] = {}

    def _wait_fb() -> None:
        res = state.request_feedback({
            'decision_id': 'dec-42',
            'instruction': 'Enter Rahul',
            'candidates': [{'candidate_id': 0}, {'candidate_id': 1}],
        })
        if res is not None:
            received.update(res)

    th = threading.Thread(target=_wait_fb)
    th.start()
    while state.snapshot()['pending_decision'] is None:
        pass

    stale = client.post('/api/feedback', json={'decision_id': 'wrong-id'})
    assert stale.status_code == 409

    ok = client.post(
        '/api/feedback',
        json={
            'decision_id': 'dec-42',
            'approved_candidate_id': 0,
            'rejected_candidate_ids': [1],
            'literal_span': [6, 11],
            'goal_satisfied': False,
            'constraint_violated': None,
            'terminal_outcome': None,
        },
    )
    assert ok.status_code == 200
    th.join(timeout=2.0)
    assert received['approved_candidate_id'] == 0
    assert received['literal_span'] == [6, 11]


def test_browser_keyboard_and_uninitialized_banner(
    browser: sync_api.Browser,
) -> None:
    port = conftest._free_port()
    state = _make_state(nl_ready=False)
    app = workspace.create_workspace_app(state)
    ws_srv = live_demo.start_server(app, port)
    context = browser.new_context(viewport={'width': 768, 'height': 864})
    try:
        ws_page = context.new_page()
        run_calls = 0

        def _on_req(req):
            nonlocal run_calls
            if req.url.endswith('/api/run') and req.method == 'POST':
                run_calls += 1

        ws_page.on('request', _on_req)
        ws_page.goto(
            f'http://127.0.0.1:{port}/', wait_until='domcontentloaded'
        )

        assert ws_page.locator('#transcript').is_visible()
        assert 'New WAM runtime not initialized' in (
            ws_page.locator('#transcript').inner_text()
        )
        assert ws_page.locator('#exp-entity').count() == 0
        assert ws_page.locator('#dev-diagnostics').count() == 0

        ta = ws_page.locator('#instruction')
        assert ta.is_visible()
        assert ta.is_editable()
        ta.click()
        typed = 'Book an appointment for Ada Example.'
        ta.press_sequentially(typed, delay=10)
        ws_page.wait_for_timeout(700)
        assert ta.input_value() == typed
        assert ws_page.evaluate(
            'document.activeElement && document.activeElement.id'
        ) == 'instruction'
        ta.press('Enter')
        ws_page.wait_for_timeout(300)
        assert run_calls == 1
        assert ta.input_value() == typed
        assert ws_page.locator('#error-banner').is_visible()
        assert (
            ws_page.locator('#status-pill').inner_text()
            == 'unsupported_instruction'
        )
        assert 'bad' in (
            ws_page.locator('#status-pill').get_attribute('class') or ''
        )
    finally:
        context.close()
        ws_srv.should_exit = True
