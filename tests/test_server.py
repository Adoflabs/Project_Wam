# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import dataclasses

import pytest
from starlette import testclient

from browser_world.environment import booking
from browser_world.environment import server

_REQUEST = {
    'service': 'checkup',
    'doctor': 'dr-patel',
    'date': '2026-11-24',
    'slot': '09:00',
    'name': 'Sam Rivera',
    'email': 'sam@example.com',
    'phone': '5550102030',
    'consent': 'true',
}


@pytest.fixture
def client() -> testclient.TestClient:
    scenario = dataclasses.replace(booking.default_scenario(),
                                   slot_latency_ms=0)
    return testclient.TestClient(server.create_app(scenario))


def _headers(session_id: str = 's1') -> dict[str, str]:
    return {server.SESSION_HEADER: session_id}


def test_page_is_served_without_session(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'Book an appointment' in response.text


def test_api_requires_session(client):
    response = client.get('/api/services')
    assert response.status_code == 400


def test_catalog_endpoints(client):
    services = client.get('/api/services', headers=_headers()).json()
    assert [s['key'] for s in services] == ['checkup', 'dental', 'eye']
    doctors = client.get('/api/doctors?service=dental',
                         headers=_headers()).json()
    assert doctors == [{'key': 'dr-okafor', 'name': 'Dr. Ngozi Okafor'}]
    dates = client.get('/api/dates?doctor=dr-patel',
                       headers=_headers()).json()
    assert dates[0] == {'iso': '2026-11-24', 'label': 'Tue 24 Nov'}


def test_validate_unknown_step_is_404(client):
    response = client.post('/api/validate/payment', json={},
                           headers=_headers())
    assert response.status_code == 404


def test_booking_lifecycle(client):
    created = client.post('/api/bookings', json=_REQUEST, headers=_headers())
    assert created.status_code == 201
    assert created.json() == {'reference': 'BK-0001'}

    again = client.post('/api/bookings', json=_REQUEST, headers=_headers())
    assert again.status_code == 422
    assert again.json()['errors'] == {'slot': booking.MSG_SLOT_UNAVAILABLE}

    contested = client.post('/api/bookings',
                            json=dict(_REQUEST, slot='11:00'),
                            headers=_headers())
    assert contested.status_code == 409

    truth = client.get('/api/truth/bookings', headers=_headers()).json()
    assert [b['reference'] for b in truth] == ['BK-0001']
    other = client.get('/api/truth/bookings', headers=_headers('s2')).json()
    assert not other

    client.post('/api/reset', headers=_headers())
    assert not client.get('/api/truth/bookings', headers=_headers()).json()
