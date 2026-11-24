"""HTTP server for the appointment-booking environment.

Serves the booking web page and a small JSON API over `booking.py`. Every
API request names its isolated session in the `X-WAM-Session` header; the
page reads the session id from its own URL (`/?session=<id>`).

Endpoints under `/api/truth/` expose backend ground truth for data
collection and evaluation. The page never calls them, so a model that only
sees the page cannot see them either.

Run directly:

  python -m browser_world.server --port 7800
  # then open http://127.0.0.1:7800/?session=demo
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
import dataclasses
import pathlib

from starlette import applications
from starlette import requests
from starlette import responses
from starlette import routing
import uvicorn

from browser_world import booking

SESSION_HEADER = 'X-WAM-Session'
_STATIC_DIR = pathlib.Path(__file__).parent / 'static'


class _MissingSessionError(Exception):
    """The request did not name a session."""


def _session(request: requests.Request) -> booking.BookingSession:
    session_id = request.headers.get(SESSION_HEADER, '')
    if not session_id:
        raise _MissingSessionError()
    registry: booking.SessionRegistry = request.app.state.registry
    return registry.get(session_id)


async def _form(request: requests.Request) -> dict[str, str]:
    """Returns the JSON body as a flat str-to-str mapping."""
    body = await request.json()
    if not isinstance(body, dict):
        return {}
    return {str(k): str(v) for k, v in body.items()}


async def _page(request: requests.Request) -> responses.Response:
    del request  # Unused.
    return responses.FileResponse(_STATIC_DIR / 'app.html')


async def _app_js(request: requests.Request) -> responses.Response:
    del request  # Unused.
    return responses.FileResponse(_STATIC_DIR / 'app.js',
                                  media_type='text/javascript')


async def _presentation(request: requests.Request) -> responses.Response:
    presentation = _session(request).scenario.presentation
    return responses.JSONResponse({'layout': presentation.layout,
                                   'text': dict(presentation.text)})


async def _services(request: requests.Request) -> responses.Response:
    session = _session(request)
    return responses.JSONResponse(
        [{'key': s.key, 'name': s.name} for s in session.services()])


async def _doctors(request: requests.Request) -> responses.Response:
    session = _session(request)
    service = request.query_params.get('service', '')
    return responses.JSONResponse(
        [{'key': d.key, 'name': d.name} for d in session.doctors(service)])


async def _dates(request: requests.Request) -> responses.Response:
    session = _session(request)
    doctor = request.query_params.get('doctor', '')
    return responses.JSONResponse([{
        'iso': day.isoformat(),
        'label': day.strftime('%a %d %b'),
    } for day in session.dates(doctor)])


async def _slots(request: requests.Request) -> responses.Response:
    session = _session(request)
    latency_ms = session.scenario.slot_latency_ms
    if latency_ms:
        await asyncio.sleep(latency_ms / 1000)
    doctor = request.query_params.get('doctor', '')
    date_iso = request.query_params.get('date', '')
    return responses.JSONResponse([{
        'time': s.time,
        'available': s.available,
    } for s in session.slots(doctor, date_iso)])


async def _validate(request: requests.Request) -> responses.Response:
    session = _session(request)
    validators = {
        'appointment': session.validate_appointment,
        'details': session.validate_details,
    }
    validator = validators.get(request.path_params['step'])
    if validator is None:
        return responses.JSONResponse({'error': 'Unknown step.'},
                                      status_code=404)
    return responses.JSONResponse({'errors': validator(await _form(request))})


async def _book(request: requests.Request) -> responses.Response:
    session = _session(request)
    form = await _form(request)
    try:
        created = session.book(form)
    except booking.ValidationError as e:
        return responses.JSONResponse({'errors': e.errors}, status_code=422)
    except booking.SlotTakenError as e:
        return responses.JSONResponse({'error': str(e)}, status_code=409)
    return responses.JSONResponse(
        {'reference': created.reference}, status_code=201)


async def _reset(request: requests.Request) -> responses.Response:
    _session(request).reset()
    return responses.JSONResponse({'ok': True})


async def _truth_bookings(request: requests.Request) -> responses.Response:
    session = _session(request)
    return responses.JSONResponse(
        [dataclasses.asdict(b) for b in session.bookings])


async def _missing_session(request: requests.Request,
                           exc: Exception) -> responses.Response:
    del request, exc  # Unused.
    return responses.JSONResponse(
        {'error': f'Missing {SESSION_HEADER} header.'}, status_code=400)


def create_app(
        scenario: booking.Scenario | Callable[[str], booking.Scenario]
) -> applications.Starlette:
    """Builds the ASGI app.

    Args:
      scenario: The scenario for every session, or a function from a
        session id to its scenario. The page learns its scenario only
        through the API, never through its URL.

    Returns:
      A Starlette application.
    """
    app = applications.Starlette(
        routes=[
            routing.Route('/', _page),
            routing.Route('/static/app.js', _app_js),
            routing.Route('/api/presentation', _presentation),
            routing.Route('/api/services', _services),
            routing.Route('/api/doctors', _doctors),
            routing.Route('/api/dates', _dates),
            routing.Route('/api/slots', _slots),
            routing.Route('/api/validate/{step:str}', _validate,
                          methods=['POST']),
            routing.Route('/api/bookings', _book, methods=['POST']),
            routing.Route('/api/reset', _reset, methods=['POST']),
            routing.Route('/api/truth/bookings', _truth_bookings),
        ],
        exception_handlers={_MissingSessionError: _missing_session},
    )
    app.state.registry = booking.SessionRegistry(scenario)
    return app


def main() -> None:
    """Runs the server with the default scenario."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7800)
    args = parser.parse_args()
    uvicorn.run(create_app(booking.default_scenario()),
                host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
