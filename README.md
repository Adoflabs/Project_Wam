# Project WAM — World Action Model

Browser world model v0 (BW0). The locked plan is in `docs/BW0-spec.md`.

## Browser world (`browser_world/`)

| Module | Role |
|---|---|
| `booking.py` | Ground-truth booking rules and isolated per-session state (no HTTP, no browser) |
| `server.py` | Starlette server: the booking page plus a JSON API. `/api/truth/*` is backend ground truth that the page never calls |
| `static/app.html`, `static/app.js` | Three-step booking wizard (appointment → details → review → confirmation) |
| `observer.py` | Four-plane Chromium observer (semantic, geometry, motion, temporal) over CDP. No screenshots, no HTML |
| `delta.py` | What changed between two observations (the world model's prediction target) |
| `inspect_page.py` | Prints what the observer sees |
| `live_demo.py`, `static/viewer.html` | Live viewer: use the site in a real Chromium window and watch the observation update |

```bash
pip install -r requirements.txt
python -m browser_world.server --port 7800   # open http://127.0.0.1:7800/?session=demo
python -m browser_world.live_demo            # Chromium window + live viewer at http://127.0.0.1:7801
python -m browser_world.inspect_page         # observe the booking page once
python -m pytest                             # 52 tests: rules, API, browser E0.1 acceptance, observer, delta
pylint browser_world tests                   # Google-style lint config in .pylintrc
```

If a different Playwright version is installed, launch Chromium with
`executable_path='/opt/pw-browsers/chromium'` (cloud container) or run
`playwright install chromium`.

## Kaggle tooling

- `tools/kgl.py` — runs the Kaggle CLI with the API token supplied by the cloud environment's proxy:
  `python tools/kgl.py <kaggle args>`
- `kaggle/gpu-smoke/` — a Kaggle GPU smoke test (confirms a T4 is attached and usable).
