# Tsunami Warning RL Console Testing Guide

This guide covers the current React/Next.js dashboard in `console/`, the FastAPI backend in `api/`, and the existing automated test paths.

## Local development

From the repository root:

```bash
python -m uvicorn option_a_tsunami_rl.api.server:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd option_a_tsunami_rl/console
npm run dev
```

The console will be available at `http://localhost:3000`, and the backend will be available at `http://localhost:8000`.

## Docker quick start

From `option_a_tsunami_rl/`:

```bash
docker build -t tsunami-rl-console .
docker run -d --name tsunami-console -p 4000:3000 -p 4001:8000 tsunami-rl-console
```

Expected ports:

| Port | Service |
| ---- | ------- |
| `4000` | Next.js dashboard |
| `4001` | FastAPI backend |

## API smoke checks

Health:

```bash
curl http://localhost:8000/api/health
```

Expected:

```json
{"status":"ok"}
```

Catalog:

```bash
curl http://localhost:8000/api/catalog | python -m json.tool | head -20
```

Expected:
- JSON array
- each event has `event_group_id`, `danger_tier`, `danger_label`

Checkpoints:

```bash
curl http://localhost:8000/api/checkpoints | python -m json.tool
```

Expected:
- non-empty JSON array
- each checkpoint has `name` and `path`
- the dashboard-facing list is intentionally curated to a small set of user-friendly labels

Rule-based simulation:

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"agent_type":"rule","seed":42,"danger_filter":"All"}'
```

PPO simulation:

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"agent_type":"ppo","seed":42,"danger_filter":"All"}'
```

Expected for both:
- `event_metadata`
- `frames` with exactly `12` entries
- `total_return`
- `outcome_summary`

Additional PPO fields:
- `agent_probabilities`
- `value_estimate`
- `hidden_trace`

`hidden_trace` is the simulator's hidden evidence trace for the episode, not the model's internal LSTM state.

Invalid agent type:

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"agent_type":"invalid_agent","seed":42}'
```

Expected:
- HTTP `422`

## UI expectations

Initial load:
- left control panel is visible
- idle overlay is visible
- threat filter defaults to `All`
- scenario defaults to `Random from catalog`
- agent type defaults to `PPO-LSTM (Deep RL)`
- checkpoint labels use user-facing names such as `Recommended PPO Policy`
- play, pause, step, and reset are disabled

After starting a PPO episode:
- map alert banner is visible
- event info overlay is visible
- timeline shows `1 / 12 steps`
- telemetry shows metrics, observations, valid actions, state summary
- policy probabilities and `VALUE EST` are visible

After starting a rule-based episode:
- checkpoint selector is hidden
- telemetry still shows observations, valid actions, and rule recommendation
- policy probabilities are hidden
- `VALUE EST` is hidden

Startup failure behavior:
- if `/api/catalog` or `/api/checkpoints` fails during page load, a visible error banner appears

## Automated tests

Python tests:

```bash
python -m pytest -q
```

Console build:

```bash
cd option_a_tsunami_rl/console
npm run build
```

Playwright against local dev servers:

```bash
cd option_a_tsunami_rl/console
npx playwright test --reporter=list
```

Playwright against a running Docker deployment:

```bash
cd option_a_tsunami_rl/console
BASE_URL=http://localhost:4000 npx playwright test --reporter=list
```

When `BASE_URL` is provided, Playwright targets that deployment instead of starting local web servers.

Current spec coverage:

| Spec file | Coverage |
| --------- | -------- |
| `01-layout-and-controls.spec.ts` | layout, default controls, dropdown behavior |
| `02-simulation-flow.spec.ts` | start, play, pause, step, reset |
| `03-map-telemetry-timeline.spec.ts` | map overlays, telemetry, episode timeline |
| `04-agent-types-and-scenarios.spec.ts` | PPO vs rule mode, filters, seeds, speed controls |
| `05-api-errors-accessibility.spec.ts` | API integration, startup/simulation errors, accessibility basics |
| `06-end-to-end-workflows.spec.ts` | longer dashboard workflows |
| `07-api-edge-cases.spec.ts` | event lookup, filter interactions, response integrity |

Expected automated result:
- `pytest` passes
- `npm run build` passes
- all Playwright specs pass

## Quick validation checklist

| Check | Pass criteria |
| ----- | ------------- |
| API health | `/api/health` returns `{"status":"ok"}` |
| Catalog | `/api/catalog` returns a JSON array |
| Checkpoints | `/api/checkpoints` returns a non-empty JSON array |
| Rule simulation | `/api/simulate` with `agent_type=rule` returns 12 frames |
| PPO simulation | `/api/simulate` with `agent_type=ppo` returns 12 frames plus probabilities/value estimate |
| Invalid agent rejection | `/api/simulate` with invalid `agent_type` returns `422` |
| Dashboard load | idle overlay and control panel render |
| Episode replay | start + step reaches `12 / 12` with outcome banner |
