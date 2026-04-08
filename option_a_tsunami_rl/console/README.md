# Tsunami RL Operations Console

A single-page operations console for the tsunami early warning RL decision support system.

## Architecture

- **FastAPI backend** (`api/server.py`): Wraps the Python simulator, generates complete episode frame sequences as JSON
- **Next.js frontend** (`console/`): Single-page console with MapLibre map, ECharts telemetry, and frame replay

## Running

### 1. Start the API server

From the **workspace root** (`cscn8040/`):

```bash
python -m uvicorn option_a_tsunami_rl.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the frontend

```bash
cd option_a_tsunami_rl/console
npm run dev
```

Then open **http://localhost:3000**

## Layout

```
┌──────────┬──────────────────────────────┐
│  Control │       Simulation Map         │
│  Panel   │   (MapLibre GL + overlays)   │
│          │                              │
│ - Filter ├───────────────────┬──────────┤
│ - Event  │  Episode Timeline │          │
│ - Agent  ├───────────────────┤ Telemetry│
│ - Speed  │                   │ & RL     │
│ - Start  │  Observations     │ Diagnost │
│ - Pause  │  Agent Probs      │ Charts   │
│ - Step   │  State Summary    │          │
│ - Reset  │                   │          │
└──────────┴───────────────────┴──────────┘
```

## How it works

1. The left panel collects episode parameters (event, agent type, seed, speed)
2. Pressing **Start Episode** sends a POST to `/api/simulate`
3. The backend runs the full episode through the RL environment and returns all 12 frames
4. The frontend replays those frames with:
   - Animated map showing epicenter, wave propagation, and sensor status
   - Telemetry panel with observations, agent probabilities, and reward charts
   - Timeline cards for each step showing action, alert, and reward

## Technology Stack

- **Next.js + React** for the app shell
- **MapLibre GL JS** for the base map with dark styling
- **Tailwind CSS** for utility styling
- **Custom SVG charts** for reward/uncertainty/wave sparklines
- **FastAPI** backend wrapping the existing Python simulator
