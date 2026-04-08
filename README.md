# CSCN8040 Tsunami RL

Dedicated repository for the reinforcement learning tsunami early-warning project.

## Repository layout

- `option_a_tsunami_rl/`: standalone RL application, data pipeline, trained outputs, dashboard, API, and tests

## Start the app with Docker

```bash
cd option_a_tsunami_rl
docker compose up --build
```

Then open `http://localhost:3000`.

## Quick start

```bash
cd option_a_tsunami_rl
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

For the operations console, Docker workflow, and testing details, see:

- `option_a_tsunami_rl/README.md`
- `option_a_tsunami_rl/DASHBOARD_README.md`
- `option_a_tsunami_rl/TESTING_GUIDE.md`
