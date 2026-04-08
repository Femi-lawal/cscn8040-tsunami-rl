from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook(root: Path) -> Path:
    notebook_path = root / "notebooks" / "option_a_tsunami_rl_analysis.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        new_markdown_cell(
            "# Option A: Tsunami Warning Reinforcement Learning\n"
            "\n"
            "This notebook summarizes the locally scraped BMKG bulletin archive, the event-summary dataset used to seed a data-informed warning simulator, and the reinforcement-learning results for tsunami warning decision support."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "\n"
            "ROOT = Path.cwd().resolve()\n"
            "DATA = ROOT / 'data' / 'processed'\n"
            "TABLES = ROOT / 'outputs' / 'tables'\n"
            "FIGS = ROOT / 'outputs' / 'figures'\n"
            "event_summary = pd.read_csv(DATA / 'bmkg_event_summary_enriched.csv')\n"
            "event_splits = pd.read_csv(TABLES / 'event_summary_with_splits.csv')\n"
            "training_history = pd.read_csv(TABLES / 'training_history.csv')\n"
            "evaluation_summary = pd.read_csv(TABLES / 'agent_evaluation_summary.csv')\n"
            "evaluation_by_seed = pd.read_csv(TABLES / 'agent_evaluation_by_seed.csv')\n"
            "experiment_configuration = pd.read_csv(TABLES / 'experiment_configuration.csv')\n"
            "external_enrichment = pd.read_csv(DATA / 'external_enrichment_summary.csv')\n"
            "synthetic_counts = pd.read_csv(TABLES / 'synthetic_train_label_counts.csv')\n"
            "dp_policy = pd.read_csv(TABLES / 'dp_optimal_policy.csv')\n"
            "event_summary.head()"
        ),
        new_markdown_cell(
            "## Data Backbone\n"
            "\n"
            "The event catalog starts with BMKG InaTSP public bulletin pages and linked detail pages, then adds external enrichment from NOAA NCEI tsunami events and the USGS earthquake catalog. Training is further augmented with synthetic bootstrap scenarios generated only from the real training split. The simulator still does not try to reproduce full tsunami physics, but it now uses more than one official agency source to anchor earthquake characteristics and wave/runup proxies."
        ),
        new_code_cell(
            "event_splits[['time_split', 'danger_label', 'event_group_id']].groupby(['time_split', 'danger_label']).count().rename(columns={'event_group_id': 'count'})"
        ),
        new_code_cell("external_enrichment"),
        new_code_cell("synthetic_counts"),
        new_code_cell(
            "display(Image(filename=str(FIGS / 'danger_label_distribution.png')))\n"
            "display(Image(filename=str(FIGS / 'training_returns.png')))"
        ),
        new_markdown_cell(
            "## Evaluation Snapshot\n"
            "\n"
            "The main comparison here is between pure tabular SARSA/Q-learning policies, rule-backed safe SARSA/Q-learning policies, and a rule-based baseline on class-aware chronological validation and test splits. Each held-out event is evaluated once per seed. The dynamic-programming table is kept separately as a small fully observed oracle, not as a directly comparable baseline for the partially observed RL task."
        ),
        new_code_cell("evaluation_summary"),
        new_code_cell("experiment_configuration"),
        new_code_cell("evaluation_by_seed.head(10)"),
        new_code_cell("dp_policy.head(15)"),
        new_markdown_cell(
            "## Pitch Summary\n"
            "\n"
            "- **Environment:** a time-stepped tsunami warning simulator informed by BMKG bulletin history.\n"
            "- **External enrichment:** NOAA NCEI tsunami records, USGS earthquake catalog matches, and synthetic train-only bootstrap scenarios.\n"
            "- **Use case:** a decision-support agent for alerting, escalation, and cancellation.\n"
            "- **Target:** learn the warning policy that minimizes missed dangerous events, false alarms, and delay.\n"
            "- **Utility:** strong positive reward for correct early warnings, strong penalties for missed severe tsunamis and false warnings, plus smaller penalties for delay, invalid actions, and alert thrashing.\n"
            "- **Concepts implemented:** MDP framing, Bellman value updates, a value-iteration oracle on a tiny toy MDP, episodic rollout evaluation, temporal-difference learning, SARSA, Q-learning, exploration vs exploitation, comparison against a rule-based baseline, and an optional rule-backed safe fallback for deployment.\n"
            "- **Limits:** the simulator is data-informed but still heuristic, especially for wave evolution, and the RL baselines are tabular rather than deep."
        ),
    ]

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return notebook_path


def execute_notebook(notebook_path: Path) -> None:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    executor = ExecutePreprocessor(timeout=1200, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(notebook_path.parent.parent)}})

    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
