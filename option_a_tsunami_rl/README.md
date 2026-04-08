# Option A: Tsunami Warning RL

This folder contains a self-contained implementation of Option A: reinforcement learning for tsunami early warning decision support.

## What is here

- `src/data_pipeline.py`: scrapes BMKG InaTSP public bulletin pages and linked detail pages, saves raw HTML locally, and builds the base event-summary dataset.
- `src/enrichment.py`: pulls additional official tsunami and earthquake data from NOAA NCEI and USGS, matches them onto BMKG events, and generates train-only synthetic augmentation.
- `src/environment.py`: defines the time-stepped data-informed warning environment with explicit action semantics and persistent sensor evidence.
- `src/dp_baseline.py`: small fully observed oracle MDP used for a value-iteration reference policy.
- `src/agents.py`: SARSA and Q-learning trainers, a rule-based policy, and both pure-RL and rule-backed safe evaluation helpers.
- `src/experiment.py`: runs class-aware chronological splits, repeated-seed held-out evaluation, and output generation.
- `src/deep_agents.py`: masked recurrent PPO-LSTM training, evaluation, checkpointing, and TensorBoard logging.
- `src/metrics.py`: shared operational scoring used for safety-aware model selection.
- `run_all.py`: orchestrates the full local pipeline.
- `training/train_ppo_lstm.py`: local PPO-LSTM training entrypoint for reproducibility runs.
- `training/build_repro_config.py`: writes the local reproducibility configuration for the strongest bundled PPO run.
- `training/launch_local_run.py`: turns the reproducibility config into a local training manifest and optional executable command.
- `training/local_repro_config.json`: selected PPO hyperparameters and warm-start path used for the bundled reproducibility setup.
- `notebooks/option_a_tsunami_rl_analysis.ipynb`: executed notebook with local results.

## Local outputs

Running `python run_all.py` creates these folders and artifacts:

- `data/raw/`: scraped BMKG bulletin pages and detail pages.
- `data/processed/`: parsed bulletin rows, parsed detail pages, and the event-summary dataset.
- `logs/`: `search_log.csv` and `pipeline.log`.
- `outputs/figures/`: summary plots and training curves.
- `outputs/tables/`: training history, evaluation summaries, and DP policy tables.
- `notebooks/`: executed notebook.
- `baseline_results/`: frozen pre-upgrade baseline config, split CSVs, metrics snapshot, and seed list.

For a direct summary of the saved logs, checkpoints, TensorBoard files, and evaluation tables that show training artifacts are present in the repo, see:

- `TRAINING_EVIDENCE.md`

## Data enrichment note

The pipeline now enriches BMKG with:

- NOAA NCEI tsunami events for external runup and tsunami-impact proxies
- USGS earthquake catalog events for external earthquake metadata and tsunami flags
- synthetic train-only bootstrap scenarios generated from the real training split

The enrichment summary is saved in `data/processed/external_enrichment_summary.csv`, and the synthetic train catalog is saved in `data/processed/synthetic_train_events.csv`.

## Scope note

This is a data-informed warning-policy simulator, not a hydrodynamic tsunami physics model. Real BMKG archive features are used to seed scenario type, escalation timing, target alert tiers, and some evidence cues, but parts of the sequential evidence process are still modeled heuristically. The BMKG archive is especially sparse on observed wave amplitudes, so many wave trajectories remain imputed rather than directly observed.

## Policy note

The outputs now separate:

- pure tabular deployment policies: `q_learning_pure` and `sarsa_pure`
- rule-backed safe deployment policies: `q_learning_safe` and `sarsa_safe`
- masked recurrent PPO deployment policy: `ppo_lstm_pure`

The safe variants only override the rule baseline when the learned Q-table shows a clear value advantage. The warm start is still rule-seeded, but it is smaller than the main RL training budget and is now logged in `outputs/tables/experiment_configuration.csv`.

## Evaluation note

The archive is strongly time-imbalanced by class, so the project now uses class-aware chronological splits rather than one global date cutoff. Within each danger class, older events go to train, later events go to validation and test, and held-out evaluation runs each event exactly once per seed.

## PPO selection note

PPO checkpoints are now selected with a safety-aware validation objective instead of raw return alone. The default score is:

- `avg_return - 300 * severe_miss_rate - 50 * false_warning_rate`

That same metric is used to choose the bundled local reproducibility checkpoint and remains exposed throughout the local training tooling as `validation_safety_score`.

## Latest PPO result

The strongest validated PPO result currently comes from the bundled local reproducibility run summarized in `outputs/tables/evaluation_summary_best_available.csv`.

- `ppo_lstm_pure` validation avg return: `25.52`
- `ppo_lstm_pure` test avg return: `25.34`
- `ppo_lstm_pure` test severe miss rate: `0.00`
- `ppo_lstm_pure` test false warning rate: `0.00`

On the current held-out test split, that run is stronger than the rule baseline at `20.40` average return while preserving the same zero severe-miss and zero false-warning rates.

## React dashboard

The repo also includes a richer React/Next.js operations console in `console/` with a FastAPI backend in `api/`. It adds:

- interactive map playback
- timeline replay controls
- telemetry panels for observations, valid actions, rewards, and rule/PPO diagnostics
- a small curated PPO checkpoint selector for meaningful model comparison

Start the application with Docker:

- `cd option_a_tsunami_rl`
- `docker compose up --build`

Open:

- `http://localhost:3000`

Run it locally without Docker:

- `python -m uvicorn option_a_tsunami_rl.api.server:app --host 127.0.0.1 --port 8000`
- `cd option_a_tsunami_rl/console && npm run dev`

Open:

- `http://localhost:3000`

The dashboard intentionally exposes only a few user-facing PPO choices instead of every training artifact:

- `Recommended PPO Policy`
- `Stable PPO Policy`
- `Baseline PPO Policy`

See `DASHBOARD_README.md` and `TESTING_GUIDE.md` for dashboard-specific details.

## Local reproducibility note

If you want to reproduce the bundled PPO-LSTM training setup locally, the repo includes:

- `python -m option_a_tsunami_rl.training.build_repro_config --output training/local_repro_config.json`
- `python -m option_a_tsunami_rl.training.launch_local_run --output-manifest outputs/tables/local_repro_manifest.json`
- `python -m option_a_tsunami_rl.training.launch_local_run --execute`

The bundled local reproducibility configuration keeps the selected hyperparameters, seeds, curriculum fractions, and warm-start checkpoint path together in one place:

- `training/local_repro_config.json`

## Reward note

The current environment reward is now mostly outcome- and evidence-driven. It does not use `recommended_alert_schedule`, `expected_alert`, or distance-to-schedule shaping. Dense step penalties are limited to time cost, alert changes, ignored strong evidence, and obvious over-warning; the dominant reward comes from terminal warning outcomes.

## Baseline freeze note

Before the longer-horizon POMDP refactor, the previous split, metrics, and config were frozen in `baseline_results/`. That folder is intentionally not overwritten so later changes can be compared against a fixed reference.
