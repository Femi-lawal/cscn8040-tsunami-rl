# Training Evidence

This note summarizes the concrete files in the repository that show agent training and evaluation artifacts are present.

## Tabular agent training

The repo includes saved episode-level training traces for the tabular agents:

- `outputs/tables/training_history.csv`

That file contains:

- `24,000` rows for `q_learning`
- `24,000` rows for `sarsa`

Each row records per-episode values such as:

- `episode`
- `algorithm`
- `epsilon`
- `return`
- `danger_tier`
- `missed_severe`
- `false_warning`
- `invalid_action_count`
- `run_seed`

## PPO-LSTM training

The repo also includes saved PPO-LSTM training outputs:

- `outputs/tables/ppo_lstm_training_history.csv`
- `outputs/tables/ppo_lstm_seed77_training_history.csv`
- `outputs/tensorboard/`
- `outputs/models/`

The aggregated PPO training history contains `954` logged updates. The rows include intermediate optimization and safety metrics such as:

- `policy_loss`
- `value_loss`
- `entropy`
- `clip_fraction`
- `recent_avg_return`
- `recent_severe_miss_rate`
- `recent_false_warning_rate`
- `validation_safety_score`
- `run_seed`

The saved TensorBoard event files in `outputs/tensorboard/` are additional training-run evidence.

## Local reproducibility bundle

The strongest bundled PPO run is preserved in:

- `outputs/local_repro/metrics.json`
- `outputs/local_repro/metric_lines.txt`
- `outputs/local_repro/tables/ppo_lstm_training_history.csv`
- `outputs/local_repro/tables/ppo_lstm_evaluation_by_seed.csv`
- `outputs/local_repro/tables/ppo_lstm_evaluation_summary.csv`
- `outputs/local_repro/models/`
- `outputs/local_repro/tensorboard/`

That bundle includes:

- validation and test metrics
- per-seed evaluation results for seeds `77`, `88`, and `99`
- multiple PPO checkpoints
- TensorBoard event files for the bundled run

## Evaluation artifacts

Saved held-out evaluation evidence is available in:

- `outputs/tables/agent_evaluation_summary.csv`
- `outputs/tables/agent_evaluation_by_seed.csv`
- `outputs/tables/evaluation_summary.csv`
- `outputs/tables/evaluation_summary_best_available.csv`
- `outputs/tables/ppo_lstm_evaluation_summary.csv`
- `outputs/tables/ppo_lstm_evaluation_by_seed.csv`
- `outputs/tables/ppo_lstm_evaluation_episode_metrics.csv`

These files show final validation and test performance for:

- rule-based policy
- Q-learning pure and safe variants
- SARSA pure and safe variants
- PPO-LSTM

## Saved model artifacts

The repo contains trained PPO checkpoints in:

- `outputs/models/ppo_lstm_best.pt`
- `outputs/models/ppo_lstm_last.pt`
- `outputs/models/ppo_lstm_recommended.pt`
- `outputs/models/ppo_lstm_stable.pt`
- `outputs/models/ppo_lstm_baseline.pt`
- `outputs/models/ppo_lstm_warmstart_reference.pt`

These are concrete serialized model artifacts, not placeholder filenames.

## Important caveat

The repository clearly contains saved training logs, evaluation tables, checkpoints, and TensorBoard artifacts.

What this proves:

- training artifacts were generated and committed
- evaluation was run and saved
- model selection artifacts are present

What this does not prove by itself:

- that every artifact was freshly regenerated after the latest repository cleanup
- that a full clean end-to-end retraining was rerun on this exact machine

For local reruns, see:

- `training/local_repro_config.json`
- `outputs/tables/local_repro_manifest.json`
- `README.md`
