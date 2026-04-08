# Frozen Baseline

This folder freezes the pre-refactor baseline before the outcome-driven reward and longer POMDP trace changes.

## Test Snapshot

| Policy | Avg Return | Severe Miss Rate | False Warning Rate |
|---|---:|---:|---:|
| q_learning_pure | -33.68 | 0.625 | 0.036 |
| sarsa_pure | -39.48 | 0.667 | 0.054 |
| q_learning_safe | 23.16 | 0.000 | 0.000 |
| sarsa_safe | 21.66 | 0.000 | 0.000 |
| rule_based | 21.76 | 0.000 | 0.000 |
