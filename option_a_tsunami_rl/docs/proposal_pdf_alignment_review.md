# Proposal PDF Alignment Review

This note compares the original PDF proposal with the current `option_a_tsunami_rl` implementation.

## Overall verdict

The PDF is **mostly aligned** with the project that was ultimately built. Its main framing is still accurate:

- tsunami early warning is treated as a sequential decision problem
- environment design and evaluation are central
- rule-based, Q-learning, and SARSA baselines are included
- a visual replay or dashboard layer is part of the deliverables
- the project is framed as decision support rather than full tsunami physics

## What should be updated

Some PDF statements are now outdated relative to the current repo:

1. The PDF says `CSCN8020`, but this project belongs to **CSCN8040**.
2. The PDF says **PPO-LSTM** is only an extension "if time permits." In the final project, masked recurrent PPO-LSTM is implemented and evaluated.
3. The final project is better described as **partially observable / POMDP-like** rather than a plain MDP.
4. The final repo includes **NOAA and USGS enrichment**, synthetic train-only augmentation, and a governed dashboard workflow, which go beyond the PDF's earlier scope wording.
5. The strongest PPO result highlighted in the repo comes from the bundled **local reproducibility run**, not from the default lighter local single-seed pipeline, so that distinction should be stated clearly.

## Safe presentation framing

The safest accurate high-level description is:

> This is a data-informed reinforcement learning project for tsunami warning decision support. It focuses on the warning-policy layer under uncertainty, compares rule-based and RL policies, and evaluates them on held-out events using operational safety metrics. It is not a hydrodynamic tsunami simulator or a claim that RL should replace official warning authority.
