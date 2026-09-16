# BARI emergent swarm research platform

## Objective and design philosophy

The project studies whether sophisticated collective outcomes can arise when each
robot executes a small policy using only egocentric, finite-range information.
The objective is not to make motion that merely looks swarm-like. Every task has
an explicit success condition, privileged evaluation metrics, deterministic
seeds, and weak baselines.

The implementation follows five rules:

1. Controllers never receive robot IDs, absolute pose, a global map, other
   robots' actions, or swarm-level metrics.
2. Centralized state is permitted only in simulation mechanics, training scores,
   logging, and evaluation.
3. A collective claim needs a baseline and a metric; transport additionally has
   a hard multi-robot capability threshold.
4. Detailed physics and fast experimental iteration are different needs, so the
   existing MuJoCo robot is preserved beside a fast planar backend.
5. A learned method is selected only when evidence beats the simpler rule. The
   current learned aggregation policy is useful, but the rule remains the default.

## Platform at a glance

The repository now contains two complementary simulation levels:

- `bari_sim/` retains the three-link MuJoCo crawler, force/range sensing,
  directional contact, attachments, gap/step environments, logging, and manual
  or heuristic controllers.
- `bari_sim/swarm/` provides a deterministic bounded 2-D unicycle model designed
  for hundreds of headless episodes. It implements aggregation, coverage, and
  collective transport with rule, random, stationary, and learned controllers.

The planar level is an experimental abstraction, not a claim that the crawler is
a differential-drive robot. It answers controller and task-design questions
quickly; promising mechanisms should later be transferred to a steerable MuJoCo
morphology or hardware-calibrated model.

## Documentation map

- [Architecture](architecture.md): existing and new structure, policy boundary,
  morphology, dynamics, and ownership.
- [Reference analysis](reference-analysis.md): what was learned from PhySwarm,
  alternatives considered, and why this project takes a different route.
- [Tasks and controllers](tasks-and-controllers.md): local rules, learned policy,
  success criteria, and every metric.
- [Experiments](experiments.md): protocol, results, failed approaches, and design
  iterations.
- [Running and extending](extending.md): commands and contracts for adding tasks,
  policies, metrics, or higher-fidelity physics.
- [Raw episode results](results/baseline_episodes.csv) and
  [aggregated results](results/baseline_episodes_summary.csv).
- [Transport scaling episodes](results/transport_scaling_episodes.csv) and
  [scaling summary](results/transport_scaling_episodes_summary.csv).

## Fast start

```bash
python -m pip install -e ".[dev]"
bari-swarm run --task aggregation --controller rules --seed 7
bari-swarm run --task coverage --controller rules --seed 7
bari-swarm run --task transport --controller rules --seed 7
bari-swarm benchmark --seeds 10
bari-swarm sweep --task transport --controller rules \
  --robot-counts 1 2 3 4 6 8 12 --seeds 10
```

See [Running and extending](extending.md) for policy training, outputs, MuJoCo
commands, and extension steps.
