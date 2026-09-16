# Experiments, results, and iterations

## Protocol

The final benchmark uses seeds 1000--1019 for every applicable
task/controller pair. Initial states are identical between controller variants
for a given task and seed. Reference scenarios are:

| Task | Robots | Maximum simulated time |
|---|---:|---:|
| Aggregation | 12 | 45 s |
| Coverage | 16 | 60 s |
| Transport | 12 | 35 s |

An episode ends early only after its sustained success condition is met.
Timeouts remain failures and retain the full horizon in time metrics. Summary
standard deviations are population standard deviations over 20 independent
seeds. Raw results are in [baseline_episodes.csv](results/baseline_episodes.csv),
with the machine-generated wide summary in
[baseline_episodes_summary.csv](results/baseline_episodes_summary.csv).

## Final results

| Task / controller | Success | Primary outcome | Time (s) | Collision rate |
|---|---:|---:|---:|---:|
| Aggregation / rules | 100% | component 1.000; radius 0.531 m | convergence 12.22 | 0.0000 |
| Aggregation / learned | 80% | component 0.950; radius 0.828 m | convergence 22.04 | 0.0005 |
| Aggregation / random | 0% | component 0.383; radius 1.471 m | timeout 45.00 | 0.0733 |
| Aggregation / stationary | 0% | component 0.429; radius 1.428 m | timeout 45.00 | 0.0000 |
| Coverage / rules | 100% | coverage 0.616; peak 0.635 | sustained 11.90 | 0.0096 |
| Coverage / random | 0% | coverage 0.441; peak 0.510 | timeout 60.00 | 0.1481 |
| Coverage / stationary | 0% | coverage 0.165 | timeout 60.00 | 0.0000 |
| Transport / rules | 100% | progress 0.887; 5.05 max pushers | delivery 13.77 | 0.1304 |
| Transport / random | 0% | progress 0.0004; 2.60 max pushers | timeout 35.00 | 0.1289 |
| Transport / stationary | 0% | progress 0.000 | timeout 35.00 | 0.0000 |

Values are means. For aggregation and coverage, an early successful stop reports
the state after the hold interval. Transport progress stops at the 0.35 m goal
tolerance, so successful progress is about 0.887 rather than 1.0.

These results support three narrow conclusions:

1. Reciprocal local spring/alignment terms reliably create one compact cluster;
   neither initial placement nor generic motion explains success.
2. Neighbor/wall repulsion produces substantially higher, safer spatial coverage
   than random walking from the same compact initial state.
3. Local staging and alignment create sustained simultaneous push groups. The
   physics threshold makes the task impossible for a single robot and random
   walkers almost never reach the required aligned effort.

They do not prove transfer to the articulated MuJoCo robot or physical hardware.

## Robot-count scaling

The transport rule was also evaluated on the same 20 held-out seeds with all
conditions fixed except swarm size. Raw data and the generated summary are in
[transport_scaling_episodes.csv](results/transport_scaling_episodes.csv) and
[transport_scaling_episodes_summary.csv](results/transport_scaling_episodes_summary.csv).

| Robots | Success | Progress | Cooperative-push fraction |
|---:|---:|---:|---:|
| 1 | 0% | 0.000 | 0.000 |
| 2 | 0% | 0.000 | 0.000 |
| 3 | 0% | 0.043 | 0.021 |
| 4 | 0% | 0.065 | 0.034 |
| 6 | 0% | 0.213 | 0.113 |
| 8 | 55% | 0.792 | 0.476 |
| 12 | 100% | 0.887 | 0.869 |

The mechanical minimum of three pushers is necessary but not sufficient: three
robots can occasionally align, yet cannot sustain enough collective effort to
deliver. Reliability rises sharply between six and twelve robots. This is a
stronger collective-capability result than comparing twelve robots with a
single scripted failure alone.

## Learning experiment

The aggregation policy was trained with seed 2026 for 12 generations, 24
candidates per generation, elite fraction 0.20, and training environment seeds
101, 202, and 303. Each training evaluation used 12 robots and 32 s. The best
training objective increased from 0.672 in generation 0 to 1.033 by generation
10. Full history is in
[aggregation_training.csv](results/aggregation_training.csv).

Held-out seeds 1000--1019 show that learning is real but not the best method:
80% success versus 0% random establishes that the small actor acquired
aggregation; 100% rule success, lower effort (0.649 versus 0.712), shorter paths
(2.89 versus 4.97 m/robot), and faster convergence select the rule as the current
default. This is an intentional negative selection result, not a hidden failure.

## Failures and design iterations

### Aggregation collision failure

The first spring/alignment controller aggregated on 9/10 calibration seeds but
averaged a 0.355 collision robot-step rate. Attraction could cancel within a
dense cluster. Adding an explicit close-range separation vector and increasing
preferred spacing produced 10/10 success with zero collisions on the same
calibration set. The held-out 20-seed result also has zero collisions.

### Coverage threshold mismatch

An initial 0.68 coverage condition was not reached. Ten calibration seeds showed
the rule plateaued near 0.653 peak while random walking peaked near 0.520. The
condition was changed to 0.60 with a separate minimum-spacing requirement and a
15-step hold. This is a task calibration, not a controller-specific shortcut:
on held-out seeds the rule succeeds 20/20 and both baselines 0/20.

### Learned-policy limitation

The memoryless linear policy occasionally leaves a disconnected component and
has worse compactness variance than the rule. More training could improve its
training score but would not yet justify a more complex default. A sensible next
learning experiment would add one small recurrent state or train across robot
counts, while holding task seeds, information, and metrics fixed.

### Transport contact cost

Transport has a roughly 0.13 robot-robot collision rate even for the successful
12-robot rule. Productive robot-load contact is recorded separately as pusher
utilization, but the kinematic exclusion model does not distinguish harmless
overlap corrections from high-impulse robot-robot impacts. The raw metric is
preserved; force-aware MuJoCo transfer is required before making a safety claim.

## Reproduction

```bash
bari-swarm benchmark \
  --tasks aggregation coverage transport \
  --controllers rules random stationary learned \
  --seeds 20 --seed-start 1000 \
  --policy models/swarm/aggregation_linear.json \
  --output docs/results/baseline_episodes.csv
```

This command writes the episode file and a sibling `_summary.csv`. It is
deterministic for the checked-in implementation and policy artifact.

Reproduce the scaling ablation with:

```bash
bari-swarm sweep --task transport --controller rules \
  --robot-counts 1 2 3 4 6 8 12 \
  --seeds 20 --seed-start 1000 --duration 35 \
  --output docs/results/transport_scaling_episodes.csv
```
