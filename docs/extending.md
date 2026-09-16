# Running, modifying, and extending

## Installation and common commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# One deterministic episode, printed as JSON
bari-swarm run --task aggregation --controller rules --seed 7
bari-swarm run --task coverage --controller random --seed 7
bari-swarm run --task transport --controller rules --robots 12 --duration 35

# Matched-seed study with raw and aggregate CSV output
bari-swarm benchmark --seeds 20 --seed-start 1000 \
  --output logs/my_study.csv

# Hold everything fixed while varying population size
bari-swarm sweep --task transport --controller rules \
  --robot-counts 1 2 3 4 6 8 12 --seeds 20 \
  --output logs/transport_scaling.csv

# Refit the small local aggregation actor
bari-swarm train --generations 12 --population 24 \
  --policy models/swarm/aggregation_linear.json \
  --history logs/aggregation_training.csv

# Use the learned actor
bari-swarm run --task aggregation --controller learned \
  --policy models/swarm/aggregation_linear.json
```

The original MuJoCo workflows remain available:

```bash
mjpython scripts/run_manual.py --robots 3 --environment flat
python scripts/run_gait_test.py --gait wave --duration 8
python scripts/run_attachment_test.py
python scripts/run_baseline.py --environment gap --controller bridge \
  --robots 2 --duration 20
```

Validation:

```bash
python -m pytest
ruff check bari_sim scripts tests
```

## Add a task

1. Implement the `EpisodeTask` protocol in `bari_sim/swarm/tasks.py`.
2. Define a `WorldConfig` and deterministic `reset()` distribution.
3. Convert task sensors to body-frame values in `observation()`. Do not expose
   world coordinates, agent index, swarm summaries, or success state.
4. Put load/object/terminal mechanics in `after_step()`, after every controller
   has acted from the same snapshot.
5. Declare a sustained success condition and return diagnostic metrics from
   `results()`; do not discard timeouts.
6. Register the task in `TASKS` and add reference robot count/duration in
   `SCENARIOS`.
7. Add stationary/random checks and at least one task-relevant invariant test.

If a task needs communication, use the bounded one-step `broadcast` channel and
document range, bandwidth, noise, and whether messages are anonymous. Do not
read the world arrays from a controller as a shortcut.

## Add or change a controller

Controllers implement two methods:

```python
def reset(self, seed: int) -> None: ...
def act(self, observation: LocalSwarmObservation) -> SwarmAction: ...
```

Add the class to `controllers.py` and register a factory. Keep state private to
one controller instance; the episode runner creates one instance per robot.
When comparing a learned method, keep the task, seeds, observation information,
action limits, horizon, and success condition fixed. Save an observation-layout
version with learned weights.

## Add a metric

Pure reusable geometry/graph metrics belong in `swarm/metrics.py`. Task-specific
accumulation belongs in the task. A metric must state:

- unit and direction (higher/lower is better);
- sampling denominator, especially for collisions;
- handling of early termination and failures;
- whether it uses privileged state;
- any threshold and hold time.

Add it to per-episode results first; the benchmark summarizer automatically
emits mean and population standard deviation columns.

## Change planar morphology or dynamics

`WorldConfig` owns arena and robot limits. Changes to `PlanarWorld.step()` alter
all tasks and should be treated as a new experimental condition, not mixed into
a controller comparison. Add conservation/bounds/determinism tests and regenerate
benchmark artifacts.

Obstacles, differential-wheel acceleration, sensor noise, failures, or explicit
packet loss are reasonable next additions. Each should remain simulator-owned
and surface only local consequences to controllers.

## Transfer a mechanism to MuJoCo

Do not feed MuJoCo world pose into the planar controller. Instead, build an
adapter that derives equivalent neighbor vectors, wall/object cues, and contact
bits from simulated sensors. A steerable crawler morphology will likely require
yaw joints, asymmetric gait steering, or a modular drive unit. Validate in this
order:

1. isolated turning and forward-speed envelopes;
2. local sensor equivalence and noise;
3. two-robot separation/cohesion interactions;
4. one task on matched initial layouts;
5. task metrics over seeds, including attachment/contact failures.

The planar success threshold should not be copied numerically into MuJoCo when
geometry or sensing differs. Preserve the metric meaning and predeclare a new
physical tolerance.

## Stop conditions for future work

A feature is ready when its task is reproducible, local information is audited,
weak baselines cannot exploit the setup, results persist per seed, and focused
plus full regression tests pass. Add complexity only when an observed failure
identifies what the new state, model, sensor, or learning capacity must solve.
