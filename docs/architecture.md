# Architecture

## Existing BARISimulation architecture

Before this work, BARISimulation was a focused MuJoCo feasibility prototype for
a thin articulated crawler swarm:

```text
TOML config
  -> ModelBuilder generates MJCF
  -> SwarmSimulator owns MuJoCo state and stepping
       -> Robot applies joint/attachment actions
       -> SensorSuite builds local force/range observations
       -> AttachmentManager and DirectionalFrictionModel apply mechanics
       -> EnvironmentEvaluator and MetricsLogger inspect privileged state
  -> manual, gait, exploration, or bridge controller
```

This structure already had a valuable invariant: `LocalObservation` excludes
world pose, while `GlobalRobotState` is a separate evaluation interface. It also
had realistic actuator saturation, contact-derived sensing, breakable
attachments, and regression tests. Those capabilities were preserved.

The limiting factor for the new objective was not code quality but morphology
and experiment throughput. The crawler has pitch joints but no direct yaw
steering. Its current calibrated gait is principally one-dimensional, and a
large MuJoCo sweep is unnecessarily expensive while task/controller contracts
are still changing. Building foraging-like two-dimensional tasks directly into
that model would have hidden the real steering limitation behind scripted world
logic.

## Two-level experimental architecture

The platform therefore separates fast emergence studies from detailed physical
validation:

```text
                         decentralized execution
task observation(world, i) -> LocalSwarmObservation[i]
                              -> controller[i]
                              -> SwarmAction[i]
                              -> PlanarWorld.step(all actions)
                              -> task mechanics and privileged metrics

matched seeds -> benchmark -> per-episode CSV -> aggregate CSV -> documentation
training seeds -> CEM       -> 24 parameters  -> held-out benchmark

promising task/controller mechanism
    -> future steerable MuJoCo morphology and contact-level validation
```

Module ownership is deliberately small:

| Module | Responsibility |
|---|---|
| `swarm/types.py` | Immutable local observation and normalized action contracts |
| `swarm/core.py` | Bounded unicycle dynamics, collision exclusion, local sensing, episode lifecycle |
| `swarm/tasks.py` | Initial state, task-local cues, collective mechanics, success conditions |
| `swarm/controllers.py` | Independent rules, weak baselines, linear learned actor |
| `swarm/metrics.py` | Pure privileged metric functions |
| `swarm/experiments.py` | Matched-seed benchmarks, summaries, CEM training, CSV output |
| `swarm/cli.py` | `run`, `benchmark`, and `train` commands |

## Policy information boundary

`LocalSwarmObservation` contains only:

- time;
- finite-range neighbors as body-frame relative positions and headings;
- distance to the rectangular boundary along front/left/right/rear body rays;
- an opaque task cue made only from egocentric sensors;
- a task contact Boolean.

Neighbor readings deliberately omit identity. The contract contains no absolute
position, absolute heading, goal coordinates, map, robot count, global density,
component membership, success flag, or metric. Optional broadcasts are received
only from currently sensed neighbors and are one step delayed.

Tasks and evaluators may inspect world state only after all controllers have
selected actions from the same snapshot. This synchronous ordering prevents one
controller from observing another controller's current action. Tests verify the
surface contract, its rigid-transform invariance for neighbor readings, episode
determinism, and collective thresholds.

## Chosen planar robot and environment

The fast robot is a disc with unicycle kinematics:

```text
heading(t+dt) = heading(t) + normalized_turn * max_turn_rate * dt
position(t+dt) = position(t)
               + normalized_speed * max_speed * heading_unit * dt
```

Default radius is 0.10 m, timestep 0.1 s, speed 0.35 m/s (0.38 m/s for
transport), turn rate 2.4 rad/s, and neighbor sensing 1.1--1.25 m depending on
the task. Robots cannot overlap; collision resolution is simulator
infrastructure and collision exposure is recorded as a metric.

Rectangular walls and rays make boundary avoidance local. There is no occupancy
map, GPS, all-to-all communication, leader, scheduler, auctioneer, or task
allocation service. Transport supplies an egocentric goal-beacon direction and
a finite-range object sensor, both plausible onboard modalities. Object motion
is produced only by thresholded simultaneous push actions.

## Why the two morphologies coexist

The abstract robot offers steerability and throughput needed to discover useful
collective mechanisms. The articulated MuJoCo robot retains the harder questions
of gait production, force transmission, attachment failure, stacking, and gap
crossing. Replacing it would have discarded working functionality; claiming the
planar results transfer unchanged would overstate evidence. The intended
research loop is:

1. establish task and decentralized-policy evidence in the planar backend;
2. select mechanisms using multi-seed metrics and ablations;
3. add or identify a steerable articulated morphology;
4. port one controller at a time through a sensor/action adapter;
5. re-run the same success criteria with physically appropriate tolerances.
