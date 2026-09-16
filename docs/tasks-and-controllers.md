# Tasks, local control, and metrics

## Common controller model

Every robot runs the same controller class but has private state and, where
needed, a private seeded wander phase. Controllers receive one
`LocalSwarmObservation` and return normalized forward speed, turn rate, and an
optional broadcast. There is no leader and no per-robot plan.

The stationary and persistent random-walk controllers are weak controls. They
detect success caused by favorable initialization, thresholds, or arena
mechanics. The rule controller is task-specific because local information has
different physical meaning in each task.

## Aggregation

Twelve robots start randomly dispersed in a 5.5 x 4.0 m arena. No target, beacon,
or global center exists.

Local rule:

- with neighbors, sum a preferred-distance spring, explicit close-range
  separation, and neighbor-heading alignment;
- near a wall, add body-ray repulsion;
- with no neighbors, follow a private slow curved search trajectory.

Clusters arise from reciprocal local attraction. Separate components merge only
when searching boundary robots come within sensing range; nobody knows the
centroid or component graph.

Success requires all robots in one 0.78 m proximity component and RMS centroid
radius at most 0.86 m for 20 consecutive steps.

Metrics:

- largest connected-component fraction;
- RMS swarm radius;
- heading polarization;
- convergence time, including the hold interval start;
- collision robot-step rate, mean command effort, and path length;
- an aggregation score used for policy search, combining connectivity,
  compactness, and collision cost.

## Distributed coverage

Sixteen robots start in a compact central patch of a 6 x 4 m arena. A robot has
only finite-range neighbors and four wall rays; it has no map, cell ownership,
global density, or Voronoi computation.

Local rule:

- repel neighbors within 1.05 m;
- repel locally sensed walls;
- otherwise maintain forward motion with a small private oscillatory turn.

Mutual repulsion converts a compact swarm into a spatial distribution. Boundary
feedback keeps agents in the arena without telling them where unvisited cells
are.

Evaluation overlays an 18 x 12 privileged grid. A cell is covered when it lies
within a 0.62 m sensor footprint of any robot. Success requires at least 0.60
coverage and 0.42 m mean nearest-neighbor spacing for 15 consecutive steps.

Metrics:

- final and peak grid coverage fraction;
- mean nearest-neighbor spacing;
- time to sustained coverage;
- collision rate, effort, and path length;
- coverage minus a collision penalty.

The grid is evaluation-only. Supplying cell gradients to the controller would
turn this into centralized coverage planning.

## Collective transport

Twelve robots start behind and around a circular load. The task supplies a
finite-range egocentric object vector, a local contact bit, and an egocentric
goal-beacon direction. Robots do not observe object coordinates, global load
velocity, number of pushers, or remaining task distance.

Local rule:

- move to a staging point behind the sensed load relative to the beacon;
- use neighbor separation to avoid selecting the same physical point;
- on contact, align with the beacon and push;
- if the object is not visible, search in the beacon direction.

The load moves only when at least three robots push simultaneously and their
combined goal-aligned normalized effort is at least 1.65. A one-robot run cannot
move it, regardless of policy. Motion is capped at 0.20 m/s and terminates within
0.35 m of the goal.

Metrics:

- delivery success and time;
- normalized transport progress and remaining goal distance;
- maximum simultaneous pushers;
- cooperative-push fraction and pusher utilization;
- mean push alignment;
- collision rate, command effort, and path length.

## Lightweight learned aggregation

`LinearLocalController` summarizes an observation into 12 local features:
cohesion, close separation, heading alignment, wall avoidance, local density,
nearest-neighbor distance, and neighbor absence, plus a bias. A shared 2 x 12
weight matrix produces speed and turn commands. It has no recurrence and 24
trainable numbers.

The cross-entropy method samples parameter populations, evaluates three matched
training seeds, retains the top 20%, and refits a diagonal Gaussian with a small
variance floor. The global aggregation score is allowed during training, just as
a centralized critic may use privileged state. The saved actor still consumes
only the strict local observation at execution time.

The learned artifact is `models/swarm/aggregation_linear.json`. It is versioned
by controller kind and feature count so a future observation change fails
explicitly instead of silently loading incompatible weights.
