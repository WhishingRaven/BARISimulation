# Reference and approach analysis

## PhySwarm ideas that informed this project

The sibling `../PhySwarm` repository was inspected before the new architecture
was selected. Its strongest reusable ideas are methodological rather than code
to copy:

- Three distinct task families share an experiment lifecycle while scenario
  specifications own phase, field, transition, observation, and action meaning.
- Actor observations remain decentralized even when a centralized critic or
  privileged physics side channel is used during training.
- Pure task/physics/metric modules are separated from Webots adapters and the
  recurrent MAPPO implementation.
- Evaluation records smoothness, collisions, efficiency, formation error,
  navigation time, relay connectivity, failures, and censored timeouts instead
  of relying on a visual rollout.
- Comparisons preserve per-seed results, use matched conditions, include weak
  and ablated baselines, and keep raw artifacts separate from summaries.
- Learned micro-dynamics and macro density/ADR residuals are treated as optional
  training structure with named diagnostic terms, not as task success metrics.

BARISimulation adopts the shared lifecycle, strict actor/evaluator separation,
named task contracts, matched-seed comparisons, and raw-plus-summary artifacts.
The code is independent because the simulator, morphology, dependencies, and
research question differ.

## What was not reproduced

PhySwarm uses Webots E-pucks, recurrent MAPPO, task-specific action heads,
centralized value learning, density reconstruction, and micro/macro
physics-informed loss terms. Importing that stack would add PyTorch, Webots
process orchestration, checkpoint compatibility, and several layers of learned
parameter semantics before BARISimulation had validated tasks.

That would not resolve the existing crawler's lack of planar steering, and it
would make it difficult to tell whether behavior came from local interaction or
network capacity. The initial BARISimulation learning baseline is therefore a
24-parameter shared memoryless actor trained by a dependency-free cross-entropy
method. It is intentionally small enough to inspect and compare with rules.

PhySwarm's foraging/navigation/rescue scenarios were also not cloned. The new
tasks test different primitives—cohesive self-organization, distributed spatial
dispersion, and simultaneous force production—while retaining the broader idea
that one platform should support multiple measurable collective problems.

## Other approaches considered

| Approach | Decision and reason |
|---|---|
| Boids-style cohesion, separation, alignment | Used for aggregation because each term is local and ablatable. |
| Artificial-potential coverage | Used with local neighbor and wall repulsion; no global Voronoi diagram is exposed to controllers. |
| Thresholded cooperative manipulation | Used because it supplies an unambiguous single-agent impossibility result. |
| Quorum site selection | Deferred. It is valuable, but needs a communication/noise protocol and adds a fourth task before the physical task suite is validated. |
| Relay-chain rescue | Deferred for this iteration. A credible version needs explicit packet propagation and robust graph-connectivity dynamics, not a decorative chain animation. |
| Behavior trees or centralized assignment | Rejected as defaults because role allocation would be prescribed rather than emergent. |
| Full MARL/MAPPO | Deferred until a compact policy fails on a task where memory or nonlinear representation is demonstrably needed. |
| Graph neural controllers | Deferred for the same reason and because variable-neighbor aggregation can hide substantial per-agent complexity. |
| Direct MuJoCo-only task sweeps | Deferred until steerability is added; current pitch-only locomotion is not a sound base for 2-D coverage or transport routing. |
| Continuum/PDE regularization | Kept as a future option for scale generalization; present tasks do not yet justify its training and diagnostic complexity. |

## Selection rule

The simplest controller that meets the predeclared task condition on held-out
seeds remains the reference. Learning is useful when it improves success,
robustness, energy, or scaling enough to justify added complexity. Current
evidence selects local rules for all three tasks; the learned aggregation actor
is retained as a reproducible research baseline because it clearly learns a
collective behavior, even though it does not win.
