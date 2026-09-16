# BARI emergent swarm simulation

Research platform for studying how simple local robot policies produce measurable collective behavior. It combines fast planar multi-seed experiments with the existing MuJoCo framework for testing whether thin articulated robots can locomote, climb, stack, anchor, bridge gaps, and carry structural load.

The planar backend implements aggregation, distributed coverage, and thresholded collective transport with strict egocentric observations, quantitative metrics, matched-seed baselines, and a lightweight learned policy. The MuJoCo model remains a configurable three-link crawler with force/range sensors and a front gripper. `example.png` informed only its qualitative layout; exact hardware performance is not claimed.

## Emergent swarm experiments

```bash
python -m pip install -e ".[dev]"
bari-swarm run --task aggregation --controller rules --seed 7
bari-swarm run --task coverage --controller rules --seed 7
bari-swarm run --task transport --controller rules --seed 7
bari-swarm benchmark --seeds 10
```

The transport load requires at least three simultaneous aligned pushers, so a
single robot cannot solve it. Controllers receive no IDs, absolute pose, global
map, or swarm metrics. See [the research documentation](docs/README.md) for the
architecture, PhySwarm analysis, task definitions, experiment results, failures,
and extension guide.

## Install and run on macOS

MuJoCo's passive viewer must run through `mjpython` on macOS.
Current verification target is MuJoCo 3.13.x.
Headless shells can expose no GLFW monitor; run manual viewer from a logged-in desktop terminal and use headless scripts in CI.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
mjpython scripts/run_manual.py --robots 3 --environment flat
```

`--robots` means formation rows, not total units. Every row contains five robots, so `--robots 3` creates 15 robots in a dense 3 x 5 formation. IDs 1-5 form the front row, IDs 6-10 the next row, and IDs 11-15 the rear row. Adjacent columns use exactly one body width (0.110 m) between centerlines; adjacent rows use exactly one straight robot length (0.590 m). Both configured gaps are zero, so collision envelopes start in contact without overlap.

Gap and step shortcuts:

```bash
mjpython scripts/run_gap.py --robots 2
mjpython scripts/run_step.py --robots 2
```

Headless experiments do not need a display:

```bash
python scripts/run_gait_test.py --gait wave --duration 8
python scripts/run_gait_test.py --gait inchworm --duration 8
python scripts/run_attachment_test.py
python scripts/run_baseline.py --environment gap --controller bridge --robots 2 --duration 20
```

Add `--viewer` to gait or baseline scripts and launch them with `mjpython` on macOS. Export exact generated MJCF for inspection:

```bash
python scripts/export_model.py --environment gap --robots 2
```

## Manual controls

The robot's front is its gripper end and local `+X`. `W` always requests motion toward that front; `S` requests motion toward the rear. A gait keeps running after the key is released.

| Key | Detailed behavior |
|---|---|
| `1`-`9`, `0` | Select robot 1-10. Selection changes commands and overlay text only; it never changes model visibility. |
| `B` / `N` | Select previous/next robot, wrapping at the ends. Useful when more than ten robots exist. |
| `W` | Start the selected robot's forward wave in local `+X`. The phase restarts smoothly when changing direction. |
| `S` | Start its reverse wave in local `-X`. |
| `C` | Stop any gait and hold the current joint angles. |
| `Q` / `A` | Increase/decrease joint 1 target by 5 degrees and enter manual joint mode. |
| `E` / `D` | Increase/decrease joint 2 target by 5 degrees and enter manual joint mode. Extra joints remain held. |
| `I` | Toggle the selected robot's attachment-seeking inchworm state machine. |
| `Space` | Toggle the selected gripper: attach at its current contact point, or detach if already attached. A request without contact is safely rejected and logged. |
| `X` | Force a detach request; useful when attachment state is hard to see. |
| `P` | Pause/resume physics. Joint targets can still be edited while paused. |
| `R` | Reset the whole simulation, controllers, attachments, and joint targets. |
| `-` / `+` | Change playback among 0.25x, 0.5x, 1x, 2x, and 4x. `[` / `]` remain aliases. This changes wall-clock playback, not gait dynamics. |
| `V` | Toggle changing range-sensor rays. They are hidden by default for a stable view. |

Viewer overlay shows selected robot, persistent controller mode, pause/playback state, debug state, attachment load, and last command. World labels identify every robot. Active attachments remain visible as a red line and contact point. Other robots keep their current modes when selection changes, enabling cooperative teleoperation.

MuJoCo also assigns several manual-control keys to rendering shortcuts: number keys toggle rendering groups, `W` toggles wireframe, and `S` toggles shadows. The visualizer therefore reapplies a fixed opaque presentation before every viewer synchronization. Environment, colored robot shells, shadows, and markers stay visible; wireframe and nearly transparent collision shells stay hidden. Repeated movement or robot-selection keys can no longer alter rendering. Contact points, constraints, sensor rays, and the right-side MuJoCo UI are off by default to avoid flicker and clutter.

## Robot detailed specification

These are current simulation parameters, not measured hardware specifications. The straight robot points along local `+X`, from rear link 1 to the front gripper.

| Subsystem | Current model |
|---|---|
| Topology | Three serial rigid links, two actuated pitch hinges, and a six-DOF free root |
| Straight dimensions | 0.590 m overall length including gripper; 0.110 m body width; 0.00625 m body thickness |
| Link geometry | 0.180 / 0.200 / 0.180 m long boxes; separate opaque visual and full-size collision geoms |
| Mass | 0.075 / 0.090 / 0.075 kg; 0.240 kg total; each COM at geometric center |
| Inertia | Analytic uniform-box diagonal inertia, calculated independently for each link |
| Joint axes | Local `+Y`, producing pitch in the local X-Z plane |
| Joint limits | -135 to +135 degrees on both joints |
| Joint output limits | 0.35 N m maximum torque; 40.0 rad/s commanded speed limit, four times the prior 10.0 rad/s limit |
| Position servo | Torque-producing PD control: 4.0 N m/rad proportional gain and 0.12 N m s/rad derivative gain |
| Passive joint terms | 0.030 N m s/rad damping, zero stiffness, 0.00002 kg m2 armature |
| Ground-contact mechanism | Two thin, visible underside pads per link, entirely within the 0.110 m body envelope. Contacts use low forward-recovery and high backward-slip resistance while ordinary sliding friction is disabled. |
| Anchor-pad coefficients | Forward recovery 0.20, backward slip 2.00, lateral 0.90; configured in `robot.directional_friction` |
| Front gripper | 0.030 x 0.035 x 0.018 m collision envelope on the front link |
| Attachment | Arbitrary contact-point connection; 2.354 N force-only break threshold by default; environment and other robots allowed |
| Range sensing | Forward, forward-left, forward-right, and forward-down rays; 0.60 m range; 50 Hz; ideal zero-noise default |
| Surface-force sensing | 12 scalar normal-force channels per robot: two top patches and one left/right patch on each of three links |
| Default wave | 40.1-degree amplitude, 1.5 Hz, 75-degree inter-joint phase, 0.15 s smooth startup; the center link is the swing link while front and rear links remain the two gripping stance links |
| Default spawn | 3 rows x 5 columns = 15 robots; zero edge gap; front-row rear-link origin at X = -0.55 m; 0.004 m initial ground clearance |

The thin anchor pads are active directional-contact elements: manual `W` or `S` orients their high-resistance direction to match the requested gait direction, so reverse locomotion remains available. They still apply force only at a real pad contact; no root translation or external thrust is injected. Dense neighbors change the result through physical contact. This is a simulation calibration result under guessed mass/contact parameters, not a hardware performance claim.

### Force-sensor layout

Every link has four visible sensing patches and four local observation channels:

- `link_N.top_rear`: rear half of the upper surface;
- `link_N.top_front`: front half of the upper surface;
- `link_N.left`: full left side;
- `link_N.right`: full right side.

For three links this produces exactly 12 channels. Each returns normal force in newtons through `LocalObservation.force_sensors`; pressure and the old bottom/front/rear aggregate regions were removed. Ground load on the unsensored lower face is intentionally absent. `gripper_contact` remains a separate Boolean because attachment control still needs it. MuJoCo touch-site objects define and render the physical patches, while contact normals route corner contacts to one face so top and side channels do not double-count the same load.

## Physics behavior

- Motors receive torque only. Position control is a target-rate-limited PD controller whose output is clipped by configured actuator torque. Drive torque that would accelerate farther beyond configured joint speed is cut off; braking remains available and external loads can still back-drive faster. `Robot.set_joint_target()` and `Robot.set_joint_torque()` expose both low-level modes.
- Traveling-wave and inchworm controllers command joints/attachments only. They never edit root pose or velocity. Failed or backward locomotion is logged unchanged.
- Positive gait direction is defined as robot-local `+X`. Because forward body motion is opposite the traveling shape wave, the controller intentionally propagates its joint phase toward the rear. Automated flat-ground tests require forward and reverse commands to produce opposite signed displacement. Different hardware friction can still require physical recalibration.
- Default locomotion uses two visible, thin underside pads per link. Each pad remains within the body-width collision envelope, avoiding the artificial side impacts caused by protruding pads. Their contacts have zero built-in sliding friction; `DirectionalFrictionModel` supplies a smooth asymmetric Coulomb reaction only at scheduled stance-pad contacts. In the default three-link gait, the front and rear links are the two stance links and the center link is the swing link. The coefficient is low when a pad recovers in the selected travel direction and high when it slips opposite it. `W`/`S` explicitly orient the pads with the requested wave, preserving both directions without moving a root body directly.
- Each source gripper has inactive MuJoCo point-constraint slots to every link on every other robot and to the environment. On contact, the contact point is transformed into the target body's local frame, its target site is moved there, and only that equality is activated. No docking socket is required.
- Attachment reaction load comes from equality-constraint solver forces. Default failure is force-only at about one robot weight. A combined force/moment criterion is implemented behind the same interface; point connections currently transmit no reported moment.
- Gripper teeth, penetration, material deformation, electrical behavior, and CAD fidelity are deliberately outside this prototype.

The automated attachment test anchors a contacting gripper to the flat environment and ramps an external load. Multi-robot arbitrary-link attachment is covered by runtime tests and available through the same `robot.attach()` call during teleoperation.

## Local policy boundary

`Robot.get_observation()` returns only:

- joint positions and velocities;
- 12 link-surface force readings: two top and one left/right channel per link;
- forward, forward-left, forward-right, and forward-down ray ranges;
- local robot detections from contact/rays;
- attachment state, force, moment, and utilization.

It does **not** return absolute position, orientation, or a global map. `SwarmSimulator.global_state()` is a separate privileged interface for evaluation, logging, rewards, and visualization. Controllers in `bari_sim/controllers/` accept `LocalObservation` only; none constructs a centralized bridge plan.

Potential future wrapper flow remains direct:

```text
LocalObservation[i] -> policy[i] -> RobotAction[i] -> SwarmSimulator.step()
GlobalRobotState     -> diagnostics/reward/evaluation only
```

## Environments and metrics

- `flat`: infinite plane for contact, friction, gait, and interaction tests.
- `gap`: two finite platforms with configurable width, height, dimensions, and friction.
- `step`: ground plus a configurable obstacle taller than one link's thickness and ordinary clearance.

CSV state logs include time, global diagnostic pose/velocity, joint state, torque commands, the 12 force-sensor readings, ranges, attachment load/utilization, maximum structural load, and gap/obstacle crossing flags. Separate event CSV files record attach, detach, missed attach, and load failure events. Files are written under `logs/` unless `--no-log` is used.

## Configuration and calibration status

All values use SI units. Runtime model values come only from TOML files in `config/`; generated XML is an inspectable artifact, not a second configuration source.

| Quantity | Default | Status / measurement needed |
|---|---:|---|
| Total mass | 0.240 kg | Guessed; weigh assembled prototype |
| Link masses | 0.075/0.090/0.075 kg | Guessed; weigh links separately |
| Link lengths | 0.18/0.20/0.18 m | Estimated qualitatively from photo; measure pivots/endpoints |
| Width/thickness | 0.110/0.00625 m | Width doubled for lateral stability; thickness reduced to one quarter; measure collision envelope |
| Gripper collision envelope | 0.030 x 0.035 x 0.018 m | Estimated; replace after mechanism characterization |
| Center of mass | Geometric center | Unknown; balance or identify each link |
| Inertia | Analytic uniform boxes | Placeholder; pendulum/CAD estimate later |
| Joint range | +/-135 deg | Expanded simulation range; measure hard stops |
| Maximum torque | 0.35 N m | Guessed; use actuator/gearbox data and bench test |
| Maximum speed | 40.0 rad/s | Requested four-times-speed hypothesis; unloaded and loaded test needed |
| Position-control gains | 4.0 N m/rad, 0.12 N m s/rad | Stable starting guess; tune against step response |
| Joint damping/stiffness/armature | 0.030 N m s/rad / 0 / 0.00002 kg m2 | Guessed; system identification needed |
| Anchor-pad friction | 0.20 forward / 2.00 backward / 0.90 lateral | Directional-contact hypothesis; calibrate with drag tests |
| Attachment break force | 2.354 N | Intentional one-weight starting hypothesis; tensile test needed |
| Attachment moment capacity | 0.080 N m | Placeholder for combined criterion; bend test needed |
| Ultrasonic range/noise | 0.60 m / 0 | Guessed/idealized; sensor characterization needed |

Edit `config/robot.toml`, `config/environment.toml`, and `config/simulation.toml`. Changing link count also requires matching link lengths, masses, COM offsets, and `link_count - 1` joint-limit entries; model generation itself supports this without controller redesign.

## Architecture

```text
config/                      assumed physical and run parameters
bari_sim/model_builder.py    MJCF morphology, environments, constraint slots
bari_sim/robot.py            actuator-level robot API
bari_sim/attachment.py       contact selection, local anchors, failure criteria
bari_sim/sensors.py          12 local surface-force channels and ray sensors
bari_sim/friction.py         optional explicit directional-friction hypothesis
bari_sim/simulator.py        physics orchestration and multi-agent boundary
bari_sim/environment.py      privileged crossing evaluation
bari_sim/metrics.py          CSV state/event logging
bari_sim/controllers/        manual, gait, inchworm, exploration, bridge heuristics
bari_sim/swarm/              fast planar tasks, local policies, metrics, experiments
scripts/                     manual and feasibility experiment entry points
models/                      generated MJCF/CAD assets and learned policy artifacts
environments/                environment design notes
docs/                        design, task, experiment, result, and extension records
tests/                       policy contracts, task behavior, compile, physics tests
```

Run verification:

```bash
python -m pytest
ruff check bari_sim scripts tests
python scripts/run_gait_test.py --duration 1 --no-log
python scripts/run_attachment_test.py --ramp-duration 1 --no-log
```

## Research interpretation

Default parameters answer only “what happens under these explicit hypotheses?” They do not validate real hardware. Use parameter sweeps over torque, friction, attachment strength, gap width, and step height; preserve failures as results. Before publishing feasibility claims, replace guessed values with measurements and report sensitivity to remaining uncertainty.
