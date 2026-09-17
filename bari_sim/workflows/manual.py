"""Keyboard teleoperation for the declared discrete action space."""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import SimpleQueue

import mujoco

from ..robot import GripAction, LiftAction, MotionAction, RobotAction
from ..simulation import Simulation


@dataclass
class ManualState:
    selected_robot_id: int = 0
    paused: bool = False
    reset_requested: bool = False
    message: str = "ready"


@dataclass(frozen=True)
class ManualCameraPose:
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float = 90.0
    elevation: float = -65.0


_ROBOT_VISUAL_FLAGS = (
    mujoco.mjtVisFlag.mjVIS_AUTOCONNECT,
    mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
    mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT,
    mujoco.mjtVisFlag.mjVIS_ISLAND,
    mujoco.mjtVisFlag.mjVIS_LIGHT,
    mujoco.mjtVisFlag.mjVIS_PERTFORCE,
    mujoco.mjtVisFlag.mjVIS_STATIC,
    mujoco.mjtVisFlag.mjVIS_TEXTURE,
)
_ROBOT_RENDER_FLAGS = (
    mujoco.mjtRndFlag.mjRND_REFLECTION,
    mujoco.mjtRndFlag.mjRND_SHADOW,
    mujoco.mjtRndFlag.mjRND_WIREFRAME,
)
_MOTION_KEYS = {
    "W": MotionAction.CURL_BODY,
    "S": MotionAction.FLATTEN_BODY,
    "A": MotionAction.TURN_LEFT,
    "D": MotionAction.TURN_RIGHT,
}
_MAC_KEY_CODES = {"A": 0, "S": 1, "D": 2, "W": 13}
_GLFW_KEYPAD_DIGITS = {320 + digit: str(digit) for digit in range(10)}


@dataclass(frozen=True)
class ManualShortcutDefaults:
    geom_groups: tuple[int, ...]
    visual_flags: tuple[int, ...]
    render_flags: tuple[int, ...]

    @classmethod
    def capture(cls, viewer) -> ManualShortcutDefaults:
        return cls(
            geom_groups=tuple(int(value) for value in viewer.opt.geomgroup),
            visual_flags=tuple(
                int(viewer.opt.flags[int(flag)]) for flag in _ROBOT_VISUAL_FLAGS
            ),
            render_flags=tuple(
                int(viewer.user_scn.flags[int(flag)]) for flag in _ROBOT_RENDER_FLAGS
            ),
        )

    def restore(self, viewer) -> None:
        # 0-5 and most letter keys are MuJoCo visualization shortcuts.  In
        # manual mode those same keys belong to robot selection/control, so
        # keep the viewer values at their launch defaults.
        viewer.opt.geomgroup[:] = self.geom_groups
        for flag, value in zip(_ROBOT_VISUAL_FLAGS, self.visual_flags, strict=True):
            viewer.opt.flags[int(flag)] = value
        for flag, value in zip(_ROBOT_RENDER_FLAGS, self.render_flags, strict=True):
            viewer.user_scn.flags[int(flag)] = value


class ManualKeyPoller:
    """Poll motion-key state when the passive viewer omits key-up events."""

    def __init__(self, key_state: Callable[[str], bool] | None = None):
        self._key_state = key_state or self._platform_key_state()

    def pressed_motion_keys(self) -> set[str]:
        return {key for key in _MOTION_KEYS if self._key_state(key)}

    @staticmethod
    def _platform_key_state() -> Callable[[str], bool]:
        if sys.platform != "darwin":
            # On other platforms, native key-repeat callbacks still produce
            # finite motion pulses, but exact release polling is unavailable.
            return lambda _key: False
        core_graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        key_state = core_graphics.CGEventSourceKeyState
        key_state.argtypes = (ctypes.c_uint32, ctypes.c_uint16)
        key_state.restype = ctypes.c_bool
        return lambda key: bool(key_state(0, _MAC_KEY_CODES[key]))


class ManualController:
    def __init__(self, robot_count: int):
        self.robot_count = robot_count
        self.state = ManualState()
        self._actions = {robot_id: RobotAction() for robot_id in range(robot_count)}
        self._keys: SimpleQueue[str] = SimpleQueue()
        self._held_motion: tuple[int, str] | None = None
        self._latched_turn: tuple[int, MotionAction] | None = None
        self._held_key_misses = 0
        self._step_pending = False
        self._halt_requested = False
        self._root_motion_lock_pending = False
        self._active_actions = dict(self._actions)
        self._last_overlay_payload: tuple[str, str, str] | None = None

    def key_callback(self, keycode: int) -> None:
        try:
            self._keys.put(_GLFW_KEYPAD_DIGITS.get(keycode, chr(keycode)))
        except (ValueError, OverflowError):
            return

    def process_keys(self) -> None:
        while not self._keys.empty():
            self.handle_key(self._keys.get())

    def handle_key(self, raw_key: str) -> None:
        key = raw_key.upper()
        if raw_key.isdigit():
            selected = 9 if raw_key == "0" else int(raw_key) - 1
            if selected < self.robot_count:
                self.state.selected_robot_id = selected
                self._latched_turn = None
                self.state.message = f"selected robot {selected}"
            return
        if key in {"B", "N"}:
            offset = -1 if key == "B" else 1
            self.state.selected_robot_id = (
                self.state.selected_robot_id + offset
            ) % self.robot_count
            self._latched_turn = None
            self.state.message = f"selected robot {self.state.selected_robot_id}"
            return
        robot_id = self.state.selected_robot_id
        current = self._actions[robot_id]
        if key in {"A", "D"}:
            action = RobotAction(_MOTION_KEYS[key], current.lift, current.grip)
            # Native key-state polling is not reliable in every passive
            # MuJoCo viewer.  Turn is therefore latched by key-down and runs
            # until an explicit stop or another physical command.
            self._held_motion = None
            self._held_key_misses = 0
            self._latched_turn = (robot_id, action.motion)
            self._step_pending = True
            self.state.message = action.motion.name
        elif key in {"W", "S"}:
            action = RobotAction(_MOTION_KEYS[key], current.lift, current.grip)
            self._latched_turn = None
            if self._held_motion != (robot_id, key):
                self._held_motion = (robot_id, key)
                self._held_key_misses = 0
                self._step_pending = True
            self.state.message = action.motion.name
        elif key == "C":
            action = RobotAction(MotionAction.STOP, current.lift, current.grip)
            self._held_motion = None
            self._latched_turn = None
            self._held_key_misses = 0
            self._halt_requested = True
            self.state.message = "STOP"
        elif key == "R":
            action = RobotAction(MotionAction.STOP, LiftAction.LIFT_FRONT, current.grip)
            self._latched_turn = None
            self._root_motion_lock_pending = True
            self._step_pending = True
            self.state.message = "LIFT_FRONT"
        elif key == "F":
            action = RobotAction(
                MotionAction.STOP, LiftAction.UNLIFT_FRONT, current.grip
            )
            self._latched_turn = None
            self._root_motion_lock_pending = True
            self._step_pending = True
            self.state.message = "UNLIFT_FRONT"
        elif raw_key == " ":
            action = RobotAction(MotionAction.STOP, current.lift, GripAction.ATTACH)
            self._latched_turn = None
            self._root_motion_lock_pending = True
            self._step_pending = True
            self.state.message = "ATTACH"
        elif key == "X":
            action = RobotAction(MotionAction.STOP, current.lift, GripAction.DETACH)
            self._latched_turn = None
            self._root_motion_lock_pending = True
            self._step_pending = True
            self.state.message = "DETACH"
        elif key == "P":
            self.state.paused = not self.state.paused
            self.state.message = "paused" if self.state.paused else "running"
            return
        elif key == "Z":
            self.state.reset_requested = True
            self.state.message = "reset requested"
            return
        else:
            return
        self._actions[robot_id] = action

    def actions(self) -> dict[int, RobotAction]:
        return dict(self._actions)

    def selected_action(self) -> RobotAction:
        if self._step_pending:
            return self._actions[self.state.selected_robot_id]
        return self._active_actions[self.state.selected_robot_id]

    def refresh_held_motion(self, pressed_keys: set[str]) -> None:
        if self._held_motion is None:
            return
        robot_id, key = self._held_motion
        if key not in pressed_keys:
            # CoreGraphics can report a single false sample while a passive
            # MuJoCo viewer is syncing.  Require three consecutive misses so
            # a held W/S command is not truncated after one step.
            self._held_key_misses += 1
            if self._held_key_misses < 3:
                return
            self._held_motion = None
            self._held_key_misses = 0
            self._halt_requested = True
            # Cancel a repeat action queued by an earlier in-step poll.  A
            # released key must not run one additional 0.5-second command.
            self._step_pending = False
            current = self._actions[robot_id]
            self._actions[robot_id] = RobotAction(
                MotionAction.STOP, current.lift, current.grip
            )
            return
        self._held_key_misses = 0
        current = self._actions[robot_id]
        self._actions[robot_id] = RobotAction(
            _MOTION_KEYS[key], current.lift, current.grip
        )
        self._step_pending = True

    def take_pending_actions(self) -> dict[int, RobotAction] | None:
        if not self._step_pending and self._latched_turn is not None:
            robot_id, motion = self._latched_turn
            current = self._actions[robot_id]
            self._actions[robot_id] = RobotAction(motion, current.lift, current.grip)
            self._step_pending = True
        if not self._step_pending:
            return None
        actions = dict(self._actions)
        self._active_actions = dict(actions)
        # Each field is an action chosen for this control step, not a UI state.
        # The next step starts from the neutral action unless it receives a new
        # command (or an intentionally held W/S or A/D motion command).
        self._actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        self._step_pending = False
        return actions

    def has_pending_action(self) -> bool:
        """Return whether a new command should preempt the running chunk."""

        return self._step_pending

    def take_halt_request(self) -> bool:
        requested = self._halt_requested
        self._halt_requested = False
        return requested

    def take_root_motion_lock(self) -> bool:
        locked = self._root_motion_lock_pending
        self._root_motion_lock_pending = False
        return locked

    def mark_stopped(self) -> None:
        self._active_actions = dict(self._actions)

    def mark_step_complete(self) -> None:
        """Show neutral actions after the just-dispatched control step ends."""

        if not self._step_pending:
            self._active_actions = {
                robot_id: RobotAction() for robot_id in range(self.robot_count)
            }

    def mark_turn_complete(self, robot_id: int) -> None:
        """Stop reissuing A/D after its five-degree segment has settled."""

        if self._latched_turn is None or self._latched_turn[0] != robot_id:
            return
        self._latched_turn = None
        self._step_pending = False
        current = self._actions[robot_id]
        stopped = RobotAction(MotionAction.STOP, current.lift, current.grip)
        self._actions[robot_id] = stopped
        self._active_actions[robot_id] = stopped
        self.state.message = "STOP"

    def reset(self) -> None:
        self._actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        self._held_motion = None
        self._latched_turn = None
        self._held_key_misses = 0
        self._step_pending = False
        self._halt_requested = False
        self._root_motion_lock_pending = False
        self._active_actions = dict(self._actions)
        self._last_overlay_payload = None
        self.state.reset_requested = False

    @staticmethod
    def help_text() -> str:
        return (
            "Hold W/S curl/flatten | A/D turn 5 degrees | C stop | "
            "R lift front | F lower front | Space attach | X detach | "
            "1-9/0 or B/N select | P pause | Z reset"
        )


def manual_overlay_text(
    controller: ManualController, simulation: Simulation | None = None
) -> tuple[str, str]:
    """Return the controls and LED-style current action status."""

    action = controller.selected_action()
    motion = action.motion.name
    lift = action.lift.name
    grip = action.grip.name
    controls = (
        "BARI MANUAL CONTROLS\n"
        "Hold W/S  curl / flatten\n"
        "A/D turn 5 degrees\n"
        "C stop | R/F lift / lower front\n"
        "Space/X attach / detach\n"
        "B/N previous / next robot\n"
        "P pause | Z reset"
    )
    status = (
        f"ROBOT {controller.state.selected_robot_id + 1}\n"
        "ACTION\n"
        f"  MOTION  ● {motion}\n"
        f"  LIFT    ● {lift}\n"
        f"  GRIP    ● {grip}"
    )
    if simulation is not None:
        observation = simulation.observations()[controller.state.selected_robot_id]
        nearby = ", ".join(
            str(robot_id + 1) for robot_id in observation.nearby_robot_ids
        )
        status += "\nOBSERVATION"
        if observation.is_attaching:
            status += (
                f"\nSTRAIN  {observation.strain_value:.1f} / "
                f"{simulation.robot.maximum_strain_g:.1f} g"
            )
        status += (
            f"\nRANGE   F {observation.distance1:.3f}  D {observation.distance2:.3f}"
            f"\n        L {observation.distance3:.3f}  R {observation.distance4:.3f}"
            f"\nSTATE   curled={int(observation.is_curled)} "
            f"lifted={int(observation.is_front_lifted)}"
            f"\nATTACH  possible={int(observation.is_possible_to_attach)} "
            f"active={int(observation.is_attaching)} "
            f"detached={int(observation.is_detached)}"
            f"\nNEARBY  {nearby or '-'}"
        )
    return controls, status


def manual_selection_overlay_text() -> str:
    """Return the number-key robot-selection legend for the lower-left UI."""

    return "ROBOT SELECT\n1→R1 2→R2 3→R3 4→R4 5→R5\n6→R6 7→R7 8→R8 9→R9 0→R10"


def _sync_robot_labels(viewer, simulation: Simulation) -> None:
    """Draw each robot number as a camera-facing label above its body."""

    user_scene = viewer.user_scn
    attachment_labels = simulation.attachments.active_labels()
    label_count = min(
        simulation.robot_count + len(attachment_labels), user_scene.maxgeom
    )
    user_scene.ngeom = label_count
    for robot_id in range(min(simulation.robot_count, label_count)):
        geom = user_scene.geoms[robot_id]
        body_position = simulation.data.xpos[simulation.root_body_ids[robot_id]]
        position = body_position + (0.0, 0.0, simulation.robot.height_m * 2.5)
        _init_label_geom(geom, position, str(robot_id + 1))
    for index, (position, strain_g) in enumerate(
        attachment_labels, start=simulation.robot_count
    ):
        if index >= label_count:
            break
        _init_label_geom(
            user_scene.geoms[index],
            position + (0.0, 0.0, simulation.robot.height_m),
            f"strain: {strain_g:.1f} g",
        )


def _init_label_geom(geom, position, label: str) -> None:
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LABEL,
        (0.0, 0.0, 0.0),
        position,
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 1.0, 0.35, 1.0),
    )
    geom.label = label


def _sync_manual_overlay(
    viewer, controller: ManualController, simulation: Simulation
) -> None:
    controls, status = manual_overlay_text(controller, simulation)
    selection = manual_selection_overlay_text()
    payload = (controls, selection, status)
    if controller._last_overlay_payload == payload:
        return
    controller._last_overlay_payload = payload
    viewer.set_texts(
        [
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                controls,
                "",
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                selection,
                "",
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                status,
                "",
            ),
        ]
    )


def run_manual_viewer(simulation: Simulation) -> None:
    import mujoco.viewer

    controller = ManualController(simulation.robot_count)
    key_poller = ManualKeyPoller()
    print(ManualController.help_text())
    with mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
        key_callback=controller.key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        shortcut_defaults = ManualShortcutDefaults.capture(viewer)
        viewer.sync()
        _apply_manual_camera(viewer, simulation, reset_orientation=True)
        _sync_robot_labels(viewer, simulation)
        _sync_manual_overlay(viewer, controller, simulation)
        while viewer.is_running():
            controller.process_keys()
            controller.refresh_held_motion(key_poller.pressed_motion_keys())
            shortcut_defaults.restore(viewer)
            if controller.state.reset_requested:
                simulation.reset()
                controller.reset()
                _apply_manual_camera(viewer, simulation, reset_orientation=True)
            if controller.state.paused:
                viewer.sync()
                _sync_manual_overlay(viewer, controller, simulation)
                shortcut_defaults.restore(viewer)
                time.sleep(1.0 / 60.0)
                continue
            actions = controller.take_pending_actions()
            halt_requested = controller.take_halt_request()
            lock_root_motion = (
                (controller.state.selected_robot_id,)
                if controller.take_root_motion_lock()
                else None
            )
            if actions is None:
                if halt_requested:
                    simulation.halt_motion()
                    simulation.end_turn_sessions()
                    controller.mark_stopped()
                viewer.sync()
                _sync_manual_overlay(viewer, controller, simulation)
                shortcut_defaults.restore(viewer)
                time.sleep(1.0 / 60.0)
                continue

            def sync_frame(_simulation: Simulation) -> bool:
                controller.process_keys()
                controller.refresh_held_motion(key_poller.pressed_motion_keys())
                shortcut_defaults.restore(viewer)
                _sync_robot_labels(viewer, simulation)
                viewer.sync()
                # A turn can span policy intervals, but a later input must not
                # wait for it to settle.  End this small render chunk and let
                # the outer loop dispatch the newly selected action.
                return viewer.is_running() and not controller.has_pending_action()

            # Text updates cross into the viewer thread.  Keep them outside
            # the high-frequency physics/render callback so mouse camera
            # interaction remains responsive.
            _sync_manual_overlay(viewer, controller, simulation)
            # Manual mode drives one selected robot at a time.  Stop and hold
            # only the non-selected roots; preserving the selected robot's
            # velocity avoids a visible jerk at every 0.5-second boundary.
            selected_robot_id = controller.state.selected_robot_id
            locked_robot_ids = set(range(simulation.robot_count))
            locked_robot_ids.discard(selected_robot_id)
            if lock_root_motion:
                locked_robot_ids.update(lock_root_motion)
            simulation.halt_robots(locked_robot_ids)
            simulation.step(
                actions,
                frame_callback=sync_frame,
                realtime=True,
                lock_root_motion=tuple(sorted(locked_robot_ids)),
            )
            controller.mark_step_complete()
            if (
                actions[selected_robot_id].motion
                in {MotionAction.TURN_LEFT, MotionAction.TURN_RIGHT}
                and simulation.turn_is_settled(selected_robot_id)
            ):
                simulation.end_turn_sessions()
                controller.mark_turn_complete(selected_robot_id)
            if halt_requested:
                simulation.halt_motion()
                simulation.end_turn_sessions()
                controller.mark_stopped()
            _sync_manual_overlay(viewer, controller, simulation)


def manual_camera_pose(simulation: Simulation) -> ManualCameraPose:
    """Frame every spawned robot tightly while preserving its visible aspect."""

    robot = simulation.robot
    positions = tuple(
        (
            float(simulation.data.xpos[body_id, 0]),
            float(simulation.data.xpos[body_id, 1]),
        )
        for body_id in simulation.root_body_ids
    )
    rear_overhang = robot.rear_length_m / 2.0
    front_overhang = robot.length_m - rear_overhang
    minimum_x = min(position[0] - rear_overhang for position in positions)
    maximum_x = max(position[0] + front_overhang for position in positions)
    minimum_y = min(position[1] - robot.width_m / 2.0 for position in positions)
    maximum_y = max(position[1] + robot.width_m / 2.0 for position in positions)
    span = max(maximum_x - minimum_x, maximum_y - minimum_y)
    return ManualCameraPose(
        lookat=(
            (minimum_x + maximum_x) / 2.0,
            (minimum_y + maximum_y) / 2.0,
            robot.height_m / 2.0,
        ),
        distance=max(0.28, 1.55 * span),
    )


def _apply_manual_camera(
    viewer, simulation: Simulation, *, reset_orientation=False
) -> None:
    camera = manual_camera_pose(simulation)
    viewer.cam.lookat[:] = camera.lookat
    viewer.cam.distance = camera.distance
    if reset_orientation:
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
