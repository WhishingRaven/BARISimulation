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
        self._step_pending = False
        self._halt_requested = False
        self._root_motion_lock_pending = False
        self._active_actions = dict(self._actions)

    def key_callback(self, keycode: int) -> None:
        try:
            self._keys.put(chr(keycode))
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
                self.state.message = f"selected robot {selected}"
            return
        if key in {"B", "N"}:
            offset = -1 if key == "B" else 1
            self.state.selected_robot_id = (
                self.state.selected_robot_id + offset
            ) % self.robot_count
            self.state.message = f"selected robot {self.state.selected_robot_id}"
            return
        robot_id = self.state.selected_robot_id
        current = self._actions[robot_id]
        if key in _MOTION_KEYS:
            action = RobotAction(_MOTION_KEYS[key], current.lift, current.grip)
            if self._held_motion != (robot_id, key):
                self._held_motion = (robot_id, key)
                self._step_pending = True
            self.state.message = action.motion.name
        elif key == "C":
            action = RobotAction(MotionAction.STOP, current.lift, current.grip)
            self._held_motion = None
            self._halt_requested = True
            self.state.message = "STOP"
        elif key == "R":
            action = RobotAction(MotionAction.STOP, LiftAction.LIFT_FRONT, current.grip)
            self._root_motion_lock_pending = True
            self._step_pending = True
            self.state.message = "LIFT_FRONT"
        elif key == "F":
            action = RobotAction(
                MotionAction.STOP, LiftAction.UNLIFT_FRONT, current.grip
            )
            self._root_motion_lock_pending = True
            self._step_pending = True
            self.state.message = "UNLIFT_FRONT"
        elif raw_key == " ":
            action = RobotAction(MotionAction.STOP, current.lift, GripAction.ATTACH)
            self._root_motion_lock_pending = True
            self._step_pending = True
            self.state.message = "ATTACH"
        elif key == "X":
            action = RobotAction(MotionAction.STOP, current.lift, GripAction.DETACH)
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
            self._held_motion = None
            self._halt_requested = True
            if not self._step_pending:
                current = self._actions[robot_id]
                self._actions[robot_id] = RobotAction(
                    MotionAction.STOP, current.lift, current.grip
                )
            return
        current = self._actions[robot_id]
        self._actions[robot_id] = RobotAction(
            _MOTION_KEYS[key], current.lift, current.grip
        )
        self._step_pending = True

    def take_pending_actions(self) -> dict[int, RobotAction] | None:
        if not self._step_pending:
            return None
        actions = dict(self._actions)
        self._active_actions = dict(actions)
        self._actions = {
            robot_id: RobotAction(MotionAction.STOP, action.lift, action.grip)
            for robot_id, action in self._actions.items()
        }
        self._step_pending = False
        return actions

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

    def reset(self) -> None:
        self._actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        self._held_motion = None
        self._step_pending = False
        self._halt_requested = False
        self._root_motion_lock_pending = False
        self._active_actions = dict(self._actions)
        self.state.reset_requested = False

    @staticmethod
    def help_text() -> str:
        return (
            "W curl | S flatten | A/D turn whole body | C stop | "
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
        "Hold A/D  turn left / right\n"
        "C stop | R/F lift / lower front\n"
        "Space/X attach / detach\n"
        "1-9/0 select | B/N previous / next\n"
        "P pause | Z reset"
    )
    status = (
        f"ROBOT {controller.state.selected_robot_id + 1}\n"
        f"MOTION  ● {motion}\n"
        f"LIFT    ● {lift}\n"
        f"GRIP    ● {grip}"
    )
    if simulation is not None:
        observation = simulation.observations()[controller.state.selected_robot_id]
        nearby = ", ".join(
            str(robot_id + 1) for robot_id in observation.nearby_robot_ids
        )
        status += (
            f"\nSTRAIN  {observation.strain_value:.1f} / "
            f"{simulation.robot.maximum_strain_g:.1f} g"
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


def _sync_manual_overlay(
    viewer, controller: ManualController, simulation: Simulation
) -> None:
    camera_lookat = tuple(float(value) for value in viewer.cam.lookat)
    camera_distance = float(viewer.cam.distance)
    camera_azimuth = float(viewer.cam.azimuth)
    camera_elevation = float(viewer.cam.elevation)
    controls, status = manual_overlay_text(controller, simulation)
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
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                status,
                "",
            ),
        ]
    )
    # set_texts crosses into the viewer thread; restore the user's camera
    # state in case that synchronization copied native viewer defaults.
    viewer.cam.lookat[:] = camera_lookat
    viewer.cam.distance = camera_distance
    viewer.cam.azimuth = camera_azimuth
    viewer.cam.elevation = camera_elevation


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
                    controller.mark_stopped()
                viewer.sync()
                _sync_manual_overlay(viewer, controller, simulation)
                shortcut_defaults.restore(viewer)
                time.sleep(1.0 / 60.0)
                continue

            def sync_frame(_simulation: Simulation) -> bool:
                controller.process_keys()
                shortcut_defaults.restore(viewer)
                viewer.sync()
                _sync_manual_overlay(viewer, controller, simulation)
                shortcut_defaults.restore(viewer)
                return viewer.is_running()

            simulation.step(
                actions,
                frame_callback=sync_frame,
                realtime=True,
                lock_root_motion=lock_root_motion,
            )
            if halt_requested:
                simulation.halt_motion()
                controller.mark_stopped()


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
