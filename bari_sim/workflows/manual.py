"""Latched keyboard teleoperation for the declared discrete action space."""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import SimpleQueue

from ..robot import GripAction, LiftAction, MotionAction, RobotAction
from ..simulation import Simulation


@dataclass
class ManualState:
    selected_robot_id: int = 0
    paused: bool = False
    reset_requested: bool = False
    message: str = "ready"


class ManualController:
    def __init__(self, robot_count: int):
        self.robot_count = robot_count
        self.state = ManualState()
        self._actions = {robot_id: RobotAction() for robot_id in range(robot_count)}
        self._keys: SimpleQueue[str] = SimpleQueue()

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
        motion_keys = {
            "W": MotionAction.CURL_BODY,
            "S": MotionAction.FLATTEN_BODY,
            "A": MotionAction.TURN_LEFT,
            "D": MotionAction.TURN_RIGHT,
            "C": MotionAction.STOP,
        }
        if key in motion_keys:
            action = RobotAction(motion_keys[key], current.lift, current.grip)
            self.state.message = action.motion.name
        elif key == "R":
            action = RobotAction(current.motion, LiftAction.LIFT_FRONT, current.grip)
            self.state.message = "LIFT_FRONT"
        elif key == "F":
            action = RobotAction(current.motion, LiftAction.UNLIFT_FRONT, current.grip)
            self.state.message = "UNLIFT_FRONT"
        elif raw_key == " ":
            action = RobotAction(current.motion, current.lift, GripAction.ATTACH)
            self.state.message = "ATTACH"
        elif key == "X":
            action = RobotAction(current.motion, current.lift, GripAction.DETACH)
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

    def reset(self) -> None:
        self._actions = {
            robot_id: RobotAction() for robot_id in range(self.robot_count)
        }
        self.state.reset_requested = False

    @staticmethod
    def help_text() -> str:
        return (
            "W curl | S flatten | A/D turn whole body | C stop | "
            "R lift front | F lower front | Space attach | X detach | "
            "1-9/0 or B/N select | P pause | Z reset"
        )


def run_manual_viewer(simulation: Simulation) -> None:
    import mujoco.viewer

    controller = ManualController(simulation.robot_count)
    print(ManualController.help_text())
    with mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
        key_callback=controller.key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -28
        viewer.cam.distance = max(1.2, 0.35 * simulation.request.grid.rows + 1.0)
        while viewer.is_running():
            controller.process_keys()
            if controller.state.reset_requested:
                simulation.reset()
                controller.reset()
            if controller.state.paused:
                viewer.sync()
                time.sleep(1.0 / 60.0)
                continue

            def sync_frame(_simulation: Simulation) -> bool:
                controller.process_keys()
                viewer.sync()
                return viewer.is_running()

            simulation.step(
                controller.actions(),
                frame_callback=sync_frame,
                realtime=True,
            )
