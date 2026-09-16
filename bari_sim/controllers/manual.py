"""macOS/mjpython-safe keyboard teleoperation via viewer key callbacks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import radians
from queue import SimpleQueue
from typing import TYPE_CHECKING

from ..types import LocalObservation, RobotAction
from .gait import TravelingWaveGait
from .inchworm import InchwormController

if TYPE_CHECKING:
    from ..simulator import SwarmSimulator


@dataclass
class ManualState:
    selected_robot_id: int = 0
    paused: bool = False
    realtime_scale: float = 1.0
    debug_visible: bool = False
    message: str = "manual position control"


class ManualTeleop:
    SPEEDS = (0.25, 0.50, 1.0, 2.0, 4.0)

    def __init__(self, robot_count: int, joint_count: int, joint_step_deg: float = 5.0):
        self.robot_count = robot_count
        self.joint_count = joint_count
        self.joint_step_rad = radians(joint_step_deg)
        self.state = ManualState()
        self._keys: SimpleQueue[str] = SimpleQueue()
        self._targets = {
            robot_id: [0.0] * joint_count for robot_id in range(robot_count)
        }
        self._modes = {robot_id: "manual" for robot_id in range(robot_count)}
        self._directions = {robot_id: 1.0 for robot_id in range(robot_count)}
        self._gaits = {robot_id: TravelingWaveGait() for robot_id in range(robot_count)}
        self._inchworm = {
            robot_id: InchwormController() for robot_id in range(robot_count)
        }
        self._pending_attach_robot_id: int | None = None
        self._pending_detach_robot_id: int | None = None
        self._pending_reset = False

    def key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode)
        except (ValueError, OverflowError):
            return
        self._keys.put(key)

    def update(
        self,
        simulator: SwarmSimulator,
        observations: Mapping[int, LocalObservation],
    ) -> Mapping[int, RobotAction]:
        while not self._keys.empty():
            self._handle_key(self._keys.get(), simulator)
        if self._pending_reset:
            observations = simulator.reset()
            self._targets = {
                robot_id: list(observations[robot_id].joint_positions_rad)
                for robot_id in range(self.robot_count)
            }
            self._modes = {robot_id: "manual" for robot_id in range(self.robot_count)}
            for controller in self._inchworm.values():
                controller.reset()
            for gait in self._gaits.values():
                gait.reset()
            self._pending_attach_robot_id = None
            self._pending_detach_robot_id = None
            self._pending_reset = False
            self.state.message = "simulation reset"

        actions: dict[int, RobotAction] = {}
        for robot_id in range(self.robot_count):
            observation = observations[robot_id]
            mode = self._modes[robot_id]
            if mode == "wave":
                gait = self._gaits[robot_id]
                gait.direction = self._directions[robot_id]
                simulator.set_anchor_direction(robot_id, gait.direction)
                simulator.set_stance_links(
                    robot_id,
                    gait.stance_links(
                        observation.simulation_time_s, self.joint_count + 1
                    ),
                )
                base_action = gait.act(observation)
            elif mode == "inchworm":
                base_action = self._inchworm[robot_id].act(observation)
            else:
                base_action = RobotAction(
                    joint_targets_rad=tuple(self._targets[robot_id])
                )

            detach = base_action.detach or robot_id == self._pending_detach_robot_id
            attach = (
                base_action.attach or robot_id == self._pending_attach_robot_id
            ) and not detach
            actions[robot_id] = RobotAction(
                joint_targets_rad=base_action.joint_targets_rad,
                joint_torques_nm=base_action.joint_torques_nm,
                attach=attach,
                detach=detach,
            )
        self._pending_attach_robot_id = None
        self._pending_detach_robot_id = None
        return actions

    def status_text(self, observation: LocalObservation) -> tuple[str, str]:
        mode = self._modes[self.state.selected_robot_id]
        attachment = observation.attachment
        left = "Selected robot\nMode\nPaused\nRealtime\nDebug rays\nAttachment\nLoad\nStatus"
        right = (
            f"R{self.state.selected_robot_id + 1} (ID {self.state.selected_robot_id})\n"
            f"{mode}\n{self.state.paused}\n{self.state.realtime_scale:.2f}x\n"
            f"{'shown' if self.state.debug_visible else 'hidden'}\n"
            f"{'active' if attachment.active else 'open'}\n"
            f"{attachment.force_n:.3f} N ({attachment.utilization:.0%})\n"
            f"{self.state.message}"
        )
        return left, right

    @staticmethod
    def help_text() -> tuple[str, str]:
        return (
            "Keys\n1-9/0, B/N\nW / S / C\nQ / A\nE / D\nI\nSpace / X\nP / R\n- / +\nV",
            "Action\nselect, previous/next robot\nforward / reverse / stop\njoint 1 + / -\njoint 2 + / -\ninchworm toggle\nattach toggle / force detach\npause / reset\nslower / faster\ndebug rays",
        )

    def _handle_key(self, raw_key: str, simulator: SwarmSimulator) -> None:
        key = raw_key.upper()
        if raw_key.isdigit():
            selected = 9 if raw_key == "0" else int(raw_key) - 1
            if selected < self.robot_count:
                self.state.selected_robot_id = selected
                self.state.message = f"selected robot {selected + 1}"
            return
        if key in {"B", "N"}:
            offset = -1 if key == "B" else 1
            self.state.selected_robot_id = (
                self.state.selected_robot_id + offset
            ) % self.robot_count
            self.state.message = f"selected robot {self.state.selected_robot_id + 1}"
            return
        robot_id = self.state.selected_robot_id
        if key in {"Q", "A", "E", "D"}:
            joint_id = 0 if key in {"Q", "A"} else 1
            if joint_id < self.joint_count:
                direction = 1.0 if key in {"Q", "E"} else -1.0
                low, high = simulator.config.robot.joint.limits_rad[joint_id]
                target = (
                    self._targets[robot_id][joint_id] + direction * self.joint_step_rad
                )
                self._targets[robot_id][joint_id] = min(max(target, low), high)
                self._modes[robot_id] = "manual"
                self.state.message = f"joint {joint_id + 1}: {self._targets[robot_id][joint_id]:+.2f} rad"
        elif key == "W":
            if self._modes[robot_id] != "wave" or self._directions[robot_id] != 1.0:
                self._gaits[robot_id].reset()
            self._modes[robot_id] = "wave"
            self._directions[robot_id] = 1.0
            self.state.message = "traveling wave forward"
        elif key == "S":
            if self._modes[robot_id] != "wave" or self._directions[robot_id] != -1.0:
                self._gaits[robot_id].reset()
            self._modes[robot_id] = "wave"
            self._directions[robot_id] = -1.0
            self.state.message = "traveling wave reverse"
        elif key == "C":
            self._modes[robot_id] = "manual"
            self._targets[robot_id] = list(
                simulator.robot(robot_id).joint_positions_rad()
            )
            self.state.message = "gait stopped"
        elif key == "I":
            self._modes[robot_id] = (
                "manual" if self._modes[robot_id] == "inchworm" else "inchworm"
            )
            self._inchworm[robot_id].reset()
            self.state.message = self._modes[robot_id]
        elif raw_key == " ":
            if simulator.robot(robot_id).get_attachment_state().active:
                self._pending_detach_robot_id = robot_id
                self.state.message = "detach requested"
            else:
                self._pending_attach_robot_id = robot_id
                self.state.message = "attach requested"
        elif key == "X":
            self._pending_detach_robot_id = robot_id
            self.state.message = "detach requested"
        elif key == "P":
            self.state.paused = not self.state.paused
            self.state.message = "paused" if self.state.paused else "running"
        elif key == "R":
            self._pending_reset = True
        elif raw_key in {"[", "-", "_"}:
            self._change_speed(-1)
        elif raw_key in {"]", "=", "+"}:
            self._change_speed(1)
        elif key == "V":
            self.state.debug_visible = not self.state.debug_visible
            state = "shown" if self.state.debug_visible else "hidden"
            self.state.message = f"debug rays {state}"

    def _change_speed(self, direction: int) -> None:
        closest = min(
            range(len(self.SPEEDS)),
            key=lambda index: abs(self.SPEEDS[index] - self.state.realtime_scale),
        )
        index = min(max(closest + direction, 0), len(self.SPEEDS) - 1)
        self.state.realtime_scale = self.SPEEDS[index]
        self.state.message = f"realtime {self.state.realtime_scale:.2f}x"
