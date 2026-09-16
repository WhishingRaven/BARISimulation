"""Lightweight CSV state and event logging."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

from .types import AttachmentEvent, LocalObservation

if TYPE_CHECKING:
    from .robot import Robot


class MetricsLogger:
    STATE_FIELDS = (
        "simulation_time_s",
        "robot_id",
        "position_m",
        "orientation_wxyz",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
        "joint_positions_rad",
        "joint_velocities_rad_s",
        "actuator_commands_nm",
        "force_sensor_readings_n",
        "range_distances_m",
        "attachment_active",
        "attachment_force_n",
        "attachment_moment_nm",
        "attachment_utilization",
        "maximum_structural_load_n",
        "crossing_event",
    )
    EVENT_FIELDS = (
        "simulation_time_s",
        "event",
        "source_robot_id",
        "target_robot_id",
        "target_body",
        "force_n",
        "moment_nm",
        "detail",
    )

    def __init__(self, directory: Path, interval_s: float, prefix: str = "run"):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        self.state_path = directory / f"{prefix}_{stamp}.csv"
        self.event_path = directory / f"{prefix}_{stamp}_events.csv"
        self.interval_s = interval_s
        self.environment_name = prefix
        self.next_log_time_s = 0.0
        self.maximum_structural_load_n = 0.0
        self._state_stream = self.state_path.open("w", newline="", encoding="utf-8")
        self._event_stream = self.event_path.open("w", newline="", encoding="utf-8")
        self._state_writer = csv.DictWriter(
            self._state_stream, fieldnames=self.STATE_FIELDS
        )
        self._event_writer = csv.DictWriter(
            self._event_stream, fieldnames=self.EVENT_FIELDS
        )
        self._state_writer.writeheader()
        self._event_writer.writeheader()

    def record(
        self,
        simulation_time_s: float,
        robots: Iterable[Robot],
        observations: Mapping[int, LocalObservation],
        attachment_events: Iterable[AttachmentEvent],
        crossing_robot_ids: set[int],
    ) -> None:
        events = tuple(attachment_events)
        for event in events:
            self.maximum_structural_load_n = max(
                self.maximum_structural_load_n, event.force_n
            )
            self._event_writer.writerow(
                {
                    "simulation_time_s": event.simulation_time_s,
                    "event": event.event,
                    "source_robot_id": event.source_robot_id,
                    "target_robot_id": ""
                    if event.target_robot_id is None
                    else event.target_robot_id,
                    "target_body": event.target_body,
                    "force_n": event.force_n,
                    "moment_nm": event.moment_nm,
                    "detail": event.detail,
                }
            )
        crossing_name = (
            "gap_crossing"
            if self.environment_name == "gap"
            else "obstacle_crossing"
            if self.environment_name == "step"
            else "crossing"
        )
        for robot_id in sorted(crossing_robot_ids):
            self._event_writer.writerow(
                {
                    "simulation_time_s": simulation_time_s,
                    "event": crossing_name,
                    "source_robot_id": robot_id,
                    "target_robot_id": "",
                    "target_body": self.environment_name,
                    "force_n": 0.0,
                    "moment_nm": 0.0,
                    "detail": "privileged evaluation event",
                }
            )
        for observation in observations.values():
            self.maximum_structural_load_n = max(
                self.maximum_structural_load_n,
                observation.attachment.force_n,
            )

        if simulation_time_s + 1e-12 < self.next_log_time_s:
            self._event_stream.flush()
            return
        self.next_log_time_s = simulation_time_s + self.interval_s
        for robot in robots:
            observation = observations[robot.robot_id]
            state = robot.get_global_state()
            self._state_writer.writerow(
                {
                    "simulation_time_s": simulation_time_s,
                    "robot_id": robot.robot_id,
                    "position_m": self._json(state.position_m),
                    "orientation_wxyz": self._json(state.orientation_wxyz),
                    "linear_velocity_m_s": self._json(state.linear_velocity_m_s),
                    "angular_velocity_rad_s": self._json(state.angular_velocity_rad_s),
                    "joint_positions_rad": self._json(observation.joint_positions_rad),
                    "joint_velocities_rad_s": self._json(
                        observation.joint_velocities_rad_s
                    ),
                    "actuator_commands_nm": self._json(robot.actuator_commands_nm),
                    "force_sensor_readings_n": self._json(
                        {
                            name: value.force_n
                            for name, value in observation.force_sensors.items()
                        }
                    ),
                    "range_distances_m": self._json(
                        {
                            name: value.distance_m
                            for name, value in observation.ranges.items()
                        }
                    ),
                    "attachment_active": int(observation.attachment.active),
                    "attachment_force_n": observation.attachment.force_n,
                    "attachment_moment_nm": observation.attachment.moment_nm,
                    "attachment_utilization": observation.attachment.utilization,
                    "maximum_structural_load_n": self.maximum_structural_load_n,
                    "crossing_event": int(robot.robot_id in crossing_robot_ids),
                }
            )
        self._state_stream.flush()
        self._event_stream.flush()

    def reset_clock(self) -> None:
        self.next_log_time_s = 0.0

    def close(self) -> None:
        if not self._state_stream.closed:
            self._state_stream.close()
        if not self._event_stream.closed:
            self._event_stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _json(value) -> str:
        return json.dumps(value, separators=(",", ":"))
