"""Swarm simulation orchestrator and future environment-wrapper boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Self

import mujoco
import numpy as np

from .attachment import AttachmentManager
from .config import PROJECT_ROOT, ExperimentConfig, load_experiment_config
from .environment import EnvironmentEvaluator
from .friction import DirectionalFrictionModel
from .metrics import MetricsLogger
from .model_builder import (
    ANCHOR_PAD_SIDES,
    BuiltModel,
    ModelBuilder,
    anchor_pad_geom_name,
    collision_geom_name,
    gripper_geom_name,
    link_body_name,
)
from .robot import Robot
from .sensors import SensorSuite
from .types import GlobalRobotState, LocalObservation, RobotAction, StepResult


class SwarmSimulator:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        logging_enabled: bool = True,
        log_directory: Path | None = None,
    ):
        self.config = config
        self.built_model: BuiltModel = ModelBuilder(config).build()
        self.model = mujoco.MjModel.from_xml_string(self.built_model.xml)
        self.data = mujoco.MjData(self.model)
        self._seed = config.simulation.random_seed
        self.geom_to_robot: dict[int, int] = {}
        self.geom_to_link: dict[int, int] = {}
        self.body_to_robot: dict[int, int] = {}
        self.body_to_link: dict[int, int] = {}
        self._index_robot_objects()
        self.attachments = AttachmentManager(
            self.model,
            self.data,
            config.robot,
            config.simulation.robot_count,
            self.geom_to_robot,
            self.body_to_robot,
            self.body_to_link,
        )
        self.sensors = tuple(
            SensorSuite(
                self.model,
                self.data,
                robot_id,
                config.robot,
                self.geom_to_robot,
                self.geom_to_link,
            )
            for robot_id in range(config.simulation.robot_count)
        )
        self.robots = tuple(
            Robot(
                self.model,
                self.data,
                robot_id,
                config.robot,
                self.sensors[robot_id],
                self.attachments,
            )
            for robot_id in range(config.simulation.robot_count)
        )
        self.directional_friction = DirectionalFrictionModel(
            self.model,
            self.data,
            config.robot.directional_friction,
            self.body_to_robot,
            self.geom_to_link,
        )
        self.environment = EnvironmentEvaluator(config.environment, config.robot)
        self.logger = (
            MetricsLogger(
                log_directory or PROJECT_ROOT / "logs",
                config.simulation.log_interval_s,
                prefix=config.environment.name,
            )
            if logging_enabled
            else None
        )
        self.reset()

    @classmethod
    def from_default_config(
        cls,
        *,
        robot_count: int | None = None,
        environment: str | None = None,
        random_seed: int | None = None,
        logging_enabled: bool = True,
        log_directory: Path | None = None,
    ) -> SwarmSimulator:
        config = load_experiment_config().with_overrides(
            robot_count=robot_count,
            environment=environment,
            random_seed=random_seed,
        )
        return cls(config, logging_enabled=logging_enabled, log_directory=log_directory)

    @property
    def robot_count(self) -> int:
        return len(self.robots)

    @property
    def timestep_s(self) -> float:
        return float(self.model.opt.timestep)

    def robot(self, robot_id: int) -> Robot:
        if robot_id < 0 or robot_id >= self.robot_count:
            raise IndexError(f"robot_id must be in [0, {self.robot_count - 1}]")
        return self.robots[robot_id]

    def step(
        self,
        actions: Mapping[int, RobotAction] | None = None,
        *,
        external_forces_n: Mapping[int, tuple[float, float, float]] | None = None,
    ) -> StepResult:
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        if actions:
            for robot_id in sorted(actions):
                self.robot(robot_id).apply_action(actions[robot_id])
        for robot in self.robots:
            robot.apply_control(self.timestep_s)
        if external_forces_n:
            for robot_id, force in external_forces_n.items():
                self.data.xfrc_applied[self.robot(robot_id).root_body_id, :3] += (
                    np.asarray(
                        force,
                        dtype=np.float64,
                    )
                )
        self.directional_friction.apply()
        mujoco.mj_step(self.model, self.data)
        self.attachments.post_step()
        observations = self.get_observations()
        events = self.attachments.drain_events()
        crossed = self.environment.update(self.global_state())
        if self.logger is not None:
            self.logger.record(
                float(self.data.time), self.robots, observations, events, crossed
            )
        return StepResult(observations=observations, attachment_events=events)

    def get_observations(self) -> Mapping[int, LocalObservation]:
        return {robot.robot_id: robot.get_observation() for robot in self.robots}

    def global_state(self) -> tuple[GlobalRobotState, ...]:
        return tuple(robot.get_global_state() for robot in self.robots)

    def reset(self) -> Mapping[int, LocalObservation]:
        self.attachments.reset()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.environment.reset()
        for sensor in self.sensors:
            sensor.seed(self._seed)
            sensor.reset()
        for robot in self.robots:
            robot.reset_control()
        if self.logger is not None:
            self.logger.reset_clock()
        return self.get_observations()

    def set_anchor_direction(self, robot_id: int, direction: float) -> None:
        """Orient the selected robot's asymmetric ground pads for its gait."""

        self.directional_friction.set_anchor_direction(robot_id, direction)

    def set_stance_links(self, robot_id: int, link_ids: tuple[int, ...]) -> None:
        """Choose the links whose outboard pads may grip during this phase."""

        self.directional_friction.set_stance_links(robot_id, link_ids)

    def run_headless(
        self,
        duration_s: float,
        controllers: Mapping[int, object] | None = None,
    ) -> Mapping[int, LocalObservation]:
        end_time = float(self.data.time) + duration_s
        observations = self.get_observations()
        while float(self.data.time) + 1e-12 < end_time:
            actions: dict[int, RobotAction] = {}
            if controllers:
                for robot_id, controller in controllers.items():
                    direction = getattr(controller, "direction", None)
                    if direction is not None:
                        self.set_anchor_direction(robot_id, float(direction))
                    stance_links = getattr(controller, "stance_links", None)
                    if callable(stance_links):
                        self.set_stance_links(
                            robot_id,
                            stance_links(
                                observations[robot_id].simulation_time_s,
                                self.config.robot.link_count,
                            ),
                        )
                    actions[robot_id] = controller.act(observations[robot_id])
            observations = self.step(actions).observations
        return observations

    def export_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.built_model.xml, encoding="utf-8")

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _index_robot_objects(self) -> None:
        for robot_id in range(self.config.simulation.robot_count):
            for link_id in range(self.config.robot.link_count):
                body_id = self.model.body(link_body_name(robot_id, link_id)).id
                geom_id = self.model.geom(collision_geom_name(robot_id, link_id)).id
                self.body_to_robot[body_id] = robot_id
                self.body_to_link[body_id] = link_id
                self.geom_to_robot[geom_id] = robot_id
                self.geom_to_link[geom_id] = link_id
                if self.config.robot.directional_friction.enabled:
                    for side in ANCHOR_PAD_SIDES:
                        pad_id = self.model.geom(
                            anchor_pad_geom_name(robot_id, link_id, side)
                        ).id
                        self.geom_to_robot[pad_id] = robot_id
                        self.geom_to_link[pad_id] = link_id
            gripper_id = self.model.geom(gripper_geom_name(robot_id)).id
            front_body_id = self.model.geom_bodyid[gripper_id]
            self.geom_to_robot[gripper_id] = robot_id
            self.geom_to_link[gripper_id] = self.body_to_link[int(front_body_id)]
