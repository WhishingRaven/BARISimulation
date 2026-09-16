"""Passive-viewer overlays and custom debug geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from .model_builder import gripper_site_name, link_body_name

if TYPE_CHECKING:
    from .controllers.manual import ManualTeleop
    from .simulator import SwarmSimulator


class DebugVisualizer:
    def __init__(self) -> None:
        stable = mujoco.MjvOption()
        stable.geomgroup[:] = 0
        stable.geomgroup[:3] = 1
        stable.sitegroup[:] = 0
        stable.sitegroup[2] = 1
        for flag in (
            mujoco.mjtVisFlag.mjVIS_TRANSPARENT,
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
            mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
        ):
            stable.flags[flag] = 0
        self._stable_visual_flags = np.asarray(stable.flags).copy()
        self._stable_geom_groups = np.asarray(stable.geomgroup).copy()
        self._stable_site_groups = np.asarray(stable.sitegroup).copy()

    def update(
        self,
        viewer,
        simulator: SwarmSimulator,
        selected_robot_id: int,
        show_sensor_rays: bool = False,
    ) -> None:
        scene = viewer.user_scn
        if scene is None:
            return
        with viewer.lock():
            self._stabilize_view_options(viewer, scene)
            scene.ngeom = 0
            for robot in simulator.robots:
                body_id = simulator.model.body(
                    link_body_name(robot.robot_id, min(1, robot.config.link_count - 1))
                ).id
                position = np.asarray(
                    simulator.data.xpos[body_id], dtype=np.float64
                ).copy()
                position[2] += 0.09
                label = f"R{robot.robot_id + 1}"
                if robot.robot_id == selected_robot_id:
                    label += " SELECTED"
                self._add_label(
                    scene, position, label, robot.robot_id == selected_robot_id
                )
                if show_sensor_rays:
                    for origin, endpoint, detected in robot.sensors.last_rays.values():
                        color = (
                            (1.0, 0.25, 0.15, 0.9)
                            if detected
                            else (0.20, 0.90, 0.35, 0.55)
                        )
                        self._add_connector(scene, origin, endpoint, color, width=2.0)

            for attachment in simulator.attachments.active_attachments():
                source_site_id = simulator.model.site(
                    gripper_site_name(attachment.source_robot_id)
                ).id
                source = np.asarray(
                    simulator.data.site_xpos[source_site_id], dtype=np.float64
                )
                target = simulator.attachments.target_world_point(attachment)
                self._add_connector(
                    scene, source, target, (1.0, 0.12, 0.08, 1.0), width=5.0
                )
                self._add_sphere(scene, target, 0.009, (1.0, 0.12, 0.08, 1.0))

    def _stabilize_view_options(self, viewer, scene) -> None:
        if not hasattr(viewer, "opt"):
            return
        # Manual-control keys overlap MuJoCo viewer shortcuts: number keys toggle
        # geom groups, W toggles wireframe, S toggles shadows, and several other
        # letters toggle visualization modes. Reassert one stable presentation
        # before every sync so controls affect physics only.
        viewer.opt.flags[:] = self._stable_visual_flags
        viewer.opt.geomgroup[:] = self._stable_geom_groups
        viewer.opt.sitegroup[:] = self._stable_site_groups
        scene.flags[:] = 0
        for flag in (
            mujoco.mjtRndFlag.mjRND_SHADOW,
            mujoco.mjtRndFlag.mjRND_REFLECTION,
            mujoco.mjtRndFlag.mjRND_SKYBOX,
            mujoco.mjtRndFlag.mjRND_HAZE,
            mujoco.mjtRndFlag.mjRND_CULL_FACE,
        ):
            scene.flags[flag] = 1

    def set_text(
        self, viewer, teleop: ManualTeleop, observation, simulator: SwarmSimulator
    ) -> None:
        if not hasattr(viewer, "set_texts"):
            return
        status_left, status_right = teleop.status_text(observation)
        help_left, help_right = teleop.help_text()
        texts = [
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                status_left,
                status_right,
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                help_left,
                help_right,
            ),
        ]
        environment = simulator.config.environment
        if environment.name == "gap":
            texts.append(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    "Gap width\nPlatform height",
                    f"{environment.gap_width_m:.3f} m\n{environment.platform_height_m:.3f} m",
                )
            )
        elif environment.name == "step":
            texts.append(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    "Step height\nStep width",
                    f"{environment.step_height_m:.3f} m\n{environment.step_width_m:.3f} m",
                )
            )
        viewer.set_texts(texts)

    @staticmethod
    def _slot(scene):
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return geom

    def _add_label(
        self, scene, position: np.ndarray, label: str, selected: bool
    ) -> None:
        geom = self._slot(scene)
        if geom is None:
            return
        rgba = np.asarray(
            (1.0, 0.95, 0.15, 1.0) if selected else (0.9, 0.9, 0.9, 1.0),
            dtype=np.float32,
        )
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LABEL,
            np.zeros(3),
            position,
            np.eye(3).reshape(-1),
            rgba,
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = label

    def _add_connector(
        self,
        scene,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[float, float, float, float],
        width: float,
    ) -> None:
        geom = self._slot(scene)
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)

    def _add_sphere(
        self,
        scene,
        position: np.ndarray,
        radius: float,
        color: tuple[float, float, float, float],
    ) -> None:
        geom = self._slot(scene)
        if geom is None:
            return
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray((radius, radius, radius), dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.eye(3).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
