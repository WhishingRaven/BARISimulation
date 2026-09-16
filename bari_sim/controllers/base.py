"""Minimal policy protocol for future multi-agent wrappers."""

from __future__ import annotations

from typing import Protocol

from ..types import LocalObservation, RobotAction


class DecentralizedController(Protocol):
    def act(self, observation: LocalObservation) -> RobotAction: ...

    def reset(self) -> None: ...
