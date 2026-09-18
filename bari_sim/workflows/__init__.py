"""User-facing manual, training, inference, and evaluation workflows."""

from .evaluation import EvaluationSummary, evaluate_policy
from .inference import run_inference
from .manual import ManualController, run_manual_viewer

__all__ = [
    "EvaluationSummary",
    "ManualController",
    "TrainingSettings",
    "TrainingSummary",
    "evaluate_policy",
    "run_inference",
    "run_manual_viewer",
    "train_policy",
]


def __getattr__(name: str):
    if name in {"TrainingSettings", "TrainingSummary", "train_policy"}:
        from ..train import TrainingSettings, TrainingSummary, train_policy

        return {
            "TrainingSettings": TrainingSettings,
            "TrainingSummary": TrainingSummary,
            "train_policy": train_policy,
        }[name]
    raise AttributeError(name)
