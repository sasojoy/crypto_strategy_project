"""Training helpers for orchestrating multi-symbol workflows."""

from .train import train_for_symbol  # noqa: F401

try:
    from .optimize import optimize_then_train_symbol  # noqa: F401
except Exception:  # pragma: no cover - optional dependency path
    optimize_then_train_symbol = None  # type: ignore

__all__ = ["train_for_symbol", "optimize_then_train_symbol"]
