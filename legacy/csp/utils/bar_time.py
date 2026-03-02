"""Utilities for aligning realtime executions to 15-minute bars."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple


def _ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime for ``dt``."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def align_15m(now_utc: datetime | None = None, delay_sec: int = 5) -> Tuple[datetime, datetime]:
    """Align *now* to the nearest 15-minute bar open.

    Parameters
    ----------
    now_utc:
        Current timestamp in UTC. When ``None`` the current UTC time will be used.
    delay_sec:
        Extra seconds to subtract before alignment to guarantee the new bar is ready.

    Returns
    -------
    (prev_open_ts, bar_open_ts):
        ``prev_open_ts`` is the open time of the last fully closed bar, while
        ``bar_open_ts`` is the current bar open (target dispatch window).
    """

    now_utc = _ensure_utc(now_utc or datetime.now(timezone.utc))
    effective = now_utc - timedelta(seconds=delay_sec)
    minute = (effective.minute // 15) * 15
    bar_open_ts = effective.replace(minute=minute, second=0, microsecond=0)
    prev_open_ts = bar_open_ts - timedelta(minutes=15)
    return prev_open_ts, bar_open_ts


__all__ = ["align_15m"]
