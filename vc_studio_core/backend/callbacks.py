from __future__ import annotations


def _noop_status(message: str) -> None:
    pass


def _noop_log(message: str) -> None:
    pass


def _noop_metrics(row: dict, player_stats: dict | None = None) -> None:
    pass
