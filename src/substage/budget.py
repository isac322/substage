"""Prospective experiment-wide USD reservations, independent of usage logs.

SQLite's short BEGIN IMMEDIATE transactions serialize processes and threads.
No transaction survives reserve()/settle(); providers and backoff never hold a
ledger lock. An unresolved attempt remains reserved indefinitely: absence of
usage, a timeout, or a crashed process is not evidence of zero provider spend.
The application creates this ledger only on a future configured budgeted call.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

from .config import APP, Settings, _xdg

_SCALE = 1_000_000_000


class BudgetError(RuntimeError):
    """A budgeted request cannot safely be dispatched or accounted for."""


def group_for(model: str, provider: str | None = None) -> str | None:
    # Endpoint, not the model's name, determines the billing family.
    if provider in {"codex", "openai-codex"}:
        return None
    wire = model.split("/", 1)[-1]
    if provider == "anthropic":
        return "opus" if wire == "claude-opus-5-5" else None
    if provider == "gemini":
        return "gemini"
    if provider is not None:
        raise BudgetError(f"no validated budget classification for provider {provider!r}")
    if model.startswith("codex/"):
        return None
    if model.startswith("anthropic/"):
        return "opus" if wire == "claude-opus-5-5" else None
    return "gemini" if "/" not in model else None


def _units(value: float, *, limit: bool = False) -> int:
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise BudgetError("USD amounts must be finite and nonnegative")
    return int((amount * _SCALE).to_integral_value(rounding=ROUND_FLOOR if limit else ROUND_CEILING))


def ledger_path(settings: Settings) -> Path:
    if settings.experiment_id is None:
        raise BudgetError("experiment_id is required before a new Gemini/Opus request")
    # Settings validates the ID before it can be used as a path component.
    return _xdg("XDG_STATE_HOME", ".local/state") / APP / "experiments" / settings.experiment_id / "budget.sqlite3"


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Commit the complete schema independently of request authorization.

    A rejected first request must not roll back the ledger's tables. The two
    tables are created in one transaction so readers never see half a schema.
    """
    connection.execute("BEGIN IMMEDIATE")
    try:
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
        if tables and not {"limits", "reservations"} <= tables:
            raise BudgetError("budget ledger schema is incomplete; cannot prove accounting")
        connection.execute("CREATE TABLE IF NOT EXISTS limits (budget_group TEXT PRIMARY KEY, units INTEGER NOT NULL)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, budget_group TEXT NOT NULL, "
            "model TEXT NOT NULL, stage TEXT NOT NULL, upper_units INTEGER NOT NULL, "
            "cost_units INTEGER, status TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


@dataclass(frozen=True)
class Reservation:
    experiment_id: str
    reservation_id: str
    group: str
    upper_usd: float
    _path: Path

    def settle(self, cost_usd: float) -> None:
        """Replace the reservation only with complete, evidenced usage cost.

        Idempotent for the same cost. An upper-bound breach is recorded and
        poisons that budget group rather than silently permitting more calls.
        """
        cost = _units(cost_usd)
        exceeded = False
        with closing(_connect(self._path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT upper_units, cost_units, status FROM reservations WHERE id = ?",
                    (self.reservation_id,),
                ).fetchone()
                if row is None:
                    raise BudgetError("reservation is missing; cannot prove budget accounting")
                if row["status"] != "reserved":
                    if row["cost_units"] != cost:
                        raise BudgetError("reservation already settled with a different spend")
                    exceeded = row["status"] == "over_bound"
                else:
                    exceeded = cost > row["upper_units"]
                    connection.execute(
                        "UPDATE reservations SET cost_units = ?, status = ? WHERE id = ?",
                        (cost, "over_bound" if exceeded else "settled", self.reservation_id),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        if exceeded:
            raise BudgetError("recorded provider spend exceeded its validated reservation; new calls blocked")


def reserve(
    settings: Settings, model: str, stage: str, upper_usd: float, *, provider: str | None = None,
) -> Reservation | None:
    """Atomically authorize one bounded network attempt before dispatch."""
    group = group_for(model, provider)
    if group is None:
        return None  # OpenAI has neither a budget requirement nor a local context cutoff.
    upper = _units(upper_usd)
    if upper <= 0:
        raise BudgetError("a validated positive request upper bound is required")
    path = ledger_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    reservation_id = uuid.uuid4().hex
    limits = {"gemini": _units(settings.gemini_budget_usd, limit=True), "opus": _units(settings.opus_budget_usd, limit=True)}
    with closing(_connect(path)) as connection:
        _ensure_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            for budget_group, cap in limits.items():
                connection.execute("INSERT OR IGNORE INTO limits VALUES (?, ?)", (budget_group, cap))
                pinned = connection.execute("SELECT units FROM limits WHERE budget_group = ?", (budget_group,)).fetchone()[0]
                if cap != pinned:
                    raise BudgetError("experiment budget limits differ from the ledger's pinned limits")
            if connection.execute(
                "SELECT 1 FROM reservations WHERE budget_group = ? AND status = 'over_bound' LIMIT 1", (group,),
            ).fetchone():
                raise BudgetError("an earlier request breached its upper bound; new calls blocked")
            committed = connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN status = 'reserved' THEN upper_units ELSE cost_units END), 0) "
                "FROM reservations WHERE budget_group = ?", (group,),
            ).fetchone()[0]
            if committed + upper > limits[group]:
                raise BudgetError(
                    f"{group} budget exhausted: spent + outstanding ${committed / _SCALE:.6f} "
                    f"+ request bound ${upper / _SCALE:.6f} exceeds approved ${limits[group] / _SCALE:.2f}"
                )
            connection.execute(
                "INSERT INTO reservations VALUES (?, ?, ?, ?, ?, NULL, 'reserved', ?)",
                (reservation_id, group, model, stage, upper, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return Reservation(settings.experiment_id, reservation_id, group, upper / _SCALE, path)


def snapshot(settings: Settings) -> dict:
    """Read accounting evidence without creating a directory or ledger."""
    path = ledger_path(settings)
    result = {
        "experiment_id": settings.experiment_id,
        "groups": {
            "gemini": {"limit_usd": settings.gemini_budget_usd, "spent_usd": 0.0, "outstanding_usd": 0.0},
            "opus": {"limit_usd": settings.opus_budget_usd, "spent_usd": 0.0, "outstanding_usd": 0.0},
        },
        "reservations": [],
    }
    if not path.exists():
        return result
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
        if not tables:
            # A connection can make the file visible before schema initialization
            # commits. This read snapshot proves no accounting has committed yet;
            # it is also the state left by an older rejected-first-call rollback.
            return result
        if not {"limits", "reservations"} <= tables:
            raise BudgetError("budget ledger schema is incomplete; cannot prove accounting")
        for row in connection.execute("SELECT budget_group, units FROM limits"):
            result["groups"][row["budget_group"]]["limit_usd"] = row["units"] / _SCALE
        for row in connection.execute("SELECT * FROM reservations ORDER BY created_at, id"):
            group = result["groups"][row["budget_group"]]
            if row["status"] == "reserved":
                group["outstanding_usd"] += row["upper_units"] / _SCALE
            else:
                group["spent_usd"] += row["cost_units"] / _SCALE
            result["reservations"].append({
                "reservation_id": row["id"], "group": row["budget_group"], "model": row["model"],
                "stage": row["stage"], "upper_usd": row["upper_units"] / _SCALE,
                "status": row["status"], "cost_usd": None if row["cost_units"] is None else row["cost_units"] / _SCALE,
            })
    return result
