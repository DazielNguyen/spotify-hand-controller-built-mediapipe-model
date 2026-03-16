"""Evaluate app-level music-control reliability metrics.

Metrics:
1. Command Success Rate
2. False Trigger Rate (per minute)
3. Mean Command Latency (ms)

Expected input JSON format:
{
  "sessions": [
    {
      "session_id": "s1",
      "duration_seconds": 120.0,
      "intended_events": [
        {"t": 10.2, "command": "play_pause"},
        {"t": 24.7, "command": "next_track"}
      ],
      "triggered_events": [
        {"t": 10.5, "command": "play_pause"},
        {"t": 15.0, "command": "volume_up"}
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    """Gesture/control event timestamped in seconds."""

    t: float
    command: str


@dataclass(frozen=True)
class SessionResult:
    """Computed metrics for one evaluation session."""

    session_id: str
    intended_count: int
    matched_count: int
    false_trigger_count: int
    mean_latency_ms: float
    duration_seconds: float


@dataclass(frozen=True)
class OverallResult:
    """Aggregate metrics across all sessions."""

    total_intended: int
    total_matched: int
    total_false_triggers: int
    total_duration_seconds: float
    command_success_rate: float
    false_trigger_rate_per_minute: float
    mean_command_latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate app-level control reliability")
    parser.add_argument(
        "--sessions-log",
        required=True,
        help="Path to evaluation sessions JSON",
    )
    parser.add_argument(
        "--match-window-sec",
        type=float,
        default=1.5,
        help="Max allowed delay (seconds) to match trigger to intended event",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output path to save metric summary as JSON",
    )
    return parser.parse_args()


def _parse_event(raw: dict[str, Any]) -> Event:
    if "t" not in raw or "command" not in raw:
        raise ValueError(f"Invalid event: {raw}")
    return Event(t=float(raw["t"]), command=str(raw["command"]))


def _parse_sessions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("Expected top-level key 'sessions' as a list")
    return sessions


def _match_events(
    intended_events: list[Event],
    triggered_events: list[Event],
    match_window_sec: float,
) -> tuple[int, int, list[float]]:
    """Greedy matching by command and time.

    A trigger is considered a match if:
    - same command as intended event
    - trigger time is >= intended time
    - trigger time - intended time <= match_window_sec
    """
    matched_count = 0
    false_trigger_count = 0
    latencies_ms: list[float] = []

    # Track which trigger events were consumed by intended events.
    used_trigger_indices: set[int] = set()

    for intended in intended_events:
        best_index: int | None = None
        best_delay: float | None = None

        for idx, triggered in enumerate(triggered_events):
            if idx in used_trigger_indices:
                continue
            if triggered.command != intended.command:
                continue
            delay = triggered.t - intended.t
            if delay < 0:
                continue
            if delay > match_window_sec:
                continue
            if best_delay is None or delay < best_delay:
                best_delay = delay
                best_index = idx

        if best_index is not None and best_delay is not None:
            used_trigger_indices.add(best_index)
            matched_count += 1
            latencies_ms.append(best_delay * 1000.0)

    false_trigger_count = len(triggered_events) - len(used_trigger_indices)
    return matched_count, false_trigger_count, latencies_ms


def evaluate_sessions(sessions: list[dict[str, Any]], match_window_sec: float) -> tuple[list[SessionResult], OverallResult]:
    session_results: list[SessionResult] = []

    total_intended = 0
    total_matched = 0
    total_false = 0
    total_duration = 0.0
    all_latencies_ms: list[float] = []

    for raw in sessions:
        session_id = str(raw.get("session_id", "unknown"))
        duration_seconds = float(raw.get("duration_seconds", 0.0))

        intended_events = [_parse_event(e) for e in raw.get("intended_events", [])]
        triggered_events = [_parse_event(e) for e in raw.get("triggered_events", [])]

        intended_events.sort(key=lambda e: e.t)
        triggered_events.sort(key=lambda e: e.t)

        matched_count, false_count, latencies_ms = _match_events(
            intended_events=intended_events,
            triggered_events=triggered_events,
            match_window_sec=match_window_sec,
        )

        mean_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        session_results.append(
            SessionResult(
                session_id=session_id,
                intended_count=len(intended_events),
                matched_count=matched_count,
                false_trigger_count=false_count,
                mean_latency_ms=mean_latency,
                duration_seconds=duration_seconds,
            )
        )

        total_intended += len(intended_events)
        total_matched += matched_count
        total_false += false_count
        total_duration += duration_seconds
        all_latencies_ms.extend(latencies_ms)

    command_success_rate = (total_matched / total_intended) if total_intended > 0 else 0.0
    minutes = (total_duration / 60.0) if total_duration > 0 else 0.0
    false_trigger_rate_per_minute = (total_false / minutes) if minutes > 0 else 0.0
    mean_command_latency_ms = (
        sum(all_latencies_ms) / len(all_latencies_ms) if all_latencies_ms else 0.0
    )

    overall = OverallResult(
        total_intended=total_intended,
        total_matched=total_matched,
        total_false_triggers=total_false,
        total_duration_seconds=total_duration,
        command_success_rate=command_success_rate,
        false_trigger_rate_per_minute=false_trigger_rate_per_minute,
        mean_command_latency_ms=mean_command_latency_ms,
    )

    return session_results, overall


def _as_dict(session_results: list[SessionResult], overall: OverallResult) -> dict[str, Any]:
    return {
        "session_results": [
            {
                "session_id": r.session_id,
                "intended_count": r.intended_count,
                "matched_count": r.matched_count,
                "false_trigger_count": r.false_trigger_count,
                "mean_latency_ms": round(r.mean_latency_ms, 3),
                "duration_seconds": r.duration_seconds,
            }
            for r in session_results
        ],
        "overall": {
            "total_intended": overall.total_intended,
            "total_matched": overall.total_matched,
            "total_false_triggers": overall.total_false_triggers,
            "total_duration_seconds": overall.total_duration_seconds,
            "command_success_rate": round(overall.command_success_rate, 6),
            "false_trigger_rate_per_minute": round(overall.false_trigger_rate_per_minute, 6),
            "mean_command_latency_ms": round(overall.mean_command_latency_ms, 3),
        },
    }


def main() -> None:
    args = parse_args()
    sessions_path = Path(args.sessions_log)

    sessions = _parse_sessions(sessions_path)
    session_results, overall = evaluate_sessions(
        sessions=sessions,
        match_window_sec=args.match_window_sec,
    )

    report = _as_dict(session_results, overall)
    print(json.dumps(report, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
