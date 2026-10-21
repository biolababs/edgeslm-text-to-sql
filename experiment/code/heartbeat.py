#!/usr/bin/env python3
"""Write status/heartbeat.json for the team's live monitor (coordinator requirement, 2026-10-16).
    python heartbeat.py --step s06 --task "Register models" --state running --done 3 --total 9 --unit models \
        --expected-minutes 10 [--started-at ISO] [--jetson-log PATH] --note "..."
--started-at defaults to the previous started_at if the task name is unchanged, else now."""
import argparse, datetime, json
from pathlib import Path
P = Path("C:/Users/biola/paper/projects/c9-text-to-sql/status/heartbeat.json")
ap = argparse.ArgumentParser()
for k in ("step", "task", "state", "unit", "note", "started-at", "jetson-log", "pc-log"):
    ap.add_argument("--" + k, default="")
ap.add_argument("--done", type=int, default=0); ap.add_argument("--total", type=int, default=0)
ap.add_argument("--expected-minutes", type=float, default=0)
a = ap.parse_args()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
prev = json.loads(P.read_text()) if P.exists() else {}
started = a.started_at or (prev.get("started_at") if prev.get("task") == a.task else now)
P.write_text(json.dumps({"topic": "c9-text-to-sql", "step": a.step, "task": a.task, "state": a.state, "done": a.done,
    "total": a.total, "unit": a.unit, "started_at": started, "updated_at": now, "expected_minutes": a.expected_minutes,
    "jetson_log": a.jetson_log, "pc_log": a.pc_log, "note": a.note}, indent=1))
