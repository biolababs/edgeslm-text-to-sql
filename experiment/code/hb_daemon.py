#!/usr/bin/env python3
"""PC-side heartbeat daemon for the team's live monitor (coordinator requirement). Every --every seconds it
reads the tail of a Jetson session log over ssh and rewrites status/heartbeat.json with the latest
'PROGRESS n/total' line and block. Stops when the session log shows SESSION*_DONE / SESSION_ABORTED, or
after --max-hours. Runs detached so it keeps working if the controlling agent is interrupted.

    pythonw hb_daemon.py --step s09 --session "Session 1 (B3/B4/B5)" --jetson-log /home/trafficedge01/c9/runs/<id>/logs/session.log \
        --expected-minutes 1200
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import time
from pathlib import Path

HB = Path("C:/Users/biola/paper/projects/c9-text-to-sql/status/heartbeat.json")
HOST = os.environ.get("JETSON_HOST", "")  # user@host for the test bed


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def probe(log):
    """ssh tail with no inherited stdin, output to a temp file (no pipes that can block after a kill), one retry."""
    import os, tempfile
    cmd = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
           "-o", "ServerAliveCountMax=2", HOST, f"tail -n 400 {log}"]
    last = None
    for _ in range(2):
        fd, tmp = tempfile.mkstemp(suffix=".hb"); os.close(fd)
        try:
            with open(tmp, "wb") as fo:
                p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=fo, stderr=subprocess.DEVNULL)
                try:
                    p.wait(timeout=45)
                except subprocess.TimeoutExpired as e:
                    p.kill(); last = e; continue
            if p.returncode == 0:
                return open(tmp, "rb").read().decode("utf-8", "replace")
            last = RuntimeError(f"ssh rc={p.returncode}")
        finally:
            try: os.remove(tmp)
            except OSError: pass
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="s09"); ap.add_argument("--session", required=True)
    ap.add_argument("--jetson-log", required=True); ap.add_argument("--pc-log", default="")
    ap.add_argument("--expected-minutes", type=float, default=0); ap.add_argument("--every", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=30)
    a = ap.parse_args()
    started = now(); t_end = time.time() + a.max_hours * 3600
    done, total, unit = 0, 0, "items"
    while time.time() < t_end:
        state, note = "running", ""
        try:
            out = probe(a.jetson_log)
            lines = [l for l in out.splitlines() if l.strip()]
            prog = [l for l in lines if l.startswith("PROGRESS ")]
            blk = [l for l in lines if "BLOCK" in l and "PROGRESS" not in l]
            flags = [l for l in lines if " FLAG " in l]
            if prog:
                m = re.match(r"PROGRESS (\d+)/(\d+)\s*(.*)", prog[-1])
                done, total = int(m.group(1)), int(m.group(2)); note = m.group(3)[:120]
                unit = "items"
                outer = [p for p in prog if re.search(r"(B3 models|B6 models|B7a folds|B7b folds|B8 |B8b |session)", p)]
                if outer:
                    note = f"{outer[-1][9:80]} | current: {note}"
            if blk:
                note = f"{blk[-1].split(' ', 1)[-1][:60]} | {note}"
            if flags:
                note += f" | flags so far: {len(flags)}"
            if any(re.search(r"SESSION\w*_DONE|SESSION_ABORTED", l) for l in lines):
                state = "done"; note = "session finished: " + [l for l in lines if re.search(r"SESSION\w*_DONE|SESSION_ABORTED", l)][-1]
        except Exception as e:  # probe failed twice: say so, keep the timestamp fresh, keep the last counts
            state, note = "running", f"heartbeat probe timed out ({type(e).__name__}); Jetson job runs unattended under nohup"
        HB.write_text(json.dumps({"topic": "c9-text-to-sql", "step": a.step, "task": a.session, "state": state,
                                  "done": done, "total": total, "unit": unit, "started_at": started, "updated_at": now(),
                                  "expected_minutes": a.expected_minutes, "jetson_log": a.jetson_log, "pc_log": a.pc_log,
                                  "note": note[:300]}, indent=1))
        if state == "done":
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
