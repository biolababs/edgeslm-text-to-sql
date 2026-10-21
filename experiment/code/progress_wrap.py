#!/usr/bin/env python3
"""Run a command, append its output (stdout+stderr) to a log, and turn tqdm-style 'n/total'
counters into 'PROGRESS n/total' lines (for the team's live monitor).

    python progress_wrap.py --log LOG [--count-bars N] --cwd DIR -- cmd args...

--count-bars N: the command prints one complete n/n bar per unit (e.g. the Spider test-suite
evaluator's per-question bar over its databases); count completed bars as units out of N.
"""
import argparse
import os
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--cwd", default=None)
ap.add_argument("--count-bars", type=int, default=0)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
pat = re.compile(r"(\d+)/(\d+)\s*\[")

with open(a.log, "a", encoding="utf-8", buffering=1) as log:
    log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} START {' '.join(cmd)}\n")
    p = subprocess.Popen(cmd, cwd=a.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=os.environ)
    last, buf, units, t_last = None, "", 0, time.time()
    while True:
        ch = p.stdout.read(1)
        if not ch:
            break
        if ch not in "\r\n":
            buf += ch
            continue
        m = pat.findall(buf)
        if m:
            n, tot = m[-1]
            if a.count_bars:
                if n == tot and last != "complete":
                    units += 1
                    if units % 5 == 0 or units == a.count_bars or time.time() - t_last > 60:
                        log.write(f"PROGRESS {units}/{a.count_bars}\n")
                        t_last = time.time()
                last = "complete" if n == tot else None
            elif f"{n}/{tot}" != last:
                log.write(f"PROGRESS {n}/{tot}\n")
                last = f"{n}/{tot}"
        elif buf.strip():
            log.write(buf + "\n")
        buf = ""
    if buf.strip():
        log.write(buf + "\n")
    rc = p.wait()
    log.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} END rc={rc}\n")
sys.exit(rc)
