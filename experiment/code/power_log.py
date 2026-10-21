#!/usr/bin/env python3
"""Run tegrastats and prefix every line with time.time_ns() (PLAN 6.2). Stdlib only.

tegrastats is the measuring tool; this wrapper only adds sub-second wall-clock timestamps so
per-query energy can be integrated against client-side send/receive timestamps.

    python3 power_log.py --out power.log [--interval 100] [--duration 180]
Stop with SIGTERM/SIGINT (or --duration). Output lines: "<epoch_ns> <tegrastats line>".
"""
import argparse
import signal
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=int, default=100, help="tegrastats interval, ms")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until signalled")
    args = ap.parse_args()

    p = subprocess.Popen(["tegrastats", "--interval", str(args.interval)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    stop = {"flag": False}

    def handler(*_):
        stop["flag"] = True
        p.terminate()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    t_end = time.time() + args.duration if args.duration else None
    with open(args.out, "a", buffering=1) as f:
        f.write(f"# power_log start {time.time_ns()} interval_ms={args.interval}\n")
        for line in p.stdout:
            f.write(f"{time.time_ns()} {line.rstrip()}\n")
            if t_end and time.time() >= t_end:
                p.terminate()
            if stop["flag"]:
                break
        f.write(f"# power_log end {time.time_ns()}\n")
    p.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
