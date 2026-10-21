"""Harness fix for the BIRD mini_dev evaluator (found in step 6), active only when the env var
C9_SQLITE_DEADLINE_S is set and this directory is on PYTHONPATH (so it also reaches the
evaluator's multiprocessing workers).

The evaluator's --meta_time_out uses func_timeout, which cannot interrupt a running sqlite3
query: the worker keeps executing (observed: gold question 518 runs 290 s, question 701 runs
1,954 s on the scoring PC) and the run blocks until it finishes. Here every sqlite3.connect()
gets an SQLite progress handler that aborts the query once C9_SQLITE_DEADLINE_S seconds have
passed since the connection was opened. The evaluator opens one connection per (prediction,
gold) pair, so this is the same budget func_timeout was meant to enforce; an aborted query
raises sqlite3.OperationalError, which the evaluator scores 0, exactly like its timeout path.
"""
import os

_T = os.environ.get("C9_SQLITE_DEADLINE_S")
if _T:
    import sqlite3
    import time

    _orig_connect = sqlite3.connect

    def _connect(*args, **kwargs):
        conn = _orig_connect(*args, **kwargs)
        deadline = time.monotonic() + float(_T)
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 100_000)
        return conn

    sqlite3.connect = _connect
