#!/usr/bin/env python3
"""Run the official test-suite-sql-eval evaluation.py (commit e97acc5) with two harness fixes
applied from outside; the evaluator's files and scoring logic are untouched (found in step 6):

1. Windows hang: exec_eval calls asyncio.run() once per query per database (tens of thousands
   of event loops). On Windows each loop opens a loopback TCP socket pair; after about 10,000
   loops the ephemeral ports are exhausted and the evaluator blocks forever with 0% CPU
   (observed twice, at questions ~150 and ~175 of 1034). We run the same query function
   synchronously instead, so no event loop is created.
2. Timeout: the evaluator's intended per-query limit (exec_eval.TIMEOUT = 60 s) never fires,
   because asyncio.wait_for cannot interrupt a blocking sqlite3 call. We enforce it with an
   SQLite progress handler; an aborted query returns ("exception", ...), exactly what the
   evaluator expects from a timeout.

Usage (cwd = the evaluator directory):
    python run_ts_eval.py --gold ... --pred ... --db ... --table ... --etype exec|match
"""
import os
import runpy
import sys
import time
import types

sys.path.insert(0, os.getcwd())
import exec_eval  # noqa: E402


def exec_on_db_sync(sqlite_path, query, process_id="", timeout=exec_eval.TIMEOUT):
    """Body of exec_eval.exec_on_db_ (unchanged), run synchronously, with an effective timeout."""
    query = exec_eval.replace_cur_year(query)
    cursor = exec_eval.get_cursor_from_path(sqlite_path)
    deadline = time.monotonic() + timeout
    cursor.connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 100_000)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        if time.monotonic() > deadline:
            return "exception", TimeoutError
        return "exception", e


exec_eval.exec_on_db = exec_on_db_sync
# exec_eval calls asyncio.run(exec_on_db(...)); exec_on_db now returns the result directly.
exec_eval.asyncio = types.SimpleNamespace(run=lambda x: x)
sys.argv = ["evaluation.py"] + sys.argv[1:]
runpy.run_path(os.path.join(os.getcwd(), "evaluation.py"), run_name="__main__")
