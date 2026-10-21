"""Drives the Jetson over SSH from the PC. Settings live in bench/jetson.json.
  python tools/jetson.py check                 is the Jetson reachable and ready?
  python tools/jetson.py push                  copy the runner, models.txt and questions over
  python tools/jetson.py start RUN_ID [ARGS]   start a run in the background (ARGS go to jetson_runner.py)
  python tools/jetson.py status RUN_ID         show progress
  python tools/jetson.py stop RUN_ID           stop a run (rerun `start` with the same RUN_ID to resume)
  python tools/jetson.py pull RUN_ID           copy results into results/raw/RUN_ID and fingerprint them"""
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
CONFIG = BENCH / "jetson.json"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def config():
    if not CONFIG.exists():
        sys.exit(f'Create {CONFIG.relative_to(ROOT)} with {{"host": "user@jetson-ip"}}')
    cfg = json.loads(CONFIG.read_text())
    cfg.setdefault("remote_dir", "slm-bench")
    return cfg


def ssh(cfg, command, check=True, capture=False):
    result = subprocess.run(["ssh", *SSH_OPTS, cfg["host"], command], text=True,
                            capture_output=capture)
    if check and result.returncode != 0:
        sys.exit(f"SSH command failed ({result.returncode}): {command}\n{result.stderr or ''}")
    return result


def scp(*args):
    if subprocess.run(["scp", *SSH_OPTS, "-q", *args]).returncode != 0:
        sys.exit(f"scp failed: {' '.join(args)}")


def check(cfg):
    script = (
        "echo \"device:   $(tr -d '\\0' < /proc/device-tree/model 2>/dev/null)\"; "
        "echo \"L4T:      $(head -n1 /etc/nv_tegra_release 2>/dev/null)\"; "
        "echo \"python3:  $(python3 --version 2>&1)\"; "
        "echo \"ollama:   $(ollama --version 2>&1 | tail -n1)\"; "
        "echo \"server:   $(curl -s http://127.0.0.1:11434/api/version || echo NOT RUNNING)\"; "
        "echo \"power:    $(nvpmodel -q 2>/dev/null | head -n1)\"; "
        "echo \"tegrastats: $(command -v tegrastats || echo missing)\"; "
        "echo \"disk free: $(df -h ~ | tail -n1 | awk '{print $4}')\"; "
        "echo \"memory:   $(free -m | awk '/Mem:/ {print $2\" MB total, \"$7\" MB available\"}')\"; "
        "echo \"models:\"; ollama list 2>/dev/null"
    )
    ssh(cfg, script)


def push(cfg):
    d = cfg["remote_dir"]
    ssh(cfg, f"mkdir -p {d}/questions")
    scp(str(BENCH / "jetson_runner.py"), str(BENCH / "models.txt"), f"{cfg['host']}:{d}/")
    scp(*[str(p) for p in sorted((BENCH / "questions").glob("*.jsonl"))], f"{cfg['host']}:{d}/questions/")
    print(f"Copied runner, models.txt and {len(list((BENCH / 'questions').glob('*.jsonl')))} question files "
          f"to {cfg['host']}:{d}")


def start(cfg, run_id, extra):
    d = cfg["remote_dir"]
    args = " ".join(shlex.quote(a) for a in extra)
    ssh(cfg, f"cd {d} && nohup python3 jetson_runner.py --run-id {shlex.quote(run_id)} {args} "
             f"> runs-{run_id}.out 2>&1 < /dev/null & echo started PID $!")


def status(cfg, run_id):
    d = cfg["remote_dir"]
    ssh(cfg, f"cd {d}; if [ -f runs/{run_id}/COMPLETE ]; then echo STATE: COMPLETE; "
             f"elif pgrep -f 'jetson_runner.py --run-id {run_id}' >/dev/null; then echo STATE: RUNNING; "
             f"else echo STATE: STOPPED; fi; tail -n 8 runs/{run_id}/progress.log 2>/dev/null; "
             f"tail -n 3 runs-{run_id}.out 2>/dev/null", check=False)


def stop(cfg, run_id):
    ssh(cfg, f"pkill -f 'jetson_runner.py --run-id {run_id}' && echo stopped || echo not running", check=False)


def pull(cfg, run_id):
    dest = ROOT / "results" / "raw"
    if (dest / run_id).exists():
        sys.exit(f"results/raw/{run_id} already exists; raw results are never overwritten. Use a new RUN_ID.")
    scp("-r", f"{cfg['host']}:{cfg['remote_dir']}/runs/{run_id}", str(dest))
    print(f"Copied results to results/raw/{run_id}")
    subprocess.run([sys.executable, str(ROOT / "tools" / "fingerprint.py")], check=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"check", "push", "start", "status", "stop", "pull"}:
        sys.exit(__doc__)
    cfg = config()
    cmd = sys.argv[1]
    if cmd in {"check", "push"}:
        {"check": check, "push": push}[cmd](cfg)
        return
    if len(sys.argv) < 3:
        sys.exit(f"Usage: python tools/jetson.py {cmd} RUN_ID")
    run_id = sys.argv[2]
    if cmd == "start":
        start(cfg, run_id, sys.argv[3:])
    else:
        {"status": status, "stop": stop, "pull": pull}[cmd](cfg, run_id)


if __name__ == "__main__":
    main()
