#!/bin/bash
# Jetson test mode (PLAN 9). Run on the Jetson as the bench user (passwordless sudo; no password is
# used or stored anywhere). Never reads ~/jetson-agent/config/api-keys.txt; the agent's own
# health.sh is called as-is.
#
#   bash testmode.sh precheck <run_dir>   9.0 go/no-go (exit 1 = something is in use: stop and report)
#   bash testmode.sh enter    <run_dir>   9.1 snapshot + stop agent/gdm/timers/cron + Ollama drop-in + clocks + swap flush
#   bash testmode.sh verify   <run_dir>   9.2 180 s idle verification (exit 1 = fail)
#   bash testmode.sh baseline <run_dir> <name> [seconds]   board-idle power baseline (default 300 s)
#   bash testmode.sh exit     <run_dir>   9.4 restore + health check + restore diff
# <run_dir> is the run workspace on the Jetson (~/c9/<run-id>); env files go to <run_dir>/env/.
set -u
cmd=$1; RUN=$2
CODE=$(cd "$(dirname "$0")" && pwd)
ENV=$RUN/env; mkdir -p $ENV/pre $ENV/post
STATE=$HOME/c9-testmode-state; mkdir -p $STATE
AGENT=$HOME/jetson-agent
DROPIN=/etc/systemd/system/ollama.service.d/zz-c9-testmode.conf
TIMERS="trafficedge-ip-reporter.timer apt-daily.timer apt-daily-upgrade.timer fwupd-refresh.timer man-db.timer motd-news.timer update-notifier-download.timer update-notifier-motd.timer anacron.timer e2scrub_all.timer fstrim.timer dpkg-db-backup.timer logrotate.timer systemd-tmpfiles-clean.timer"
SERVICES="cron fwupd packagekit"
AGENT_PAT='llama-server|opencode|proxy/server.js|workflow/server.py|jetson-agent/scripts/run-server.sh'
AGENT_PORTS="8000 8001 8787 8788 8790 58699"
log() { echo "$(date -Is) $*" | tee -a $ENV/testmode.log; }

snapshot() {  # dir
  d=$1; mkdir -p $d
  systemctl list-units --type=service --state=running --no-pager --plain --no-legend > $d/services_running.txt
  systemctl list-timers --all --no-pager > $d/timers.txt
  systemctl list-units --type=timer --state=active --no-pager --plain --no-legend | awk '{print $1}' | sort > $d/timers_active.txt
  crontab -l > $d/crontab.txt 2>&1
  systemctl show ollama -p Environment > $d/ollama_env.txt
  systemctl cat ollama > $d/ollama_unit.txt 2>&1
  nvpmodel -q > $d/nvpmodel.txt 2>&1
  sudo jetson_clocks --show > $d/jetson_clocks.txt 2>&1
  swapon --show > $d/swapon.txt; free -m > $d/free.txt
  sudo ss -tlnp > $d/ss_listen.txt
  ss -tlnH | awk '{print $4}' | sort -u > $d/listen_addrs.txt
  pgrep -af "$AGENT_PAT" > $d/agent_procs.txt
  cat $AGENT/logs/supervisor.pid > $d/agent_supervisor_pid.txt 2>/dev/null
  ps -eo user,pid,pcpu,rss,etime,args --sort=-pcpu | head -40 > $d/ps_top.txt
  snap refresh --time > $d/snap_refresh.txt 2>&1
  sudo docker ps > $d/docker_ps.txt 2>&1
}

case $cmd in
precheck)
  { echo "== who"; who; echo "== ps"; ps -eo user,pcpu,rss,etime,args --sort=-pcpu | head -30
    echo "== ollama ps"; ollama ps
    echo "== established on 11434/8000/8787"; ss -tnp state established '( sport = :11434 or sport = :8000 or sport = :8787 )'
    echo "== last agent server.log lines"; tail -n 5 $AGENT/logs/server.log 2>/dev/null; ls -l --time-style=full-iso $AGENT/logs/server.log
  } > $ENV/precheck.txt 2>&1
  timeout 30 tegrastats --interval 1000 > $ENV/precheck_tegrastats.txt
  busy=0
  [ "$(ollama ps | wc -l)" -gt 1 ] && { log "PRECHECK: ollama has a model loaded"; busy=1; }
  grep -q 'GR3D_FREQ [1-9]' $ENV/precheck_tegrastats.txt && { log "PRECHECK: GPU load seen"; busy=1; }
  [ "$(ss -tnH state established '( sport = :11434 or sport = :8000 or sport = :8787 )' | wc -l)" -gt 0 ] && { log "PRECHECK: client connections on 11434/8000/8787"; busy=1; }
  [ "$(pgrep -fc '[r]un_ollama.py|[l]oadgen_bench.py|[f]inetune.py|[l]lama-bench')" -gt 0 ] && { log "PRECHECK: another benchmark running"; busy=1; }
  log "precheck busy=$busy"; exit $busy ;;

enter)
  log "ENTER test mode"
  snapshot $ENV/pre
  cp -r $ENV/pre $STATE/pre-$(basename $RUN)
  ln -sfn $STATE/pre-$(basename $RUN) $STATE/latest
  # 2. jetson-agent stack (supervisor trap stops llama-server, opencode, proxy, workflow)
  bash $AGENT/scripts/stop.sh; log "agent stop.sh rc=$?"
  for i in $(seq 1 60); do pgrep -f "$AGENT_PAT" > /dev/null || break; sleep 1; done
  sleep 5
  if pgrep -af "$AGENT_PAT"; then log "FAIL: agent processes still running"; pgrep -af "$AGENT_PAT" >> $ENV/testmode.log; exit 1; fi
  for p in $AGENT_PORTS; do ss -tlnH "( sport = :$p )" | grep -q . && { log "FAIL: port $p still open"; exit 1; }; done
  log "agent stack stopped; ports closed"
  # 3. display manager
  sudo systemctl stop gdm; log "gdm stopped rc=$?"
  # 4. periodic jobs: stopped, never disabled; remember which were active
  : > $STATE/stopped_units.txt
  for t in $TIMERS; do systemctl is-active --quiet $t && { sudo systemctl stop $t && echo $t >> $STATE/stopped_units.txt; }; done
  for s in $SERVICES; do systemctl is-active --quiet $s && { sudo systemctl stop $s && echo $s >> $STATE/stopped_units.txt; }; done
  cp $STATE/stopped_units.txt $ENV/stopped_units.txt
  sudo snap refresh --hold=96h > $ENV/snap_hold.txt 2>&1; log "snap hold rc=$?"
  log "stopped units: $(tr '\n' ' ' < $STATE/stopped_units.txt)"
  [ -z "$(sudo docker ps -q)" ] || log "WARNING: docker containers running"
  # 5. Ollama: loopback only, f16 KV cache (D7); other variables unchanged
  sudo mkdir -p $(dirname $DROPIN)
  printf '[Service]\nEnvironment="OLLAMA_HOST=127.0.0.1:11434"\nEnvironment="OLLAMA_KV_CACHE_TYPE=f16"\n' | sudo tee $DROPIN > /dev/null
  sudo systemctl daemon-reload && sudo systemctl restart ollama
  for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version > /dev/null && break; sleep 1; done
  systemctl show ollama -p Environment > $ENV/ollama_env_testmode.txt
  grep -q 'OLLAMA_HOST=127.0.0.1:11434' $ENV/ollama_env_testmode.txt && grep -q 'OLLAMA_KV_CACHE_TYPE=f16' $ENV/ollama_env_testmode.txt \
    || { log "FAIL: ollama env not applied"; exit 1; }
  ss -tlnH '( sport = :11434 )' > $ENV/ollama_listen_testmode.txt
  grep -q '127.0.0.1:11434' $ENV/ollama_listen_testmode.txt && ! grep -qE '\*:11434|0.0.0.0:11434' $ENV/ollama_listen_testmode.txt \
    || { log "FAIL: ollama not loopback-only"; exit 1; }
  log "ollama restarted loopback-only with f16 KV cache"
  # 6. power and clocks
  sudo nvpmodel -m 2 > /dev/null 2>&1; nvpmodel -q > $ENV/nvpmodel_testmode.txt 2>&1
  grep -q MAXN_SUPER $ENV/nvpmodel_testmode.txt || { log "FAIL: not MAXN_SUPER"; exit 1; }
  sudo jetson_clocks; sudo jetson_clocks --show > $ENV/jetson_clocks_testmode.txt 2>&1
  [ "$(cat /sys/class/devfreq/17000000.gpu/min_freq)" = "$(cat /sys/class/devfreq/17000000.gpu/max_freq)" ] || { log "FAIL: GPU clock not locked"; exit 1; }
  log "MAXN_SUPER, jetson_clocks on (GPU $(cat /sys/class/devfreq/17000000.gpu/cur_freq) Hz)"
  # 7. swap flush (only if free RAM covers used swap with 1 GB margin), drop caches
  used=$(free -m | awk '/Swap/{print $3}'); avail=$(free -m | awk '/Mem/{print $7}')
  if [ $((avail - used)) -gt 1024 ]; then sudo swapoff -a && sudo swapon -a; log "swap flushed (was ${used} MB)";
  else log "WARNING: swap not flushed: used=${used} MB avail=${avail} MB"; fi
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
  swapon --show > $ENV/swapon_testmode.txt; free -m > $ENV/free_testmode.txt
  log "ENTER done" ;;

verify)
  log "VERIFY idle (180 s)"
  python3 $CODE/power_log.py --out $ENV/idle_verify_power.log --interval 100 --duration 180 &
  PL=$!
  top -b -d 5 -n 36 > $ENV/idle_verify_top.txt &
  TP=$!
  wait $PL; wait $TP
  ollama ps > $ENV/idle_verify_ollama_ps.txt
  sudo ss -tlnp > $ENV/idle_verify_ss.txt 2>&1
  free -m > $ENV/idle_verify_free.txt
  python3 $CODE/idle_check.py --power $ENV/idle_verify_power.log --top $ENV/idle_verify_top.txt \
    --ollama-ps $ENV/idle_verify_ollama_ps.txt --ss $ENV/idle_verify_ss.txt --free $ENV/idle_verify_free.txt \
    > $ENV/idle_verify.txt
  rc=$?; cat $ENV/idle_verify.txt; log "VERIFY rc=$rc"; exit $rc ;;

baseline)
  name=$3; secs=${4:-300}
  log "BASELINE $name ${secs}s"
  python3 $CODE/power_log.py --out $ENV/baseline_$name.log --interval 100 --duration $secs
  python3 $CODE/idle_check.py --power $ENV/baseline_$name.log --summary-only > $ENV/baseline_$name.json
  cat $ENV/baseline_$name.json; log "BASELINE $name done" ;;

exit)
  log "EXIT test mode"
  for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done
  sudo rm -f $DROPIN; sudo systemctl daemon-reload; sudo systemctl restart ollama
  for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version > /dev/null && break; sleep 1; done
  PRE=$STATE/latest
  systemctl show ollama -p Environment > $ENV/post/ollama_env.txt
  if diff -q $PRE/ollama_env.txt $ENV/post/ollama_env.txt > /dev/null; then log "ollama env restored"; else log "FAIL: ollama env differs"; fi
  ss -tlnH '( sport = :11434 )' | grep -q '\*:11434' && log "ollama listening on *:11434" || log "FAIL: ollama not on *:11434"
  for u in $(cat $STATE/stopped_units.txt); do sudo systemctl start $u; log "start $u rc=$?"; done
  sudo snap refresh --unhold > $ENV/snap_unhold.txt 2>&1; log "snap unhold rc=$?"
  sudo systemctl start gdm; log "gdm start rc=$?"
  # clocks: left as found at entry (locked); D6 says reboot after the campaign, not now
  # agent: start as trafficedge01 via its own script, then its own health check
  setsid nohup bash $AGENT/scripts/start.sh > /dev/null 2>&1 < /dev/null &
  ok=1
  for i in $(seq 1 24); do
    sleep 5
    if bash $AGENT/scripts/health.sh > $ENV/post/agent_health.txt 2>&1; then ok=0; break; fi
  done
  log "agent health.sh rc=$ok after $((i * 5)) s"
  for p in 8000 8001 8787 8788 8790; do ss -tlnH "( sport = :$p )" | grep -q . && log "port $p listening" || log "FAIL: port $p not listening"; done
  pgrep -af 'llama-server.*liquid-general' > /dev/null && log "liquid-general llama-server loaded" || log "WARNING: liquid-general llama-server not (yet) loaded"
  sleep 20
  snapshot $ENV/post
  { echo "== services (pre vs post; < pre only, > post only)"
    diff <(awk '{print $1}' $PRE/services_running.txt | sort) <(awk '{print $1}' $ENV/post/services_running.txt | sort)
    echo "== active timers"; diff $PRE/timers_active.txt $ENV/post/timers_active.txt
    echo "== listening addresses"; diff $PRE/listen_addrs.txt $ENV/post/listen_addrs.txt
    echo "== ollama env"; diff $PRE/ollama_env.txt $ENV/post/ollama_env.txt
    echo "== nvpmodel"; diff $PRE/nvpmodel.txt $ENV/post/nvpmodel.txt
    echo "== agent processes (command lines)"; diff <(awk '{$1=""; print}' $PRE/agent_procs.txt | sort) <(awk '{$1=""; print}' $ENV/post/agent_procs.txt | sort)
  } > $ENV/restore_diff.txt 2>&1
  cat $ENV/restore_diff.txt
  log "EXIT done (health rc=$ok)"; exit $ok ;;
*) echo "usage: $0 precheck|enter|verify|baseline|exit <run_dir>"; exit 2 ;;
esac
