#!/bin/bash
# Environment record (PLAN 9.3). Run on the Jetson: bash env_record.sh <run_dir> [tag ...]
# Writes <run_dir>/env/record_<UTC time>/. Never reads the agent's api-keys file.
RUN=$1; shift
D=$RUN/env/record_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $D
c() { echo "\$ $*"; eval "$@" 2>&1; echo; }
{
c cat /etc/nv_tegra_release
c "tr -d '\0' < /proc/device-tree/model; echo"
c dpkg -l nvidia-jetpack nvidia-l4t-core
c uname -a
c nvpmodel -q
c sudo jetson_clocks --show
c free -m; c swapon --show; c df -h / ; c uptime
c cat /sys/class/devfreq/17000000.gpu/cur_freq /sys/class/devfreq/17000000.gpu/min_freq /sys/class/devfreq/17000000.gpu/max_freq
} > $D/platform.txt
{
c ollama -v
c systemctl show ollama -p Environment
c ollama list
c ollama ps
} > $D/ollama.txt
for m in "$@"; do
  f=$(echo $m | tr ':/' '__')
  ollama show --modelfile $m > $D/modelfile_$f.txt 2>&1
  blob=$(awk '/^FROM /{print $2; exit}' $D/modelfile_$f.txt)
  [ -f "$blob" ] && echo "$(sha256sum $blob | awk '{print $1}')  $m  $blob" >> $D/model_sha256.txt
done
{
c "nvidia-smi 2>&1 | head -12"
c git -C $HOME/bench/llama.cpp log -1 --format='%H %cd'
c git -C $HOME/bench/test-suite-sql-eval log -1 --format=%H
c git -C $HOME/bench/mini_dev log -1 --format=%H
c python3 -V
c "python3 -c 'import sqlite3; print(sqlite3.sqlite_version)'"
c "$HOME/bench/ftenv/bin/pip freeze"
c "$HOME/bench/ftenv/bin/python -c 'import mlperf_loadgen as lg; print(lg.__file__)'"
c "$HOME/bench/ftenv/bin/pip show mlcommons-loadgen | head -3"
c "LD_LIBRARY_PATH=$HOME/bench/cudss/lib $HOME/bench/ftenv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'"
} > $D/software.txt
{
c systemctl list-units --type=service --state=running --no-pager --plain --no-legend
c systemctl list-timers --all --no-pager
c "ps -eo user,pid,pcpu,rss,etime,args --sort=-pcpu | head -30"
c "for z in /sys/class/thermal/thermal_zone*; do echo \$z \$(cat \$z/type) \$(cat \$z/temp 2>/dev/null); done"
c "cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"
c sudo ss -tlnp
} > $D/services.txt
echo $D
