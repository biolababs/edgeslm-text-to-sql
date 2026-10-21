#!/bin/bash
# Cool-down rule (PLAN 9.5): wait until tj <= T_idle + 2 C, minimum 180 s, maximum 1200 s.
#   bash cooldown.sh <T_idle_C> <log_file>
# T_idle is the session's idle-baseline median tj. Logs the outcome; exits 0 either way,
# writing FLAG cooldown_max_reached when the 20 min limit is hit.
tidle=$1; log=$2
TJ=""
for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = "tj-thermal" ] && TJ=$z/temp; done
target=$(python3 -c "print(int(($tidle + 2) * 1000))")
t0=$(date +%s)
while true; do
  now=$(date +%s); el=$((now - t0)); tj=$(cat $TJ)
  if [ $el -ge 180 ] && [ $tj -le $target ]; then
    echo "$(date -Is) cooldown ok elapsed=${el}s tj_mC=$tj target_mC=$target" >> "$log"; exit 0
  fi
  if [ $el -ge 1200 ]; then
    echo "$(date -Is) FLAG cooldown_max_reached elapsed=${el}s tj_mC=$tj target_mC=$target" >> "$log"; exit 0
  fi
  [ $((el % 60)) -lt 5 ] && echo "PROGRESS $el/1200 cooldown tj_mC=$tj target_mC=$target"
  sleep 5
done
