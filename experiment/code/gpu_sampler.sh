#!/bin/bash
# 1 s sysfs sampler (PLAN 6.2): GPU devfreq cur_freq, CPU scaling_cur_freq (cpu0, cpu4),
# tj temperature, and the INA3221 crit/max alarm flags. Runs until killed.
#   bash gpu_sampler.sh <out.tsv>
out=$1
H=$(ls -d /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon* | head -1)
TJ=""
for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = "tj-thermal" ] && TJ=$z/temp; done
echo -e "epoch_ns\tgpu_cur_hz\tcpu0_khz\tcpu4_khz\ttj_mC\tcrit_alarms\tmax_alarms" >> "$out"
while true; do
  crit=$(cat $H/curr*_crit_alarm 2>/dev/null | tr '\n' ',' )
  mx=$(cat $H/curr*_max_alarm 2>/dev/null | tr '\n' ',' )
  echo -e "$(date +%s%N)\t$(cat /sys/class/devfreq/17000000.gpu/cur_freq)\t$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)\t$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq)\t$(cat $TJ)\t$crit\t$mx" >> "$out"
  sleep 1
done
