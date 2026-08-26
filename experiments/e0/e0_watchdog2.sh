#!/bin/bash
# E0 看门狗 v2: 等 GPU 空闲(util<=20 持续 2 分钟)后按序跑 E0a + E0b-FFT + E0b-direct
# 结果与环境快照写 ~/e0_results.log; util<5% 标 USABLE, 5-20% 标 REFERENCE-ONLY
deadline=$(( $(date +%s) + 24*3600 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if [ "$u" -le 20 ]; then
    sleep 120
    u2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$u2" -le 20 ]; then
      {
        echo "=== E0 run at $(date) ==="
        echo "--- env snapshot ---"
        nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv,noheader
        nvidia-smi -q | grep -iE "cuda version" | head -1
        ~/envs/e0omm/bin/python -c "import openmm; print('openmm', openmm.__version__)" 2>/dev/null
        ~/.venvs/e0/bin/python -c "import cupy; from cupy.cuda import cufft; print('cupy', cupy.__version__)" 2>/dev/null
        if [ "$u2" -lt 5 ]; then echo "CONFIDENCE: USABLE (util<5%)"; else echo "CONFIDENCE: REFERENCE-ONLY (util 5-20%)"; fi
        echo "--- E0a: openmm mixed vs double ---"
        ~/envs/e0omm/bin/python /root/e0a_openmm_precision.py
        echo "--- E0b-FFT: batched cufft throughput ---"
        ~/.venvs/e0/bin/python /root/e0b_fft_throughput.py
        echo "--- E0b/E0d-direct: pair kernel ---"
        ~/.venvs/e0/bin/python /root/e0b_direct_pairbench.py
        echo "=== done $(date) ==="
      } >> ~/e0_results.log 2>&1
      exit 0
    fi
  fi
  sleep 120
done
echo "=== deadline reached, GPU never idle ===" >> ~/e0_results.log
