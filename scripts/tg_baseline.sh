#!/bin/bash
# Resolve the plain-decode baseline: measure both kernels at t=8 and t=16,
# interleaved, selecting the variant via LD_LIBRARY_PATH (never overwriting a
# mapped .so, which segfaults running processes).
set -u
B=${LLAMA_BENCH:?Set LLAMA_BENCH to the llama-bench executable}
M=${MODEL:?Set MODEL to the model file}
STOCK_LIBS=${STOCK_LIBS:?Set STOCK_LIBS to the stock library directory}
OURS_LIBS=${OURS_LIBS:?Set OURS_LIBS to the optimized library directory}
OUT=${OUT:-tg_baseline.txt}

pkill -9 -x llama-bench 2>/dev/null
pkill -9 -x llama-server 2>/dev/null
sleep 3

: > "$OUT"
for round in 1 2; do
  for v in stock ours; do
    if [[ "$v" == stock ]]; then LIBS=$STOCK_LIBS; else LIBS=$OURS_LIBS; fi
    echo "===== variant=$v round=$round threads=16 =====" >> "$OUT"
    LD_LIBRARY_PATH="$LIBS" "$B" -m "$M" -n 128 -t 16 -r 5 \
      2>/dev/null | grep tg128 >> "$OUT"
    echo "===== variant=$v round=$round threads=8 =====" >> "$OUT"
    LD_LIBRARY_PATH="$LIBS" "$B" -m "$M" -n 128 -t 8 -r 5 \
      2>/dev/null | grep tg128 >> "$OUT"
  done
done
echo "ALLDONE" >> "$OUT"
