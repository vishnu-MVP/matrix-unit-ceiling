#!/bin/bash
# Definitive prefill A/B. Interleaved, both thread counts, variant selected via
# LD_LIBRARY_PATH so no mapped library is ever overwritten. This supersedes every
# earlier prefill measurement taken with the cp-over-.so method.
set -u
B=${LLAMA_BENCH:?Set LLAMA_BENCH to the llama-bench executable}
M=${MODEL:?Set MODEL to the model file}
STOCK_LIBS=${STOCK_LIBS:?Set STOCK_LIBS to the stock library directory}
OURS_LIBS=${OURS_LIBS:?Set OURS_LIBS to the optimized library directory}
OUT=${OUT:-pp_baseline.txt}

# never measure while another bench or a server is resident
while pgrep -x llama-bench > /dev/null; do sleep 20; done
pkill -9 -x llama-server 2>/dev/null
sleep 3

: > "$OUT"
for round in 1 2; do
  for v in stock ours; do
    if [[ "$v" == stock ]]; then LIBS=$STOCK_LIBS; else LIBS=$OURS_LIBS; fi
    for t in 8 16; do
      echo "===== variant=$v round=$round threads=$t =====" >> "$OUT"
      LD_LIBRARY_PATH="$LIBS" "$B" -m "$M" -p 512 -n 0 -t "$t" -r 5 \
        2>/dev/null | grep pp512 >> "$OUT"
    done
  done
done
echo "ALLDONE" >> "$OUT"
