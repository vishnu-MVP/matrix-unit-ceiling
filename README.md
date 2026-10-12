# The Matrix-Unit Ceiling -- artifacts

Code and measurements accompanying the paper *"The Matrix-Unit Ceiling:
Characterizing the Limits of AMX Acceleration for Hybrid-Attention LLM
Inference on CPUs."*

## Contents

| path | what it is |
|---|---|
| `kernel/pq2_amx_kernel.patch` | the ternary AMX tile GEMM (+288/-10, 3 files) |
| `kernel/pr-amx-tile-config-latch.md` | upstream fix: reload tile config when the palette changes |
| `kernel/issue-deltanet-multiseq-nan.md` | upstream defect report: gated delta-net emits non-finite values when a batch holds more than one sequence |
| `scripts/pp_baseline.sh` | interleaved prefill A/B (`pp512`) |
| `scripts/tg_baseline.sh` | interleaved decode A/B (`tg128`) |
| `scripts/mtp_bench.py` | speculative-decoding throughput and draft acceptance |
| `scripts/mtp_summary.py` | summarises `mtp_bench.py` output |
| `results/results.json` | every measured number, with the machine state it came from |
| `figures.py` | regenerates the paper figures |

## Applying the kernel patch

The patch does **not** apply to upstream `ggml-org/llama.cpp`. It targets a
fork that defines the `PQ2_0` ternary type, which upstream does not have:

```
base repo:    https://github.com/PrismML-Eng/llama.cpp
base commit:  87268f7
licence:      MIT
```

Verified from a pristine checkout:

```bash
git clone https://github.com/PrismML-Eng/llama.cpp llama.cpp-pq2
cd llama.cpp-pq2
git checkout 87268f7
git apply /path/to/kernel/pq2_amx_kernel.patch     # 3 files changed, +288/-10
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release -j16
```

Requires an x86-64 CPU with `amx_tile`, `amx_int8` and `avx512_vnni`. The AMX
sources are compiled only on x86; on ARM they are excluded entirely.

## Reproducing the measurements

All scripts take paths from environment variables, so nothing is hardcoded:

```bash
export LLAMA_BENCH=/path/to/build/bin/llama-bench
export MODEL=/path/to/model.gguf
export STOCK_LIBS=/path/to/libs-stock     # immutable snapshot, baseline build
export OURS_LIBS=/path/to/libs-ours       # immutable snapshot, patched build
./scripts/pp_baseline.sh
./scripts/tg_baseline.sh
```

Variant selection uses `LD_LIBRARY_PATH` pointed at **immutable snapshot
directories**. Do not copy a shared library over one a running process has
mapped: that rewrites the code pages of the live process, which produced
segfaults and corrupted timings during this work.

Speculative decoding, against a running `llama-server`:

```bash
export OUTPUT_DIR=./results
python3 scripts/mtp_bench.py 8099 withmtp 2
python3 scripts/mtp_summary.py
```

## Measurement protocol

Numbers in `results/results.json` follow these rules, and re-runs should too:

- baseline and treatment alternate **within one machine state**
- five repetitions per invocation, two interleaved rounds
- every decode figure states its thread count: the same two binaries differ by
  25% at `t=8` and 1% at `t=16`, because at `t=8` the baseline has not yet
  saturated memory bandwidth
- no ratio is ever formed from numbers measured in different sessions
- no conclusion rests on a gap smaller than the measured dispersion

Each entry in `results.json` records the machine state it was measured under.
Entries from different states must not be combined into a ratio.

## Known limitations

- `figures.py` does not yet read `results/results.json`; the values are still
  inline. Consolidating them is outstanding.
- The `precision_floor` entry is retracted and is not reproducible.
- Measurements come from a single platform (Sapphire Rapids, 8 physical cores,
  93.7 GB/s measured read bandwidth). The Amdahl bound in the paper is
  platform-independent; the constant 172 tok/s is not.

## Licence

The kernel patch derives from llama.cpp and is MIT licensed; `LICENSE-llama.cpp`
is included. Scripts and results in this repository are MIT.
