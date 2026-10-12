#!/usr/bin/env python3
"""Measure deployed decode throughput with and without MTP speculative decoding.

Runs against a llama-server instance and reports per-request decode tok/s from the
server's own timings block, plus draft acceptance when speculation is active.
Usage: mtp_bench.py <port> <label> <n_trials>
"""
import json
import os
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")

# Fixed prompts so the comparison is content-matched: acceptance is content-dependent,
# so MTP and non-MTP must be driven with identical text.
PROMPTS = [
    "Explain how a CPU cache hierarchy works, covering L1, L2 and L3, "
    "inclusive versus exclusive policies, and why cache line size matters.",
    "Describe the differences between a matrix multiplication implemented with "
    "SIMD instructions and one implemented with dedicated matrix tile instructions.",
    "Write a clear technical explanation of why autoregressive decoding in large "
    "language models is memory bandwidth bound rather than compute bound.",
]


def one(prompt, n_predict=160):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    wall = time.time() - t0
    tim = out.get("timings", {})
    return {
        "predicted_n": tim.get("predicted_n"),
        "predicted_ms": tim.get("predicted_ms"),
        "decode_tps": tim.get("predicted_per_second"),
        "prompt_n": tim.get("prompt_n"),
        "prompt_tps": tim.get("prompt_per_second"),
        "draft_n": tim.get("draft_n"),
        "draft_accepted": tim.get("draft_n_accepted"),
        "wall_s": round(wall, 2),
    }


def main():
    rows = []
    for t in range(TRIALS):
        for i, p in enumerate(PROMPTS):
            r = one(p)
            r["trial"], r["prompt_idx"] = t, i
            rows.append(r)
            print(f"[{LABEL}] trial={t} prompt={i} "
                  f"decode={r['decode_tps']:.2f} tok/s  n={r['predicted_n']}  "
                  f"draft={r['draft_n']} acc={r['draft_accepted']}  wall={r['wall_s']}s",
                  flush=True)

    tps = [r["decode_tps"] for r in rows if r["decode_tps"]]
    tps.sort()
    n = len(tps)
    mean = sum(tps) / n
    print(f"\n=== {LABEL} summary over {n} requests ===")
    print(f"decode tok/s  mean={mean:.2f}  min={tps[0]:.2f}  max={tps[-1]:.2f}  "
          f"median={tps[n // 2]:.2f}  spread=+/-{(tps[-1] - tps[0]) / 2:.2f}")

    dn = sum(r["draft_n"] or 0 for r in rows)
    da = sum(r["draft_accepted"] or 0 for r in rows)
    if dn:
        print(f"draft acceptance {da}/{dn} = {100.0 * da / dn:.1f}%")
    else:
        print("draft acceptance: n/a (no speculation active)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"mtp_{LABEL}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
