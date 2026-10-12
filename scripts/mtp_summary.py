#!/usr/bin/env python3
"""Summarise mtp_bench.py result files into one comparison table."""
import json
import os

LABELS = ["nomtp", "withmtp", "stockmtp"]
RESULTS_DIR = os.environ.get("OUTPUT_DIR", ".")
TITLE = {
    "nomtp": "ours kernel, no drafting",
    "withmtp": "ours kernel + MTP",
    "stockmtp": "stock kernel + MTP",
}

rows = []
for lab in LABELS:
    p = os.path.join(RESULTS_DIR, "mtp_%s.json" % lab)
    if not os.path.exists(p):
        print("missing:", p)
        continue
    r = json.load(open(p))
    t = sorted(x["decode_tps"] for x in r if x.get("decode_tps"))
    if not t:
        continue
    n = len(t)
    dn = sum(x.get("draft_n") or 0 for x in r)
    da = sum(x.get("draft_accepted") or 0 for x in r)
    acc = (100.0 * da / dn) if dn else None
    rows.append((lab, n, sum(t) / n, t[0], t[-1], t[n // 2], acc, da, dn))

print("%-26s %3s %7s %7s %7s %7s  %s" %
      ("config", "n", "mean", "min", "max", "median", "acceptance"))
for lab, n, mean, lo, hi, med, acc, da, dn in rows:
    a = ("%.1f%% (%d/%d)" % (acc, da, dn)) if acc is not None else "-"
    print("%-26s %3d %7.2f %7.2f %7.2f %7.2f  %s" %
          (TITLE[lab], n, mean, lo, hi, med, a))

d = {l: m for l, _, m, *_ in [(r[0], r[1], r[2]) + tuple(r[3:]) for r in rows]}
if "withmtp" in d and "nomtp" in d:
    print("\nMTP over plain decode      : %.2fx" % (d["withmtp"] / d["nomtp"]))
if "withmtp" in d and "stockmtp" in d:
    print("our kernel's gain under MTP: %.2fx (%+.1f%%)" %
          (d["withmtp"] / d["stockmtp"], 100 * (d["withmtp"] / d["stockmtp"] - 1)))
