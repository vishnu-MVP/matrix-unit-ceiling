"""Figures for the matrix-unit ceiling paper. All values are measured on a single
AWS r7i.4xlarge (Xeon 8488C Sapphire Rapids, 8 physical cores, 93.7 GB/s measured
read bandwidth) unless marked as a model.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

INK, ACC, WARN, MUT = "#12232e", "#2d7ff9", "#e8833a", "#8896a5"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  figs/%s.png" % name)


# --- Fig 1: optimisation ladder (pp512, prefill) ------------------------------
def fig_ladder():
    steps = ["AVX-512 VNNI\nrepack GEMM\n(baseline)", "AMX tile\nscalar unpack",
             "+ vectorised\nunpack", "+ L1-resident\nC accumulator",
             "+ direct AMX\nactivation load", "+ K=64 tiles\n(AMX max)"]
    pp = [17.40, 13.40, 37.74, 51.01, 55.28, 59.67]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    cols = [MUT] + [ACC] * 5
    bars = ax.bar(range(len(pp)), pp, color=cols, width=.62)
    bars[1].set_color(WARN)
    for i, v in enumerate(pp):
        ax.text(i, v + 1.1, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(pp)))
    ax.set_xticklabels(steps, fontsize=7.4)
    ax.set_ylabel("prefill throughput  (tok/s, pp512)")
    ax.set_ylim(0, 70)
    ax.axhline(17.40, ls=":", lw=1, color=MUT)
    ax.annotate("3.43x", xy=(5, 59.67), xytext=(4.0, 66),
                fontsize=10, fontweight="bold", color=ACC)
    ax.set_title("Prefill optimisation ladder: a naive AMX port is slower than good VNNI",
                 fontsize=9, loc="left")
    save(fig, "fig1_ladder")


# --- Fig 2: bottleneck migration ---------------------------------------------
def fig_migration():
    # both distributions are measured: perf -F 999 on one pp512 run of each build
    labels = ["matmul kernel", "fp32 sequence mixing\n(GDN / FWHT / ssm_conv)",
              "threading (libgomp)", "other"]
    before = [87.8, 5.3, 3.4, 3.5]          # baseline repack GEMM, 21.31 tok/s
    after = [56.6, 17.9, 13.3, 12.2]        # our AMX tile kernel, 74.60 tok/s
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    cols = [ACC, WARN, "#b45fd6", MUT]
    for row, vals in enumerate([after, before]):
        left = 0
        for v, c in zip(vals, cols):
            ax.barh(row, v, left=left, color=c, height=.55)
            if v > 6:
                ax.text(left + v / 2, row, f"{v:.1f}%", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
            left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ours\n74.60 tok/s", "baseline\n21.31 tok/s"], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of prefill runtime (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, labels, fontsize=7.2, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.42), frameon=False)
    ax.set_title("Optimising the matmul migrates the bottleneck out of the matmul",
                 fontsize=9, loc="left")
    save(fig, "fig2_migration")


# --- Fig 3: the ceiling (Amdahl over the non-GEMM residue) --------------------
def fig_ceiling():
    # basis = the profiled prefill run: 512 tok at 74.60 tok/s, matmul share 56.61%
    p = 0.5661
    base = 74.60
    s = np.logspace(0, 2.4, 300)   # further matmul speedup
    tput = base / ((1 - p) + p / s)
    ceiling = base / (1 - p)
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.fill_between(s, tput, ceiling, color=ACC, alpha=0.07, lw=0)
    ax.plot(s, tput, color=ACC, lw=2, zorder=4)

    ax.axhline(ceiling, ls="--", lw=1.4, color=WARN, zorder=3)
    ax.text(0.95, ceiling - 14, f"ceiling {ceiling:.0f} tok/s  (matmul $\\to$ 0)",
            fontsize=8, color=WARN, fontweight="bold", ha="left")

    ax.scatter([1], [base], s=44, color=INK, zorder=6)
    ax.annotate(f"measured  {base:.1f}", xy=(1, base), xytext=(3.2, 40),                fontsize=8, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.8, color=INK,
                                shrinkA=0, shrinkB=3))

    # the K=64 experiment: 2x dpbssd throughput bought only +7.9%
    ax.scatter([2], [base * 1.079], s=44, color=WARN, zorder=6, marker="D")
    ax.annotate("K=64 tiles: $2\\times$ matmul rate,\nonly $+7.9\\%$ end-to-end",
                xy=(2, base * 1.079), xytext=(11, 98),
                fontsize=8, color=WARN, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.8, color=WARN,
                                shrinkA=0, shrinkB=4))

    ax.set_xscale("log")
    ax.set_xlim(0.85, 260)
    ax.set_ylim(0, 200)
    ax.set_xlabel("hypothetical further matmul speedup ($\\times$)")
    ax.set_ylabel("prefill throughput (tok/s)")
    ax.set_title("Matrix units cannot pass the non-GEMM residue",
                 fontsize=9, loc="left")
    save(fig, "fig3_ceiling")


# --- Fig 5: decode is kernel-invariant once bandwidth is saturated ------------
def fig_invariant():
    ks = ["AVX2 repack\nGEMM", "AMX tile\n(ours)", "AMX tile\n+ MTP"]
    tg = [6.63, 6.58, 9.92]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.bar(range(3), tg, color=[MUT, ACC, "#16a34a"], width=.55)
    for i, v in enumerate(tg):
        ax.text(i, v + .3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(13.03, ls="--", color=WARN, lw=1.4)
    ax.text(2.42, 13.35, "bandwidth roofline 13.03", fontsize=7.6, ha="right",
            color=WARN, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(ks, fontsize=7.6)
    ax.set_ylabel("decode (tok/s)"); ax.set_ylim(0, 16.2)
    # a 3.47x prefill spread moves decode by 1%; batching it via MTP is what pays
    ax.set_title("Decode is indifferent to matmul speed until MTP adds batch", fontsize=9, loc="left")
    save(fig, "fig5_decode_invariant")


def fig_threads():
    t = [2, 4, 8, 16]
    pre = [22.91, 44.33, 80.22, 75.52]
    dec = [2.06, 3.82, 6.44, 6.55]
    ideal = [pre[0] * (x / 2) for x in t]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.3))
    a1.plot(t, ideal, ls="--", color=MUT, lw=1.3, label="linear")
    a1.plot(t, pre, color=ACC, marker="o", lw=1.8, ms=4.5, label="measured")
    a1.axvspan(8, 16, color=WARN, alpha=.14)
    a1.text(11.3, 38, "SMT", fontsize=8, ha="center", color=WARN, fontweight="bold")
    a1.set_xscale("log", base=2); a1.set_xticks(t); a1.set_xticklabels(t)
    a1.set_xlabel("threads"); a1.set_ylabel("prefill (tok/s)")
    a1.set_title("Prefill: scales to 8 physical cores", fontsize=8.5, loc="left")
    a1.legend(fontsize=8, frameon=False, loc="upper left")

    a2.plot(t, dec, color=ACC, marker="o", lw=1.8, ms=4.5)
    a2.axvspan(8, 16, color=WARN, alpha=.14)
    a2.text(11.3, 2.4, "SMT", fontsize=8, ha="center", color=WARN, fontweight="bold")
    a2.set_xscale("log", base=2); a2.set_xticks(t); a2.set_xticklabels(t)
    a2.set_ylim(0, 7.6)
    a2.set_xlabel("threads"); a2.set_ylabel("decode (tok/s)")
    a2.set_title("Decode: saturates early", fontsize=8.5, loc="left")
    save(fig, "fig6_thread_scaling")


if __name__ == "__main__":
    print("writing figures ->", OUT)
    fig_ladder(); fig_migration(); fig_ceiling(); fig_invariant()
    fig_threads()
    print("done")
