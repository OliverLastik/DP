#!/usr/bin/env python3
"""
Master plot: 3 autori x 3 panely.

Riadky: Dickens (en), Fontane (de), Couperus (nl)
Stlpce:
  1) Weibull alpha drift s 95% bootstrap CI
  2) Weibull beta drift s 95% bootstrap CI
  3) Delta-AIC stacked bars pre 4 modely (Weibull / NegBinom / Lognormal / Geometric)

V kazdom riadku je navyse textova anotacia so signal/noise ratio
(inter-period JSD / intra-book JSD) z _baseline.

Vstupy:
  temporal_out/_bootstrap/weibull_bootstrap.csv
  temporal_out/_aic/model_comparison.csv
  temporal_out/_baseline/signal_noise_ratio.csv

Vystup:
  temporal_out/temporal_master.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUTHORS = [
    ("en", "dickens",  "Dickens",  "#1f77b4", "o"),
    ("de", "fontane",  "Fontane",  "#d62728", "s"),
    ("nl", "couperus", "Couperus", "#2ca02c", "D"),
]

MODEL_COLORS = {
    "weibull":   "#4c72b0",
    "nbinom":    "#dd8452",
    "lognormal": "#55a868",
    "geometric": "#8172b3",
}
MODEL_ORDER = ["weibull", "nbinom", "lognormal", "geometric"]
PERIOD_ORDER = ["early", "middle", "late"]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="temporal_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    boot_rows = read_csv(out_dir / "_bootstrap" / "weibull_bootstrap.csv")
    aic_rows  = read_csv(out_dir / "_aic" / "model_comparison.csv")
    sn_rows   = read_csv(out_dir / "_baseline" / "signal_noise_ratio.csv")

    boot_by = defaultdict(list)
    for r in boot_rows:
        boot_by[(r["lang"], r["author"])].append(r)
    for k in boot_by:
        boot_by[k].sort(key=lambda r: int(r["year"]))

    aic_by = defaultdict(dict)
    for r in aic_rows:
        aic_by[(r["lang"], r["author"])][r["period"]] = r

    sn_by = {(r["lang"], r["author"]): r for r in sn_rows}

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.5))

    for row_idx, (lang, author, label, color, marker) in enumerate(AUTHORS):
        key = (lang, author)
        rows = boot_by[key]
        years    = np.array([int(r["year"]) for r in rows])
        alpha_p  = np.array([float(r["alpha_point"]) for r in rows])
        alpha_lo = np.array([float(r["alpha_lo"]) for r in rows])
        alpha_hi = np.array([float(r["alpha_hi"]) for r in rows])
        beta_p   = np.array([float(r["beta_point"]) for r in rows])
        beta_lo  = np.array([float(r["beta_lo"]) for r in rows])
        beta_hi  = np.array([float(r["beta_hi"]) for r in rows])
        titles   = [r["title"].replace("_", " ") for r in rows]

        # ---- Col 1: alpha drift ----
        ax = axes[row_idx, 0]
        ax.errorbar(years, alpha_p,
                    yerr=[alpha_p - alpha_lo, alpha_hi - alpha_p],
                    fmt=f"-{marker}", color=color, markersize=10,
                    linewidth=2.2, capsize=5)
        for x, y, t in zip(years, alpha_p, titles):
            ax.annotate(t, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7.5, color="#333")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_ylabel(f"{label}\n" + r"Weibull $\alpha$", fontsize=11,
                      fontweight="bold")
        ax.grid(alpha=0.3)
        if row_idx == 0:
            ax.set_title(r"Shape $\alpha$ drift (95% CI)", fontsize=12)
        if row_idx == 2:
            ax.set_xlabel("Year", fontsize=11)

        # ---- Col 2: beta drift ----
        ax = axes[row_idx, 1]
        ax.errorbar(years, beta_p,
                    yerr=[beta_p - beta_lo, beta_hi - beta_p],
                    fmt=f"-{marker}", color=color, markersize=10,
                    linewidth=2.2, capsize=5)
        for x, y, t in zip(years, beta_p, titles):
            ax.annotate(t, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7.5, color="#333")
        ax.grid(alpha=0.3)
        if row_idx == 0:
            ax.set_title(r"Scale $\beta$ drift (95% CI)", fontsize=12)
        if row_idx == 2:
            ax.set_xlabel("Year", fontsize=11)

        # ---- Col 3: delta-AIC bars ----
        ax = axes[row_idx, 2]
        periods_present = [p for p in PERIOD_ORDER if p in aic_by[key]]
        x_pos = np.arange(len(periods_present))
        width = 0.2

        winner_per_period = []
        for pi, period in enumerate(periods_present):
            r = aic_by[key][period]
            winner_per_period.append(r["best_model"])
            deltas = [float(r[f"{m}_delta_aic"]) for m in MODEL_ORDER]
            for mi, (m, d) in enumerate(zip(MODEL_ORDER, deltas)):
                offset = (mi - 1.5) * width
                ax.bar(pi + offset, max(d, 0.01), width,
                       color=MODEL_COLORS[m],
                       label=m if (row_idx == 0 and pi == 0) else None,
                       edgecolor="black", linewidth=0.3)

        # star above winning bar
        for pi, winner in enumerate(winner_per_period):
            mi = MODEL_ORDER.index(winner)
            offset = (mi - 1.5) * width
            ax.plot(pi + offset, 0.5, "k*", markersize=11,
                    markerfacecolor="gold", markeredgecolor="black")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.capitalize() for p in periods_present], fontsize=10)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\Delta$AIC (log)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if row_idx == 0:
            ax.set_title("Model comparison ($\\Delta$AIC, $\\star$=winner)",
                         fontsize=12)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

        # ---- signal/noise annotation ----
        sn = sn_by.get(key)
        if sn:
            ratio = float(sn["ratio_inter_over_intra"])
            verdict = "real drift" if ratio >= 2.0 else "within noise"
            txt = (f"inter/intra JSD = {ratio:.2f}×\n→ {verdict}")
            ax_anno = axes[row_idx, 0]
            ax_anno.text(0.02, 0.97, txt,
                         transform=ax_anno.transAxes,
                         fontsize=8.5, va="top", ha="left",
                         bbox=dict(boxstyle="round,pad=0.35",
                                   facecolor="white",
                                   edgecolor=color, linewidth=1.2,
                                   alpha=0.92))

    fig.suptitle("Temporal drift of inter-punctuation-interval distribution\n"
                 "Dickens (en) · Fontane (de) · Couperus (nl)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    out_path = out_dir / "temporal_master.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
