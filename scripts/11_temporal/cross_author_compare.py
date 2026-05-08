#!/usr/bin/env python3
"""
Cross-author temporal comparison.

Nacita temporal_summary.csv z kazdej podzlozky v temporal_out/<lang_author>/
a postavi porovnavacie grafy naprieč autormi:
  - Weibull alpha / beta trajektorie (x=rok, y=parameter)
  - IPI mean / punct % trajektorie
  - Network clustering trajektorie
  - JSD medzi obdobiami v stlpci per autor
  - Overlay IPI histogramov + Weibull fitov
  - Spojena CSV tabulka
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

PUNCT_SET = {".", ",", "!", "?", ";", ":"}

AUTHOR_COLORS = {
    "dickens":  "#1f77b4",
    "fontane":  "#d62728",
    "couperus": "#2ca02c",
}
AUTHOR_MARKERS = {
    "dickens":  "o",
    "fontane":  "s",
    "couperus": "D",
}
PERIOD_ORDER = {"early": 0, "middle": 1, "late": 2}


def load_tokens(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="replace").split()


def compute_ipi(tokens):
    intervals = []
    w = 0
    seen = False
    for t in tokens:
        if t in PUNCT_SET:
            if seen:
                intervals.append(w)
            seen = True
            w = 0
        else:
            w += 1
    return np.array(intervals)


def weibull_discrete_pmf(k, alpha, beta):
    k = np.asarray(k, dtype=float)
    return np.exp(-(k / beta) ** alpha) - np.exp(-((k + 1) / beta) ** alpha)


def load_summary(subdir: Path) -> list:
    csv_path = subdir / "temporal_summary.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal-out", default="temporal_out")
    ap.add_argument("--token-dir", default="temporal_data/tokens")
    ap.add_argument("--out-dir", default="temporal_out/_cross")
    args = ap.parse_args()

    root = Path(args.temporal_out)
    tok_root = Path(args.token_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect per-author summaries
    per_author = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        rows = load_summary(sub)
        if not rows:
            continue
        # sort by year
        rows.sort(key=lambda r: int(r["year"]))
        author_key = rows[0]["author"]
        per_author[author_key] = {"lang": rows[0]["lang"], "rows": rows, "subdir": sub}

    if not per_author:
        print(f"[ERR] No per-author summaries found under {root}")
        return

    authors = sorted(per_author.keys(), key=lambda a: per_author[a]["lang"])
    print(f"[INFO] Found {len(authors)} authors: {authors}")

    # ══ Combined CSV ══
    all_rows = []
    for a in authors:
        for r in per_author[a]["rows"]:
            all_rows.append(r)
    combined_csv = out / "cross_author_summary.csv"
    with combined_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"[OK] {combined_csv}")

    # ══ Plot 1: Weibull alpha / beta trajektorie ══
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for a in authors:
        info = per_author[a]
        years = [int(r["year"]) for r in info["rows"]]
        alphas = [float(r["weibull_disc_alpha"]) for r in info["rows"]]
        betas = [float(r["weibull_disc_beta"]) for r in info["rows"]]
        color = AUTHOR_COLORS.get(a, "gray")
        marker = AUTHOR_MARKERS.get(a, "o")
        label = f"{a.capitalize()} ({info['lang']})"
        axes[0].plot(years, alphas, "-", color=color, marker=marker, markersize=9,
                     linewidth=2, label=label)
        axes[1].plot(years, betas, "-", color=color, marker=marker, markersize=9,
                     linewidth=2, label=label)

    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Year", fontsize=11)
    axes[0].set_ylabel(r"Weibull shape $\alpha$", fontsize=11)
    axes[0].set_title(r"Discrete Weibull $\alpha$ drift", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Year", fontsize=11)
    axes[1].set_ylabel(r"Weibull scale $\beta$", fontsize=11)
    axes[1].set_title(r"Discrete Weibull $\beta$ drift", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Inter-punctuation Weibull parameters across authors' careers",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "cross_weibull_drift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] cross_weibull_drift.png")

    # ══ Plot 2: IPI mean + punct % + clustering trajektorie ══
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metric_defs = [
        ("ipi_mean", "Mean IPI (words)", "Sentence-fragment length"),
        (None, "Punct density (%)", "Punctuation density"),
        ("avg_clustering", r"$\langle C \rangle$", "Network clustering"),
    ]
    for a in authors:
        info = per_author[a]
        years = [int(r["year"]) for r in info["rows"]]
        color = AUTHOR_COLORS.get(a, "gray")
        marker = AUTHOR_MARKERS.get(a, "o")
        label = f"{a.capitalize()} ({info['lang']})"

        ipi_means = [float(r["ipi_mean"]) for r in info["rows"]]
        axes[0].plot(years, ipi_means, "-", color=color, marker=marker, markersize=9,
                     linewidth=2, label=label)

        punct_pct = []
        for r in info["rows"]:
            tot = sum(float(r[k]) for k in r if k.startswith("pct_"))
            punct_pct.append(tot)
        axes[1].plot(years, punct_pct, "-", color=color, marker=marker, markersize=9,
                     linewidth=2, label=label)

        clust = [float(r["avg_clustering"]) for r in info["rows"]]
        axes[2].plot(years, clust, "-", color=color, marker=marker, markersize=9,
                     linewidth=2, label=label)

    for ax, (_, ylabel, title) in zip(axes, metric_defs):
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Temporal drift per author (3 metrics)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "cross_metric_drift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] cross_metric_drift.png")

    # ══ Plot 3: ZM alpha drift ══
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for a in authors:
        info = per_author[a]
        years = [int(r["year"]) for r in info["rows"]]
        zm = [float(r["zm_alpha"]) for r in info["rows"]]
        color = AUTHOR_COLORS.get(a, "gray")
        marker = AUTHOR_MARKERS.get(a, "o")
        ax.plot(years, zm, "-", color=color, marker=marker, markersize=9, linewidth=2,
                label=f"{a.capitalize()} ({info['lang']})")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(r"Zipf-Mandelbrot $\alpha$", fontsize=11)
    ax.set_title("Rank-frequency exponent across careers", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cross_zm_drift.png", dpi=300)
    plt.close(fig)
    print("[OK] cross_zm_drift.png")

    # ══ Plot 4: IPI distributions overlay with fits ══
    # For this we need the raw tokens → recompute IPI
    fig, axes = plt.subplots(1, len(authors), figsize=(5 * len(authors), 4.5),
                             sharey=True)
    if len(authors) == 1:
        axes = [axes]

    period_colors = {"early": "#1f77b4", "middle": "#ff7f0e", "late": "#2ca02c"}

    for ax, a in zip(axes, authors):
        info = per_author[a]
        tok_sub = tok_root / f"{info['lang']}_{a}"
        for r in info["rows"]:
            period = r["period"]
            year = r["year"]
            title = r["title"]
            fname = f"{info['lang']}_{a}_{period}_{year}_{title}.txt"
            fp = tok_sub / fname
            if not fp.exists():
                matches = list(tok_sub.glob(f"*_{period}_{year}_*.txt"))
                if matches:
                    fp = matches[0]
                else:
                    continue
            toks = load_tokens(fp)
            ipi = compute_ipi(toks)
            if len(ipi) == 0:
                continue
            bins = np.arange(0, min(50, int(ipi.max())) + 1) - 0.5
            counts, edges = np.histogram(ipi, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            color = period_colors[period]
            ax.plot(centers, counts, "o", color=color, markersize=3, alpha=0.6)
            alpha_w = float(r["weibull_disc_alpha"])
            beta_w = float(r["weibull_disc_beta"])
            if not np.isnan(alpha_w):
                k_grid = np.arange(0, int(centers.max()) + 1)
                pmf = weibull_discrete_pmf(k_grid, alpha_w, beta_w)
                ax.plot(k_grid, pmf, "-", color=color, linewidth=1.8,
                        label=f"{period} {year} ($\\alpha$={alpha_w:.2f}, $\\beta$={beta_w:.2f})")
        ax.set_xlabel("IPI $k$ (words)", fontsize=11)
        ax.set_title(f"{a.capitalize()} ({info['lang']})", fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$P(k)$", fontsize=11)
    fig.suptitle("Inter-punctuation intervals per author across periods",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "cross_ipi_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] cross_ipi_overlay.png")

    # ══ Plot 5: JSD range bar (max drift per author) ══
    # Recompute internal JSD using per-period rows and punct percentages as proxy distribution.
    # (We don't persist markov matrices in CSV, so use IPI distributions instead.)
    jsd_summary = []
    for a in authors:
        info = per_author[a]
        tok_sub = tok_root / f"{info['lang']}_{a}"
        dists = {}
        for r in info["rows"]:
            fname = f"{info['lang']}_{a}_{r['period']}_{r['year']}_{r['title']}.txt"
            fp = tok_sub / fname
            if not fp.exists():
                matches = list(tok_sub.glob(f"*_{r['period']}_{r['year']}_*.txt"))
                if matches:
                    fp = matches[0]
                else:
                    continue
            toks = load_tokens(fp)
            ipi = compute_ipi(toks)
            bins = np.arange(0, 50)
            h, _ = np.histogram(ipi, bins=bins, density=False)
            p = h.astype(float) / h.sum() if h.sum() > 0 else h.astype(float)
            dists[r["period"]] = p

        def jsd(p, q):
            return float(jensenshannon(p, q) ** 2)

        if {"early", "middle", "late"}.issubset(dists):
            em = jsd(dists["early"], dists["middle"])
            ml = jsd(dists["middle"], dists["late"])
            el = jsd(dists["early"], dists["late"])
            jsd_summary.append({"author": a, "lang": info["lang"],
                                "early_vs_middle": em,
                                "middle_vs_late": ml,
                                "early_vs_late": el})
            print(f"  {a:9s}  E-M={em:.4f}  M-L={ml:.4f}  E-L={el:.4f}")

    if jsd_summary:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(jsd_summary))
        width = 0.28
        em_vals = [d["early_vs_middle"] for d in jsd_summary]
        ml_vals = [d["middle_vs_late"] for d in jsd_summary]
        el_vals = [d["early_vs_late"] for d in jsd_summary]
        ax.bar(x - width, em_vals, width, label="Early ↔ Middle", color="#6baed6")
        ax.bar(x,         ml_vals, width, label="Middle ↔ Late",  color="#fd8d3c")
        ax.bar(x + width, el_vals, width, label="Early ↔ Late",   color="#74c476")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d['author'].capitalize()}\n({d['lang']})"
                            for d in jsd_summary], fontsize=10)
        ax.set_ylabel("JS divergence of IPI distribution", fontsize=11)
        ax.set_title("How much the inter-punctuation distribution drifts over a career",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "cross_jsd_ipi.png", dpi=300)
        plt.close(fig)
        print("[OK] cross_jsd_ipi.png")

        jsd_csv = out / "cross_jsd_ipi.csv"
        with jsd_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(jsd_summary[0].keys()))
            w.writeheader()
            w.writerows(jsd_summary)
        print(f"[OK] {jsd_csv}")

    print("\n[OK] Cross-author porovnanie hotove.")


if __name__ == "__main__":
    main()
