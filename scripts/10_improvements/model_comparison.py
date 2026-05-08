#!/usr/bin/env python3
"""
Kompletne porovnanie sietovych modelov:
  - Real words-only
  - Real words+punct
  - BA model
  - DM model
  - Shuffled null model
  - Markov synthetic

Vystup: porovnavacie bar charty, tabulkove CSV, radar plot.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


LANGS = ["en", "de", "nl"]
LANG_LABELS = {"en": "English", "de": "German", "nl": "Dutch"}


def load_csv_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="improved_plots/model_comparison")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all data sources ──
    # Words vs words+punct per-book
    wp = load_csv_safe(Path("network_out_core/words_vs_punct/words_vs_punct_metrics.csv"))

    # BA summary
    ba_summary = {}
    ba_df = load_csv_safe(Path("ba_out_core/metrics_ba_summary.csv"))
    if not ba_df.empty:
        for _, row in ba_df.iterrows():
            ba_summary[row["language"]] = row

    # DM summary
    dm_summary = {}
    dm_df = load_csv_safe(Path("dm_out_core/metrics_dm_summary.csv"))
    if not dm_df.empty:
        for _, row in dm_df.iterrows():
            dm_summary[row["language"]] = row

    # Shuffled null
    shuf_df = load_csv_safe(Path("network_out_core/shuffled_null/shuffled_summary.csv"))

    # Markov synthetic
    markov_df = load_csv_safe(Path("markov_synth_out_core/markov_synth_summary.csv"))

    # Real network summary
    real_summary = {}
    real_df = load_csv_safe(Path("network_out_core/metrics_summary_by_language.csv"))
    if not real_df.empty:
        for _, row in real_df.iterrows():
            real_summary[row["language"]] = row

    # Words+punct summary (from per-book medians)
    wp_summary = {}
    wp_sum_df = load_csv_safe(Path("network_out_core/words_vs_punct/words_vs_punct_summary.csv"))
    if not wp_sum_df.empty:
        for _, row in wp_sum_df.iterrows():
            wp_summary[row["lang"]] = row

    # ── 1. Comprehensive bar chart: clustering ──
    print("=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    metrics_config = [
        {
            "metric": "avg_clustering",
            "title": "Average clustering coefficient",
            "ylabel": "$\\langle C \\rangle$",
        },
        {
            "metric": "avg_degree",
            "title": "Average degree",
            "ylabel": "$\\langle k \\rangle$",
        },
    ]

    for mc in metrics_config:
        metric = mc["metric"]
        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(LANGS))
        width = 0.13
        models = []

        # Collect data per model
        # 1. Words-only (real)
        vals = []
        for lang in LANGS:
            if not wp_sum_df.empty and lang in wp_summary:
                vals.append(float(wp_summary[lang].get(f"w_{metric}_med", 0)))
            elif lang in real_summary:
                vals.append(float(real_summary[lang].get(metric, 0)))
            else:
                vals.append(0)
        models.append(("Words only", vals, "#1f77b4", 0.7))

        # 2. Words+punct (real)
        vals = []
        for lang in LANGS:
            if not wp_sum_df.empty and lang in wp_summary:
                vals.append(float(wp_summary[lang].get(f"wp_{metric}_med", 0)))
            else:
                vals.append(0)
        models.append(("Words+punct", vals, "#ff7f0e", 0.7))

        # 3. BA model
        vals = []
        for lang in LANGS:
            if lang in ba_summary:
                v = ba_summary[lang].get(f"{metric}_mean", 0)
                vals.append(float(v) if v else 0)
            else:
                vals.append(0)
        models.append(("BA model", vals, "#d62728", 0.7))

        # 4. DM model
        vals = []
        for lang in LANGS:
            if lang in dm_summary:
                v = dm_summary[lang].get(f"{metric}_mean", 0)
                vals.append(float(v) if v else 0)
            else:
                vals.append(0)
        models.append(("DM model", vals, "#2ca02c", 0.7))

        # 5. Shuffled null
        vals = []
        for lang in LANGS:
            if not shuf_df.empty:
                row = shuf_df[shuf_df["lang"] == lang]
                if not row.empty:
                    vals.append(float(row[f"shuf_{metric}_med"].values[0]))
                else:
                    vals.append(0)
            else:
                vals.append(0)
        models.append(("Shuffled", vals, "#9467bd", 0.7))

        # 6. Markov synthetic
        vals = []
        for lang in LANGS:
            if not markov_df.empty:
                row = markov_df[(markov_df["language"] == lang) & (markov_df["kind"] == "markov")]
                if not row.empty:
                    vals.append(float(row[f"{metric}_median"].values[0]))
                else:
                    vals.append(0)
            else:
                vals.append(0)
        models.append(("Markov synth", vals, "#8c564b", 0.7))

        # Plot bars
        n_models = len(models)
        offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)

        for i, (label, vals, color, alpha) in enumerate(models):
            if all(v == 0 for v in vals):
                continue
            bars = ax.bar(x + offsets[i], vals, width, label=label,
                          color=color, alpha=alpha, edgecolor="white", linewidth=0.5)
            # value labels
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in LANGS], fontsize=12)
        ax.set_ylabel(mc["ylabel"], fontsize=12)
        ax.set_title(mc["title"], fontsize=14)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_{metric}.png", dpi=300)
        plt.close(fig)
        print(f"[OK] compare_{metric}.png")

    # ── 2. Degree distribution overlay: Real vs BA vs DM ──
    # Already handled in improved_logbin_fit.py

    # ── 3. Summary table CSV ──
    table_rows = []
    for lang in LANGS:
        row = {"lang": lang}

        # Words-only
        if lang in wp_summary:
            row["w_clustering"] = float(wp_summary[lang].get("w_avg_clustering_med", 0))
            row["w_degree"] = float(wp_summary[lang].get("w_avg_degree_med", 0))
            row["w_density"] = float(wp_summary[lang].get("w_density_med", 0))
        elif lang in real_summary:
            row["w_clustering"] = float(real_summary[lang].get("avg_clustering", 0))
            row["w_degree"] = float(real_summary[lang].get("avg_degree", 0))

        # Words+punct
        if lang in wp_summary:
            row["wp_clustering"] = float(wp_summary[lang].get("wp_avg_clustering_med", 0))
            row["wp_degree"] = float(wp_summary[lang].get("wp_avg_degree_med", 0))
            row["wp_density"] = float(wp_summary[lang].get("wp_density_med", 0))

        # BA
        if lang in ba_summary:
            row["ba_clustering"] = float(ba_summary[lang].get("avg_clustering_mean", 0) or 0)
            row["ba_degree"] = float(ba_summary[lang].get("avg_degree_mean", 0) or 0)

        # DM
        if lang in dm_summary:
            row["dm_clustering"] = float(dm_summary[lang].get("avg_clustering_mean", 0) or 0)
            row["dm_degree"] = float(dm_summary[lang].get("avg_degree_mean", 0) or 0)

        # Shuffled
        if not shuf_df.empty:
            srow = shuf_df[shuf_df["lang"] == lang]
            if not srow.empty:
                row["shuf_clustering"] = float(srow["shuf_avg_clustering_med"].values[0])
                row["shuf_degree"] = float(srow["shuf_avg_degree_med"].values[0])

        # Markov
        if not markov_df.empty:
            mrow = markov_df[(markov_df["language"] == lang) & (markov_df["kind"] == "markov")]
            if not mrow.empty:
                row["markov_clustering"] = float(mrow["avg_clustering_median"].values[0])
                row["markov_degree"] = float(mrow["avg_degree_median"].values[0])

        table_rows.append(row)

    if table_rows:
        df_table = pd.DataFrame(table_rows)
        df_table.to_csv(out_dir / "model_comparison_table.csv", index=False)
        print(f"[OK] model_comparison_table.csv")

    # ── 4. Delta from real (words+punct) bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (metric, ylabel) in enumerate([
        ("clustering", "$\\Delta C$ from real w+punct"),
        ("degree", "$\\Delta k$ from real w+punct")
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(LANGS))
        width = 0.15

        ref_vals = []
        for lang in LANGS:
            if lang in wp_summary:
                ref_vals.append(float(wp_summary[lang].get(f"wp_avg_{metric}_med", 0)))
            else:
                ref_vals.append(0)

        compared = [
            ("Words only", "w", "#1f77b4"),
            ("BA", "ba", "#d62728"),
            ("DM", "dm", "#2ca02c"),
            ("Shuffled", "shuf", "#9467bd"),
            ("Markov", "markov", "#8c564b"),
        ]

        offsets = np.linspace(-2 * width, 2 * width, len(compared))

        for i, (label, key, color) in enumerate(compared):
            deltas = []
            for j, lang in enumerate(LANGS):
                row = table_rows[j]
                val = row.get(f"{key}_{metric}", 0)
                deltas.append(val - ref_vals[j] if val else 0)

            ax.bar(x + offsets[i], deltas, width, label=label,
                   color=color, alpha=0.7)

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in LANGS])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Deviation in {metric}", fontsize=12)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Model deviations from real words+punct network", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "model_deltas.png", dpi=300)
    plt.close(fig)
    print("[OK] model_deltas.png")

    print("\n[OK] Model comparison hotovo.")


if __name__ == "__main__":
    main()
