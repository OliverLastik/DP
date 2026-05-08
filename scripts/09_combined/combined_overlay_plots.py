#!/usr/bin/env python3
"""
Kombinovane overlay ploty pre diplomovu pracu.
Spaja vysledky z roznych analyz do prehliadnych porovnavacich grafov.
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(".")
OUT = BASE / "combined_plots"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "de", "nl"]
LANG_COLORS = {"en": "#1f77b4", "de": "#ff7f0e", "nl": "#2ca02c"}
LANG_LABELS = {"en": "English", "de": "German", "nl": "Dutch"}


# =============================================================================
# 1. Heaps + Zipf exponents comparison (bar chart)
# =============================================================================
def plot_exponents_comparison():
    """Porovnanie Heaps beta a ZM alpha cez jazyky."""
    heaps = pd.read_csv(BASE / "rankfreq_out_core/heaps_law/heaps_law_summary.csv")
    zm = pd.read_csv(BASE / "rankfreq_out_core/zipf_mandelbrot/zipf_mandelbrot_fit_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Heaps beta
    ax = axes[0]
    for lang in LANGS:
        h_w = heaps[(heaps["lang"] == lang) & (heaps["mode"] == "words")]["beta"].values[0]
        h_p = heaps[(heaps["lang"] == lang) & (heaps["mode"] == "words+punct")]["beta"].values[0]
        x = LANGS.index(lang)
        ax.bar(x - 0.15, h_w, 0.3, color=LANG_COLORS[lang], alpha=0.6, label="words" if x == 0 else "")
        ax.bar(x + 0.15, h_p, 0.3, color=LANG_COLORS[lang], alpha=1.0, label="w+punct" if x == 0 else "")
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_ylabel("$\\beta$ (Heaps)")
    ax.set_title("Heaps exponent")
    ax.legend(fontsize=8)

    # ZM alpha
    ax = axes[1]
    for lang in LANGS:
        z_w = zm[(zm["language"] == lang) & (zm["mode"] == "words")]["alpha"].values[0]
        z_p = zm[(zm["language"] == lang) & (zm["mode"] == "words_plus_punct")]["alpha"].values[0]
        x = LANGS.index(lang)
        ax.bar(x - 0.15, z_w, 0.3, color=LANG_COLORS[lang], alpha=0.6)
        ax.bar(x + 0.15, z_p, 0.3, color=LANG_COLORS[lang], alpha=1.0)
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_ylabel("$\\alpha$ (Zipf-Mandelbrot)")
    ax.set_title("ZM exponent")

    # ZM c
    ax = axes[2]
    for lang in LANGS:
        c_w = zm[(zm["language"] == lang) & (zm["mode"] == "words")]["c"].values[0]
        c_p = zm[(zm["language"] == lang) & (zm["mode"] == "words_plus_punct")]["c"].values[0]
        x = LANGS.index(lang)
        ax.bar(x - 0.15, c_w, 0.3, color=LANG_COLORS[lang], alpha=0.6)
        ax.bar(x + 0.15, c_p, 0.3, color=LANG_COLORS[lang], alpha=1.0)
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_ylabel("c (ZM shift)")
    ax.set_title("ZM parameter c")

    fig.suptitle("Frequency law exponents: words vs words+punct", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "exponents_comparison.png", dpi=300)
    plt.close(fig)
    print("[OK] exponents_comparison.png")


# =============================================================================
# 2. Node-level: punct vs words (multi-language overlay)
# =============================================================================
def plot_node_level_overlay():
    """Punct vs words C_i a k_i cez vsetky jazyky v jednom grafe."""
    nl = pd.read_csv(BASE / "network_out_core/node_level/node_level_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # C_i
    ax = axes[0]
    x = np.arange(len(LANGS))
    w = 0.35
    ci_p = [nl[nl["lang"] == l]["ci_punct_mean"].values[0] for l in LANGS]
    ci_w = [nl[nl["lang"] == l]["ci_words_mean"].values[0] for l in LANGS]
    ax.bar(x - w/2, ci_p, w, color="red", alpha=0.6, label="punctuation")
    ax.bar(x + w/2, ci_w, w, color="steelblue", alpha=0.6, label="words")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_ylabel("mean $C_i$")
    ax.set_title("Local clustering coefficient")
    ax.legend(fontsize=8)

    # k_i
    ax = axes[1]
    ki_p = [nl[nl["lang"] == l]["ki_punct_mean"].values[0] for l in LANGS]
    ki_w = [nl[nl["lang"] == l]["ki_words_mean"].values[0] for l in LANGS]
    ax.bar(x - w/2, ki_p, w, color="red", alpha=0.6, label="punctuation")
    ax.bar(x + w/2, ki_w, w, color="steelblue", alpha=0.6, label="words")
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in LANGS])
    ax.set_ylabel("mean $k_i$")
    ax.set_title("Node degree")
    ax.legend(fontsize=8)

    fig.suptitle("Node-level: punctuation as high-degree low-clustering hubs", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "node_level_overlay.png", dpi=300)
    plt.close(fig)
    print("[OK] node_level_overlay.png")


# =============================================================================
# 3. MF-DFA h(q) overlay all languages
# =============================================================================
def plot_mfdfa_hq_overlay():
    """h(q) krivky cez vsetky jazyky v jednom grafe."""
    hq_path = BASE / "rankfreq_out_core/mfdfa/mfdfa_hq.csv"
    if not hq_path.exists():
        print("[SKIP] mfdfa_hq.csv not found")
        return

    hq = pd.read_csv(hq_path)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lang in LANGS:
        sub = hq[hq["lang"] == lang].sort_values("q")
        ax.plot(sub["q"], sub["h_q"], "o-", color=LANG_COLORS[lang],
                markersize=5, label=LANG_LABELS[lang])

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="h=0.5 (uncorrelated)")
    ax.set_xlabel("q")
    ax.set_ylabel("h(q)")
    ax.set_title("Generalized Hurst exponent (sentence lengths)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "mfdfa_hq_overlay.png", dpi=300)
    plt.close(fig)
    print("[OK] mfdfa_hq_overlay.png")


# =============================================================================
# 4. Recurrence: CV vs frequency — all langs in one
# =============================================================================
def plot_recurrence_cv_overlay():
    """CV (burstiness) scatter overlay cez jazyky."""
    rec = pd.read_csv(BASE / "rankfreq_out_core/recurrence_time/recurrence_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 5))

    for lang in LANGS:
        sub = rec[rec["lang"] == lang]
        punct = sub[sub["is_punct"] == 1]
        words = sub[sub["is_punct"] == 0]

        ax.scatter(words["count"], words["cv"], c=LANG_COLORS[lang], s=12, alpha=0.4,
                   label=f"{lang} words")
        ax.scatter(punct["count"], punct["cv"], c=LANG_COLORS[lang], s=60,
                   marker="*", edgecolors="k", linewidths=0.5, zorder=5,
                   label=f"{lang} punct")

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="CV=1 (Poisson)")
    ax.set_xscale("log")
    ax.set_xlabel("Token frequency")
    ax.set_ylabel("CV of recurrence time")
    ax.set_title("Burstiness: punctuation vs words across languages")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "recurrence_cv_overlay.png", dpi=300)
    plt.close(fig)
    print("[OK] recurrence_cv_overlay.png")


# =============================================================================
# 5. Network effect of punctuation: clustering delta
# =============================================================================
def plot_clustering_effect():
    """Efekt pridania interpunkcie na klasterizaciu — per-book boxplot overlay."""
    wp = pd.read_csv(BASE / "network_out_core/words_vs_punct/words_vs_punct_metrics.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    data = []
    labels = []
    colors = []
    for lang in LANGS:
        sub = wp[wp["lang"] == lang]
        data.append(sub["d_avg_clustering"].dropna().values)
        labels.append(lang.upper())
        colors.append(LANG_COLORS[lang])

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylabel("$\\Delta$ clustering (w+punct - words)")
    ax.set_title("Effect of adding punctuation on network clustering")
    fig.tight_layout()
    fig.savefig(OUT / "clustering_effect_boxplot.png", dpi=300)
    plt.close(fig)
    print("[OK] clustering_effect_boxplot.png")


# =============================================================================
# 6. Grand summary dashboard
# =============================================================================
def plot_dashboard():
    """Velky sumarny dashboard so vsetkymi klucovymi vysledkami."""
    fig = plt.figure(figsize=(14, 10))

    # Layout: 2x3 grid
    ax1 = fig.add_subplot(2, 3, 1)  # Heaps beta
    ax2 = fig.add_subplot(2, 3, 2)  # Node C_i vs k_i
    ax3 = fig.add_subplot(2, 3, 3)  # MF-DFA h(q)
    ax4 = fig.add_subplot(2, 3, 4)  # Clustering delta
    ax5 = fig.add_subplot(2, 3, 5)  # Recurrence CV
    ax6 = fig.add_subplot(2, 3, 6)  # ZM c

    # 1. Heaps beta
    heaps = pd.read_csv(BASE / "rankfreq_out_core/heaps_law/heaps_law_summary.csv")
    for lang in LANGS:
        h_w = heaps[(heaps["lang"] == lang) & (heaps["mode"] == "words")]["beta"].values[0]
        h_p = heaps[(heaps["lang"] == lang) & (heaps["mode"] == "words+punct")]["beta"].values[0]
        x = LANGS.index(lang)
        ax1.bar(x - 0.15, h_w, 0.3, color=LANG_COLORS[lang], alpha=0.5)
        ax1.bar(x + 0.15, h_p, 0.3, color=LANG_COLORS[lang], alpha=1.0)
    ax1.set_xticks(range(len(LANGS)))
    ax1.set_xticklabels([l.upper() for l in LANGS])
    ax1.set_ylabel("$\\beta$")
    ax1.set_title("(a) Heaps exponent", fontsize=10)

    # 2. Node-level
    nl = pd.read_csv(BASE / "network_out_core/node_level/node_level_summary.csv")
    x = np.arange(len(LANGS))
    ci_p = [nl[nl["lang"] == l]["ci_punct_mean"].values[0] for l in LANGS]
    ci_w = [nl[nl["lang"] == l]["ci_words_mean"].values[0] for l in LANGS]
    ax2.bar(x - 0.17, ci_p, 0.34, color="red", alpha=0.5, label="punct")
    ax2.bar(x + 0.17, ci_w, 0.34, color="steelblue", alpha=0.5, label="words")
    ax2.set_xticks(x); ax2.set_xticklabels([l.upper() for l in LANGS])
    ax2.set_ylabel("mean $C_i$"); ax2.set_title("(b) Local clustering", fontsize=10)
    ax2.legend(fontsize=7)

    # 3. MF-DFA h(q)
    hq_path = BASE / "rankfreq_out_core/mfdfa/mfdfa_hq.csv"
    if hq_path.exists():
        hq = pd.read_csv(hq_path)
        for lang in LANGS:
            sub = hq[hq["lang"] == lang].sort_values("q")
            ax3.plot(sub["q"], sub["h_q"], "o-", color=LANG_COLORS[lang],
                     markersize=3, label=lang.upper())
        ax3.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("q"); ax3.set_ylabel("h(q)")
    ax3.set_title("(c) MF-DFA Hurst exponent", fontsize=10)
    ax3.legend(fontsize=7)

    # 4. Clustering delta
    wp = pd.read_csv(BASE / "network_out_core/words_vs_punct/words_vs_punct_metrics.csv")
    d_data = [wp[wp["lang"] == l]["d_avg_clustering"].dropna().values for l in LANGS]
    bp = ax4.boxplot(d_data, tick_labels=[l.upper() for l in LANGS], patch_artist=True)
    for patch, l in zip(bp["boxes"], LANGS):
        patch.set_facecolor(LANG_COLORS[l]); patch.set_alpha(0.5)
    ax4.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax4.set_ylabel("$\\Delta C$"); ax4.set_title("(d) Clustering boost from punct", fontsize=10)

    # 5. Recurrence CV
    rec = pd.read_csv(BASE / "rankfreq_out_core/recurrence_time/recurrence_summary.csv")
    for lang in LANGS:
        sub = rec[rec["lang"] == lang]
        p = sub[sub["is_punct"] == 1]
        w = sub[sub["is_punct"] == 0]
        ax5.scatter(w["count"], w["cv"], c=LANG_COLORS[lang], s=8, alpha=0.4)
        ax5.scatter(p["count"], p["cv"], c=LANG_COLORS[lang], s=50,
                    marker="*", edgecolors="k", linewidths=0.3, zorder=5)
    ax5.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax5.set_xscale("log"); ax5.set_xlabel("frequency")
    ax5.set_ylabel("CV"); ax5.set_title("(e) Recurrence burstiness", fontsize=10)

    # 6. ZM c
    zm = pd.read_csv(BASE / "rankfreq_out_core/zipf_mandelbrot/zipf_mandelbrot_fit_summary.csv")
    for lang in LANGS:
        c_w = zm[(zm["language"] == lang) & (zm["mode"] == "words")]["c"].values[0]
        c_p = zm[(zm["language"] == lang) & (zm["mode"] == "words_plus_punct")]["c"].values[0]
        x = LANGS.index(lang)
        ax6.bar(x - 0.15, c_w, 0.3, color=LANG_COLORS[lang], alpha=0.5)
        ax6.bar(x + 0.15, c_p, 0.3, color=LANG_COLORS[lang], alpha=1.0)
    ax6.set_xticks(range(len(LANGS)))
    ax6.set_xticklabels([l.upper() for l in LANGS])
    ax6.set_ylabel("c"); ax6.set_title("(f) ZM shift parameter", fontsize=10)

    fig.suptitle("Statistical analysis of punctuation in narrative texts — Summary", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "dashboard_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] dashboard_summary.png")


if __name__ == "__main__":
    plot_exponents_comparison()
    plot_node_level_overlay()
    plot_mfdfa_hq_overlay()
    plot_recurrence_cv_overlay()
    plot_clustering_effect()
    plot_dashboard()
    print("\n[OK] Vsetky kombinovane ploty hotove.")
