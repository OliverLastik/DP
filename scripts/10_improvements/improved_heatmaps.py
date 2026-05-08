#!/usr/bin/env python3
"""
Vylepsene heatmapy pre Markov transition matice.

Klucove vylepsenia:
  1. Merged cross-language heatmap: jeden graf, bunky obsahuju hodnoty
     zo vsetkych jazykov s farebnym oznacenim
  2. Filter < 0.2: vypustit male hodnoty
  3. Priečny rez: porovnanie konkretnych prechodov medzi jazykmi
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
import matplotlib.patches as mpatches


CORE_PUNCT = [".", ",", ";", ":", "?", "!"]
LANG_COLORS = {"en": "#1f77b4", "de": "#ff7f0e", "nl": "#2ca02c"}
LANG_LABELS = {"en": "EN", "de": "DE", "nl": "NL"}


def load_conditional_matrix(csv_path: Path, lang: str, symbols: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["language"] == lang]
    df = df[df["from"].isin(symbols) & df["to"].isin(symbols)]
    mat = df.pivot_table(index="from", columns="to", values="prob",
                         aggfunc="mean", fill_value=0.0)
    mat = mat.reindex(index=symbols, columns=symbols, fill_value=0.0)
    return mat


def plot_merged_heatmap(matrices: dict, out_png: Path, threshold=0.02):
    """
    Merged heatmap: jedna matica, v kazdej bunke su hodnoty vsetkych jazykov.
    Pozadie bunky je zafarbene podla maximalnej hodnoty.
    """
    symbols = CORE_PUNCT
    n = len(symbols)
    langs = list(matrices.keys())

    # Background: priemer cez jazyky
    avg_mat = np.zeros((n, n))
    for lang in langs:
        avg_mat += matrices[lang].values
    avg_mat /= len(langs)

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # pozadie heatmap
    im = ax.imshow(avg_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.55)

    # anotacie: v kazdej bunke su 3 riadky (EN/DE/NL)
    for i in range(n):
        for j in range(n):
            vals = {}
            for lang in langs:
                v = matrices[lang].values[i, j]
                if v >= threshold:
                    vals[lang] = v

            if not vals:
                continue

            # text pre kazdy jazyk
            lines = []
            for lang in langs:
                if lang in vals:
                    lines.append((LANG_LABELS[lang], f"{vals[lang]:.2f}", LANG_COLORS[lang]))

            # pozicia textu
            y_start = i - 0.25 + (0.5 - len(lines) * 0.17) / 2
            bg_val = avg_mat[i, j]
            for k, (label, val_str, color) in enumerate(lines):
                y_pos = y_start + k * 0.17
                txt = f"{label}:{val_str}"
                # cierna/biela podla pozadia
                text_color = color
                ax.text(j, y_pos, txt, ha="center", va="center",
                        fontsize=7, fontweight="bold", color=text_color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(symbols, fontsize=14, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(symbols, fontsize=14, fontweight="bold")
    ax.set_xlabel("to", fontsize=12)
    ax.set_ylabel("from", fontsize=12)

    # legenda
    legend_handles = [mpatches.Patch(facecolor=LANG_COLORS[l], label=LANG_LABELS[l])
                      for l in langs]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.15, 1.0),
              fontsize=10, title="Language")

    cbar = fig.colorbar(im, ax=ax, label="avg P(to|from)", shrink=0.8, pad=0.02)

    ax.set_title("Markov transition P(to|from) — all languages merged\n"
                 f"(values < {threshold} hidden)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_filtered_sidebyside(matrices: dict, out_png: Path, threshold=0.20):
    """
    Side-by-side heatmapy s threshold >= 0.20.
    Bunky pod 0.20 su prazdne/biele.
    """
    langs = list(matrices.keys())
    n = len(langs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        mat = matrices[lang].values.copy()
        mat[mat < threshold] = np.nan

        im = ax.imshow(mat, aspect="auto", vmin=0.2, vmax=0.6, cmap="YlOrRd")
        ax.set_xticks(range(len(CORE_PUNCT)))
        ax.set_xticklabels(CORE_PUNCT, fontsize=13, fontweight="bold")
        ax.set_yticks(range(len(CORE_PUNCT)))
        ax.set_yticklabels(CORE_PUNCT, fontsize=13, fontweight="bold")
        ax.set_title(f"{lang.upper()}", fontsize=14, fontweight="bold")

        # anotacie
        for i in range(len(CORE_PUNCT)):
            for j in range(len(CORE_PUNCT)):
                val = matrices[lang].values[i, j]
                if val >= threshold:
                    color = "white" if val > 0.4 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

    fig.colorbar(im, ax=axes, label="P(to|from)", shrink=0.7, pad=0.02)
    fig.suptitle(f"Markov transitions — values $\\geq$ {threshold} only", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transition_barplot(matrices: dict, out_png: Path, threshold=0.05):
    """
    Priecny rez: bar chart porovnanie konkretnych prechodov napriec jazykmi.
    Vyberie top prechody a porovna ich medzi jazykmi.
    """
    langs = list(matrices.keys())

    # Zozbieraj vsetky prechody
    transitions = []
    for i, fr in enumerate(CORE_PUNCT):
        for j, to in enumerate(CORE_PUNCT):
            avg_val = np.mean([matrices[l].values[i, j] for l in langs])
            if avg_val >= threshold:
                transitions.append((fr, to, avg_val))

    # Zorad podla priemernej hodnoty
    transitions.sort(key=lambda x: -x[2])
    top_trans = transitions[:15]  # top 15

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(top_trans))
    width = 0.25
    offsets = {"en": -width, "de": 0, "nl": width}

    for lang in langs:
        vals = []
        for fr, to, _ in top_trans:
            i = CORE_PUNCT.index(fr)
            j = CORE_PUNCT.index(to)
            vals.append(matrices[lang].values[i, j])
        ax.bar(x + offsets[lang], vals, width,
               color=LANG_COLORS[lang], alpha=0.8,
               label=lang.upper())

    # Descriptive labels for readability
    punct_names = {".": "dot", ",": "com", ";": "scol", ":": "col", "?": "qu", "!": "excl"}
    labels = [f"{punct_names.get(fr,fr)}->{punct_names.get(to,to)}" for fr, to, _ in top_trans]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, fontweight="bold", rotation=45, ha="right")
    ax.set_ylabel("P(to|from)", fontsize=12)
    ax.set_title("Cross-language comparison of top punctuation transitions", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_dominant_language_heatmap(matrices: dict, out_png: Path, threshold=0.10):
    """
    Heatmap kde farba bunky ukazuje ktory jazyk ma najvacsiu hodnotu.
    Intenzita ukazuje velkost rozdielu.
    """
    langs = list(matrices.keys())
    n = len(CORE_PUNCT)

    dominant = np.zeros((n, n, 3))  # RGB
    max_diff = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vals = {l: matrices[l].values[i, j] for l in langs}
            max_lang = max(vals, key=vals.get)
            max_val = vals[max_lang]
            min_val = min(vals.values())

            if max_val < threshold:
                dominant[i, j] = [0.95, 0.95, 0.95]  # light gray
                continue

            diff = max_val - min_val
            max_diff[i, j] = diff

            # farba podla dominantneho jazyka, intenzita podla rozdielu
            r, g, b = tuple(int(LANG_COLORS[max_lang].lstrip("#")[k:k+2], 16) / 255
                            for k in (0, 2, 4))
            intensity = min(1.0, diff / 0.15)  # scale diffs
            dominant[i, j] = [
                1 - intensity * (1 - r),
                1 - intensity * (1 - g),
                1 - intensity * (1 - b),
            ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(dominant, aspect="auto")

    for i in range(n):
        for j in range(n):
            vals = {l: matrices[l].values[i, j] for l in langs}
            max_lang = max(vals, key=vals.get)
            max_val = vals[max_lang]

            if max_val >= threshold:
                diff_str = f"Δ={max_val - min(vals.values()):.2f}"
                ax.text(j, i - 0.15, f"{LANG_LABELS[max_lang]}: {max_val:.2f}",
                        ha="center", va="center", fontsize=8, fontweight="bold")
                ax.text(j, i + 0.15, diff_str,
                        ha="center", va="center", fontsize=6.5, color="0.3")

    ax.set_xticks(range(n))
    ax.set_xticklabels(CORE_PUNCT, fontsize=13, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(CORE_PUNCT, fontsize=13, fontweight="bold")
    ax.set_xlabel("to", fontsize=12)
    ax.set_ylabel("from", fontsize=12)
    ax.set_title("Dominant language per transition\n(color = language with highest P, intensity = cross-language difference)",
                 fontsize=11)

    legend_handles = [mpatches.Patch(facecolor=LANG_COLORS[l], label=LANG_LABELS[l])
                      for l in langs]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-csv", default="markov_out_core/markov_conditional_probs_long.csv")
    ap.add_argument("--out-dir", default="improved_plots/heatmaps")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--threshold", type=float, default=0.02)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    matrices = {}
    for lang in langs:
        matrices[lang] = load_conditional_matrix(Path(args.cond_csv), lang, CORE_PUNCT)
        print(f"[OK] {lang}: matrix loaded")

    # 1. Merged heatmap
    plot_merged_heatmap(matrices, out_dir / "heatmap_merged_all_langs.png",
                        threshold=args.threshold)
    print("[OK] heatmap_merged_all_langs.png")

    # 2. Filtered side-by-side (threshold >= 0.20)
    plot_filtered_sidebyside(matrices, out_dir / "heatmap_filtered_020.png",
                             threshold=0.20)
    print("[OK] heatmap_filtered_020.png")

    # 3. Transition bar chart (priecny rez)
    plot_transition_barplot(matrices, out_dir / "transition_barplot.png",
                            threshold=0.05)
    print("[OK] transition_barplot.png")

    # 4. Dominant language heatmap
    plot_dominant_language_heatmap(matrices, out_dir / "heatmap_dominant_lang.png",
                                   threshold=0.10)
    print("[OK] heatmap_dominant_lang.png")

    print("\n[OK] Vsetky vylepsene heatmapy hotove.")


if __name__ == "__main__":
    main()
