#!/usr/bin/env python3
"""
Vylepsene Markov transition heatmapy:
  1. Filtrovane na hlavne interpunkcne znacky (. , ; : ? !)
  2. Side-by-side porovnanie vsetkych jazykov
  3. Delta heatmapy (rozdiely medzi jazykmi)
  4. Prahovanie: bunky pod threshold su prazdne
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Hlavne interpunkcne znacky (6 zakladnych)
CORE_PUNCT = [".", ",", ";", ":", "?", "!"]
CORE_LABELS = {"." : ".", "," : ",", ";" : ";", ":" : ":", "?" : "?", "!" : "!"}


def load_conditional_matrix(csv_path: Path, lang: str, symbols: list) -> pd.DataFrame:
    """Nacita P(to|from) maticu pre dany jazyk, filtrovanu na `symbols`."""
    df = pd.read_csv(csv_path)
    df = df[df["language"] == lang]
    df = df[df["from"].isin(symbols) & df["to"].isin(symbols)]

    mat = df.pivot_table(index="from", columns="to", values="prob", aggfunc="mean", fill_value=0.0)
    # reindex aby sme mali rovnaky poradie
    mat = mat.reindex(index=symbols, columns=symbols, fill_value=0.0)
    return mat


def plot_single_heatmap(mat: pd.DataFrame, title: str, out_png: Path,
                        vmin=0, vmax=0.6, annot=True, threshold=0.0):
    """Heatmapa jedneho jazyka."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    data = mat.values.copy()
    if threshold > 0:
        data[data < threshold] = np.nan

    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap="YlOrRd")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=12, fontweight="bold")

    ax.set_xlabel("to", fontsize=11)
    ax.set_ylabel("from", fontsize=11)
    ax.set_title(title, fontsize=12)

    # anotacie
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = mat.values[i, j]
                if val >= threshold and val > 0.005:
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="P(to|from)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_sidebyside(matrices: dict, out_png: Path, vmin=0, vmax=0.6, threshold=0.02):
    """Side-by-side heatmapy pre vsetky jazyky."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (lang, mat) in zip(axes, matrices.items()):
        data = mat.values.copy()
        data[data < threshold] = np.nan

        im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap="YlOrRd")

        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=12, fontweight="bold")
        ax.set_title(f"{lang.upper()}", fontsize=13, fontweight="bold")

        # anotacie
        for i in range(mat.values.shape[0]):
            for j in range(mat.values.shape[1]):
                val = mat.values[i, j]
                if val >= threshold:
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=axes, label="P(to|from)", shrink=0.7, pad=0.02)
    fig.suptitle("Markov transition P(to|from) -- core punctuation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(mat1: pd.DataFrame, mat2: pd.DataFrame,
                       lang1: str, lang2: str, out_png: Path, threshold=0.02):
    """Delta heatmapa: mat1 - mat2."""
    delta = mat1.values - mat2.values

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    vlim = max(abs(delta.min()), abs(delta.max()), 0.2)
    cmap = plt.cm.RdBu_r

    display = delta.copy()
    display[np.abs(display) < threshold] = np.nan

    im = ax.imshow(display, aspect="auto", vmin=-vlim, vmax=vlim, cmap=cmap)

    ax.set_xticks(range(len(mat1.columns)))
    ax.set_xticklabels(mat1.columns, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(mat1.index)))
    ax.set_yticklabels(mat1.index, fontsize=12, fontweight="bold")

    ax.set_xlabel("to", fontsize=11)
    ax.set_ylabel("from", fontsize=11)
    ax.set_title(f"Delta: {lang1.upper()} - {lang2.upper()}", fontsize=12)

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            if abs(val) >= threshold:
                color = "white" if abs(val) > vlim * 0.6 else "black"
                sign = "+" if val > 0 else ""
                ax.text(j, i, f"{sign}{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="difference", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-csv", default="markov_out_core/markov_conditional_probs_long.csv")
    ap.add_argument("--out-dir", default="markov_out_core/plots_v2")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="Bunky pod threshold su prazdne")
    ap.add_argument("--vmax", type=float, default=0.55)
    args = ap.parse_args()

    cond_csv = Path(args.cond_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    matrices = {}
    for lang in langs:
        mat = load_conditional_matrix(cond_csv, lang, CORE_PUNCT)
        matrices[lang] = mat

        # individual heatmap
        plot_single_heatmap(
            mat,
            f"P(to|from) -- {lang.upper()} (core punct)",
            out_dir / f"heatmap_core_{lang}.png",
            vmin=0, vmax=args.vmax, threshold=args.threshold,
        )
        print(f"[OK] {lang}: heatmap_core_{lang}.png")

    # side-by-side
    if len(matrices) > 1:
        plot_sidebyside(matrices, out_dir / "heatmap_sidebyside.png",
                        vmin=0, vmax=args.vmax, threshold=args.threshold)
        print(f"[OK] Side-by-side heatmap saved")

    # delta heatmapy (vsetky pary)
    lang_list = list(matrices.keys())
    for i in range(len(lang_list)):
        for j in range(i + 1, len(lang_list)):
            l1, l2 = lang_list[i], lang_list[j]
            plot_delta_heatmap(
                matrices[l1], matrices[l2], l1, l2,
                out_dir / f"delta_{l1}_minus_{l2}.png",
                threshold=args.threshold,
            )
            print(f"[OK] Delta: {l1} - {l2}")

    print("Hotovo!")


if __name__ == "__main__":
    main()
