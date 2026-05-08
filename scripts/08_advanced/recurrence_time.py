#!/usr/bin/env python3
"""
Recurrence time analysis: kolko tokenov prejde medzi dvoma vyskytmi toho isteho tokenu.
Porovnanie interpunkcie vs slov.
Kulig et al. skumali ci interpunkcia ma rovnake distribucne vlastnosti.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PUNCT_SET = {".", ",", "!", "?", ";", ":"}

def compute_recurrence_times(tokens: list[str]) -> dict[str, list[int]]:
    """
    Pre kazdy token spocita medzery medzi jeho po sebe iducimi vyskytmi.
    Vrati dict: token -> list of recurrence times.
    """
    last_pos = {}
    rec = defaultdict(list)

    for i, tok in enumerate(tokens):
        if tok in last_pos:
            rec[tok].append(i - last_pos[tok])
        last_pos[tok] = i

    return dict(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--outdir", default="rankfreq_out_core/recurrence_time")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--top-words", type=int, default=10,
                    help="Kolko najcastejsich slov zobrazit v CCDF")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    summary_rows = []

    for lang in langs:
        lang_dir = Path(args.token_dir) / lang
        if not lang_dir.exists():
            print(f"[WARN] {lang}: missing {lang_dir}")
            continue

        # Concatenate all books
        all_tokens = []
        for f in sorted(lang_dir.glob("*.txt")):
            text = f.read_text(encoding="utf-8", errors="replace")
            toks = text.split()
            if args.lowercase:
                toks = [t.lower() for t in toks]
            all_tokens.extend(toks)

        print(f"[INFO] {lang}: {len(all_tokens)} tokens")

        rec = compute_recurrence_times(all_tokens)

        # Token frequencies for ranking
        from collections import Counter
        freq = Counter(all_tokens)

        # Separate punct vs words
        punct_tokens = [t for t in rec if t in PUNCT_SET]
        word_tokens = [t for t in rec if t not in PUNCT_SET]

        # Sort words by frequency
        word_tokens.sort(key=lambda t: freq[t], reverse=True)
        top_words = word_tokens[:args.top_words]

        # --- Stats per token ---
        for tok in punct_tokens + top_words:
            times = rec[tok]
            if len(times) < 5:
                continue
            arr = np.array(times)
            summary_rows.append({
                "lang": lang,
                "token": tok,
                "is_punct": int(tok in PUNCT_SET),
                "count": freq[tok],
                "n_recurrences": len(times),
                "mean_rec": float(np.mean(arr)),
                "median_rec": float(np.median(arr)),
                "std_rec": float(np.std(arr)),
                "cv": float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
            })

        # --- CCDF plot: punct vs top words ---
        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot punctuation
        for tok in sorted(punct_tokens, key=lambda t: freq[t], reverse=True):
            times = rec[tok]
            if len(times) < 20:
                continue
            arr = np.sort(times)
            ccdf = np.arange(len(arr), 0, -1) / len(arr)
            ax.plot(arr, ccdf, linewidth=2, label=f"'{tok}' (punct, n={freq[tok]})")

        # Plot top words (thinner lines)
        for tok in top_words[:5]:
            times = rec[tok]
            if len(times) < 20:
                continue
            arr = np.sort(times)
            ccdf = np.arange(len(arr), 0, -1) / len(arr)
            ax.plot(arr, ccdf, "--", linewidth=1, alpha=0.7,
                    label=f"'{tok}' (word, n={freq[tok]})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Recurrence time (tokens)")
        ax.set_ylabel("CCDF P(T >= t)")
        ax.set_title(f"Recurrence time CCDF - {lang.upper()}")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"recurrence_ccdf_{lang}.png", dpi=300)
        plt.close(fig)

        # --- Mean recurrence time vs frequency (scatter) ---
        # Batch into arrays for fast rendering
        from matplotlib.lines import Line2D
        word_freqs, word_means = [], []
        punct_freqs_s, punct_means_s, punct_labels = [], [], []
        word_cvs, punct_cvs = [], []

        for tok, times in rec.items():
            if len(times) < 10:
                continue
            arr = np.array(times)
            f_tok = freq[tok]
            mean_t = np.mean(arr)
            cv = np.std(arr) / mean_t if mean_t > 0 else 0
            if tok in PUNCT_SET:
                punct_freqs_s.append(f_tok)
                punct_means_s.append(mean_t)
                punct_labels.append(tok)
                punct_cvs.append(cv)
            else:
                word_freqs.append(f_tok)
                word_means.append(mean_t)
                word_cvs.append(cv)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        if word_freqs:
            ax2.scatter(word_freqs, word_means, c="steelblue", s=8, alpha=0.3, zorder=2)
        if punct_freqs_s:
            ax2.scatter(punct_freqs_s, punct_means_s, c="red", s=40, zorder=5,
                        edgecolors="k", linewidths=0.5)
            for lbl, fx, my in zip(punct_labels, punct_freqs_s, punct_means_s):
                ax2.annotate(lbl, (fx, my), fontsize=8, ha="left", va="bottom")

        freqs_range = np.logspace(1, np.log10(max(freq.values())), 100)
        ax2.plot(freqs_range, len(all_tokens) / freqs_range, "k--", alpha=0.5)
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("Token frequency"); ax2.set_ylabel("Mean recurrence time")
        ax2.set_title(f"Mean recurrence vs frequency - {lang.upper()}")
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, markeredgecolor='k', label='punctuation'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=6, alpha=0.5, label='words'),
            Line2D([0], [0], linestyle='--', color='k', alpha=0.5, label='N/f (random)'),
        ]
        ax2.legend(handles=legend_elements, fontsize=8)
        fig2.tight_layout()
        fig2.savefig(outdir / f"recurrence_vs_freq_{lang}.png", dpi=300)
        plt.close(fig2)

        # --- Coefficient of variation (burstiness) scatter ---
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        if word_freqs:
            ax3.scatter(word_freqs, word_cvs, c="steelblue", s=8, alpha=0.3, zorder=2)
        if punct_freqs_s:
            ax3.scatter(punct_freqs_s, punct_cvs, c="red", s=40, zorder=5,
                        edgecolors="k", linewidths=0.5)
            for lbl, fx, cy in zip(punct_labels, punct_freqs_s, punct_cvs):
                ax3.annotate(lbl, (fx, cy), fontsize=8, ha="left", va="bottom")

        ax3.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax3.set_xscale("log")
        ax3.set_xlabel("Token frequency")
        ax3.set_ylabel("CV (std/mean) of recurrence time")
        ax3.set_title(f"Burstiness (CV) vs frequency - {lang.upper()}")
        legend_elements2 = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, markeredgecolor='k', label='punctuation'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=6, alpha=0.5, label='words'),
            Line2D([0], [0], linestyle=':', color='gray', alpha=0.5, label='CV=1 (Poisson)'),
        ]
        ax3.legend(handles=legend_elements2, fontsize=8)
        fig3.tight_layout()
        fig3.savefig(outdir / f"burstiness_{lang}.png", dpi=300)
        plt.close(fig3)

        print(f"[OK] {lang}: plots saved")

    # Save summary CSV
    out_csv = outdir / "recurrence_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang", "token", "is_punct", "count", "n_recurrences",
            "mean_rec", "median_rec", "std_rec", "cv"
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[OK] CSV: {out_csv}")


if __name__ == "__main__":
    main()
