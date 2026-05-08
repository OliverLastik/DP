#!/usr/bin/env python3
"""
Node-level analysis: lokalna klasterizacia C_i a priemerna vzdialenost l_i
pre interpunkcne uzly vs slovne uzly.
Ciel: ukazat ze interpunkcia sa integruje do siete ako plnohodnotny uzol.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


PUNCT_SET = {".", ",", "!", "?", ";", ":"}


def iter_bigrams(tokens):
    prev = None
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if prev is not None:
            yield prev, t
        prev = t


def build_undirected(tokens):
    G = nx.Graph()
    for u, v in iter_bigrams(tokens):
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--outdir", default="network_out_core/node_level")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--lowercase", action="store_true", default=True)
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

        # Process per book, aggregate stats
        all_ci_punct = []
        all_ci_words = []
        all_ki_punct = []
        all_ki_words = []

        files = sorted(lang_dir.glob("*.txt"))
        print(f"[INFO] {lang}: {len(files)} books")

        for fi, fp in enumerate(files):
            text = fp.read_text(encoding="utf-8", errors="replace")
            toks = text.split()
            if args.lowercase:
                toks = [t.lower() for t in toks]

            G = build_undirected(toks)
            n = G.number_of_nodes()
            if n < 10:
                continue

            freq = Counter(toks)

            # Local clustering
            ci = nx.clustering(G)
            for node, c in ci.items():
                if node in PUNCT_SET:
                    all_ci_punct.append(c)
                    all_ki_punct.append(G.degree(node))
                else:
                    all_ci_words.append(c)
                    all_ki_words.append(G.degree(node))

            if (fi + 1) % 20 == 0:
                print(f"  {lang}: {fi+1}/{len(files)} books processed")

        # --- Summary stats ---
        ci_p = np.array(all_ci_punct) if all_ci_punct else np.array([])
        ci_w = np.array(all_ci_words) if all_ci_words else np.array([])
        ki_p = np.array(all_ki_punct) if all_ki_punct else np.array([])
        ki_w = np.array(all_ki_words) if all_ki_words else np.array([])

        summary_rows.append({
            "lang": lang,
            "ci_punct_mean": float(ci_p.mean()) if len(ci_p) > 0 else float("nan"),
            "ci_punct_median": float(np.median(ci_p)) if len(ci_p) > 0 else float("nan"),
            "ci_words_mean": float(ci_w.mean()) if len(ci_w) > 0 else float("nan"),
            "ci_words_median": float(np.median(ci_w)) if len(ci_w) > 0 else float("nan"),
            "ki_punct_mean": float(ki_p.mean()) if len(ki_p) > 0 else float("nan"),
            "ki_words_mean": float(ki_w.mean()) if len(ki_w) > 0 else float("nan"),
            "n_punct_samples": len(ci_p),
            "n_word_samples": len(ci_w),
        })

        print(f"  {lang}: punct C_i={ci_p.mean():.4f}, words C_i={ci_w.mean():.4f}")
        print(f"  {lang}: punct k_i={ki_p.mean():.1f}, words k_i={ki_w.mean():.1f}")

        # --- Boxplot: C_i punct vs words ---
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # C_i boxplot
        ax = axes[0]
        data = []
        labels = []
        if len(ci_p) > 0:
            data.append(ci_p); labels.append(f"punct\n(n={len(ci_p)})")
        if len(ci_w) > 0:
            idx = np.random.choice(len(ci_w), min(1000, len(ci_w)), replace=False)
            data.append(ci_w[idx]); labels.append(f"words\n(n={len(ci_w)})")
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for p, c in zip(bp["boxes"], ["red", "steelblue"]):
                p.set_facecolor(c); p.set_alpha(0.4)
        ax.set_ylabel("$C_i$ (local clustering)")
        ax.set_title("Local clustering")

        # k_i boxplot
        ax = axes[1]
        data = []
        labels = []
        if len(ki_p) > 0:
            data.append(ki_p); labels.append("punct")
        if len(ki_w) > 0:
            idx = np.random.choice(len(ki_w), min(1000, len(ki_w)), replace=False)
            data.append(ki_w[idx]); labels.append("words")
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for p, c in zip(bp["boxes"], ["red", "steelblue"]):
                p.set_facecolor(c); p.set_alpha(0.4)
        ax.set_ylabel("$k_i$ (degree)")
        ax.set_title("Node degree")

        fig.suptitle(f"Node-level: punct vs words - {lang.upper()}", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f"node_level_{lang}.png", dpi=300)
        plt.close(fig)

        # --- C_i vs k_i scatter ---
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        if len(ki_w) > 0:
            idx = np.random.choice(len(ki_w), min(2000, len(ki_w)), replace=False)
            ax2.scatter(ki_w[idx], ci_w[idx], c="steelblue", s=5, alpha=0.3, label="words")
        if len(ki_p) > 0:
            ax2.scatter(ki_p, ci_p, c="red", s=40, zorder=5, edgecolors="k",
                        linewidths=0.5, label="punctuation")
        ax2.set_xlabel("$k_i$ (degree)")
        ax2.set_ylabel("$C_i$ (local clustering)")
        ax2.set_title(f"C(k) - {lang.upper()}")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(outdir / f"ci_vs_ki_{lang}.png", dpi=300)
        plt.close(fig2)

        print(f"[OK] {lang}: node-level plots saved")

    # Save CSV
    out_csv = outdir / "node_level_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "lang", "ci_punct_mean", "ci_punct_median", "ci_words_mean", "ci_words_median",
            "ki_punct_mean", "ki_words_mean",
            "n_punct_samples", "n_word_samples"
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[OK] CSV: {out_csv}")


if __name__ == "__main__":
    main()
