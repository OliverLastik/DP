#!/usr/bin/env python3
"""
Porovnanie sietovych metrik: words-only vs words+punct.
Pre kazdu knihu vytvori dve siete, spocita metriky, a porovna.
Vystup: CSV, boxploty, delta statistiky.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def iter_bigrams(tokens):
    prev = None
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if prev is not None:
            yield prev, t
        prev = t


def build_undirected_graph(tokens):
    G = nx.Graph()
    for u, v in iter_bigrams(tokens):
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def largest_cc(G):
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    return G.subgraph(max(comps, key=len)).copy()


def compute_metrics(tokens):
    G = build_undirected_graph(tokens)
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n < 2:
        return {"n_nodes": n, "n_edges": e, "avg_degree": 0, "density": 0,
                "avg_clustering": 0, "assortativity": float("nan"),
                "lcc_fraction": 0}

    degs = dict(G.degree())
    avg_deg = sum(degs.values()) / n
    density = nx.density(G)
    clust = nx.average_clustering(G)

    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = float("nan")

    lcc = largest_cc(G)
    lcc_frac = lcc.number_of_nodes() / n

    return {
        "n_nodes": n,
        "n_edges": e,
        "avg_degree": avg_deg,
        "density": density,
        "avg_clustering": clust,
        "assortativity": assort,
        "lcc_fraction": lcc_frac,
    }


def load_tokens(path, lowercase=True):
    text = path.read_text(encoding="utf-8", errors="replace")
    toks = text.split()
    if lowercase:
        toks = [t.lower() for t in toks]
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words-dir", default="token_out_core/tokens_words")
    ap.add_argument("--punct-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--outdir", default="network_out_core/words_vs_punct")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--max-books", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_rows = []
    metrics_keys = ["n_nodes", "n_edges", "avg_degree", "density",
                    "avg_clustering", "assortativity", "lcc_fraction"]

    for lang in langs:
        words_dir = Path(args.words_dir) / lang
        punct_dir = Path(args.punct_dir) / lang

        if not words_dir.exists():
            print(f"[WARN] {lang}: missing {words_dir}")
            continue

        book_files = sorted(words_dir.glob("*.txt"))
        if args.max_books > 0:
            book_files = book_files[:args.max_books]

        print(f"[INFO] {lang}: {len(book_files)} books")

        for bf in book_files:
            gid = bf.stem
            toks_w = load_tokens(bf, args.lowercase)
            m_w = compute_metrics(toks_w)

            pf = punct_dir / bf.name
            if pf.exists():
                toks_p = load_tokens(pf, args.lowercase)
                m_p = compute_metrics(toks_p)
            else:
                m_p = {k: float("nan") for k in metrics_keys}

            row = {"lang": lang, "book": gid}
            for k in metrics_keys:
                row[f"w_{k}"] = m_w[k]
                row[f"wp_{k}"] = m_p[k]
                # delta (words+punct - words)
                try:
                    row[f"d_{k}"] = m_p[k] - m_w[k]
                except Exception:
                    row[f"d_{k}"] = float("nan")
            all_rows.append(row)

            if len(all_rows) % 20 == 0:
                print(f"  processed {len(all_rows)} books...")

    df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = outdir / "words_vs_punct_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] CSV: {csv_path}")

    # --- Boxplots per metric ---
    plot_metrics = ["n_nodes", "avg_degree", "density", "avg_clustering",
                    "assortativity", "lcc_fraction"]

    for metric in plot_metrics:
        fig, axes = plt.subplots(1, len(langs), figsize=(4 * len(langs), 4), sharey=True)
        if len(langs) == 1:
            axes = [axes]

        for ax, lang in zip(axes, langs):
            sub = df[df["lang"] == lang]
            w_vals = sub[f"w_{metric}"].dropna().values
            p_vals = sub[f"wp_{metric}"].dropna().values

            if len(w_vals) == 0 and len(p_vals) == 0:
                continue

            data = []
            labels = []
            if len(w_vals) > 0:
                data.append(w_vals)
                labels.append("words")
            if len(p_vals) > 0:
                data.append(p_vals)
                labels.append("w+punct")

            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, c in zip(bp["boxes"], ["C0", "C1"]):
                patch.set_facecolor(c)
                patch.set_alpha(0.4)
            ax.set_title(lang.upper())

        fig.suptitle(metric.replace("_", " "), fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f"boxplot_{metric}.png", dpi=300)
        plt.close(fig)

    # --- Summary table ---
    summary_rows = []
    for lang in langs:
        sub = df[df["lang"] == lang]
        row = {"lang": lang, "n_books": len(sub)}
        for m in metrics_keys:
            w = sub[f"w_{m}"].dropna()
            p = sub[f"wp_{m}"].dropna()
            d = sub[f"d_{m}"].dropna()
            row[f"w_{m}_med"] = w.median() if len(w) > 0 else float("nan")
            row[f"wp_{m}_med"] = p.median() if len(p) > 0 else float("nan")
            row[f"d_{m}_med"] = d.median() if len(d) > 0 else float("nan")
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    sum_path = outdir / "words_vs_punct_summary.csv"
    sum_df.to_csv(sum_path, index=False, encoding="utf-8")
    print(f"[OK] Summary: {sum_path}")

    # --- Delta bar chart ---
    delta_metrics = ["n_nodes", "avg_degree", "density", "avg_clustering"]
    fig, axes = plt.subplots(1, len(delta_metrics), figsize=(3.5 * len(delta_metrics), 4))
    if len(delta_metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, delta_metrics):
        medians = []
        for lang in langs:
            sub = df[df["lang"] == lang]
            d = sub[f"d_{m}"].dropna()
            medians.append(d.median() if len(d) > 0 else 0)

        colors_bar = ["C0", "C1", "C2"]
        ax.bar(langs, medians, color=colors_bar[:len(langs)], alpha=0.7)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title(m.replace("_", " "))
        ax.set_ylabel("median delta\n(w+punct - words)")

    fig.suptitle("Effect of adding punctuation to network", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "delta_bar_chart.png", dpi=300)
    plt.close(fig)

    print("[OK] All plots saved.")


if __name__ == "__main__":
    main()
