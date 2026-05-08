#!/usr/bin/env python3
"""
Centrality analysis: betweenness a PageRank pre interpunkciu vs slova.
Porovna ci interpunkcia ma vyssiu centralitu v sieti nez bezne slova.
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


def build_graph(tokens, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    for u, v in iter_bigrams(tokens):
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def load_all_tokens(lang_dir, lowercase=True):
    tokens = []
    for f in sorted(lang_dir.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace")
        toks = text.split()
        if lowercase:
            toks = [t.lower() for t in toks]
        tokens.extend(toks)
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--outdir", default="network_out_core/centrality")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_rows = []

    for lang in langs:
        lang_dir = Path(args.token_dir) / lang
        if not lang_dir.exists():
            print(f"[WARN] {lang}: missing {lang_dir}")
            continue

        tokens = load_all_tokens(lang_dir, args.lowercase)
        print(f"[INFO] {lang}: {len(tokens)} tokens")

        # Build directed graph (for PageRank) and undirected (for degree)
        Gd = build_graph(tokens, directed=True)
        Gu = build_graph(tokens, directed=False)

        print(f"  nodes={Gd.number_of_nodes()}, edges(dir)={Gd.number_of_edges()}")

        # Compute centralities (skip betweenness — too slow on aggregated graph)
        print(f"  computing PageRank...")
        pr = nx.pagerank(Gd, weight="weight")

        print(f"  computing degree centrality...")
        dc = nx.degree_centrality(Gu)

        # Weighted degree (strength) as proxy for betweenness
        print(f"  computing weighted degree (strength)...")
        strength = {n: sum(d["weight"] for _, _, d in Gu.edges(n, data=True)) for n in Gu.nodes()}
        max_str = max(strength.values()) if strength else 1
        bc = {n: v / max_str for n, v in strength.items()}  # normalized strength

        # Frequency for reference
        freq = Counter(tokens)

        # Collect per-node stats
        for node in Gd.nodes():
            all_rows.append({
                "lang": lang,
                "token": node,
                "is_punct": int(node in PUNCT_SET),
                "freq": freq.get(node, 0),
                "pagerank": pr.get(node, 0),
                "strength": bc.get(node, 0),
                "degree_centrality": dc.get(node, 0),
            })

        # --- Top N bar charts ---
        # Sort by each centrality metric
        for metric_name, metric_dict in [("pagerank", pr), ("strength", bc)]:
            sorted_nodes = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:args.top_n]

            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [n for n, _ in sorted_nodes]
            vals = [v for _, v in sorted_nodes]
            colors = ["red" if l in PUNCT_SET else "steelblue" for l in labels]

            ax.barh(range(len(labels)), vals, color=colors, edgecolor="gray", linewidth=0.3)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel(metric_name)
            ax.set_title(f"Top {args.top_n} {metric_name} - {lang.upper()}")

            # Legend
            from matplotlib.patches import Patch
            legend_el = [Patch(facecolor="red", label="punctuation"),
                         Patch(facecolor="steelblue", label="word")]
            ax.legend(handles=legend_el, fontsize=9)

            fig.tight_layout()
            fig.savefig(outdir / f"top_{metric_name}_{lang}.png", dpi=300)
            plt.close(fig)

        # --- Scatter: PageRank vs frequency ---
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        for node in Gd.nodes():
            f_n = freq.get(node, 0)
            pr_n = pr.get(node, 0)
            is_p = node in PUNCT_SET
            if is_p:
                ax2.scatter(f_n, pr_n, c="red", s=50, zorder=5, edgecolors="k", linewidths=0.5)
                ax2.annotate(node, (f_n, pr_n), fontsize=8, ha="left", va="bottom")
            else:
                ax2.scatter(f_n, pr_n, c="steelblue", s=5, alpha=0.3, zorder=2)

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Token frequency")
        ax2.set_ylabel("PageRank")
        ax2.set_title(f"PageRank vs frequency - {lang.upper()}")
        fig2.tight_layout()
        fig2.savefig(outdir / f"pagerank_vs_freq_{lang}.png", dpi=300)
        plt.close(fig2)

        # --- Punct vs words: boxplot of centrality ---
        fig3, axes = plt.subplots(1, 3, figsize=(10, 4))
        for ax, (mname, mdict) in zip(axes, [("PageRank", pr), ("Strength", bc), ("Degree", dc)]):
            punct_vals = [mdict[n] for n in mdict if n in PUNCT_SET]
            # top 50 words by frequency (excluding punct) for comparison
            word_nodes_sorted = sorted(
                [n for n in mdict if n not in PUNCT_SET],
                key=lambda n: freq.get(n, 0), reverse=True
            )[:50]
            word_vals = [mdict[n] for n in word_nodes_sorted]

            data = []
            labels_b = []
            if punct_vals:
                data.append(punct_vals)
                labels_b.append("punct")
            if word_vals:
                data.append(word_vals)
                labels_b.append("top50 words")

            if data:
                bp = ax.boxplot(data, tick_labels=labels_b, patch_artist=True)
                for patch, c in zip(bp["boxes"], ["red", "steelblue"]):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.4)
            ax.set_title(mname)
            ax.set_ylabel(mname.lower())

        fig3.suptitle(f"Punct vs top-50 words centrality - {lang.upper()}", fontsize=12)
        fig3.tight_layout()
        fig3.savefig(outdir / f"centrality_boxplot_{lang}.png", dpi=300)
        plt.close(fig3)

        print(f"[OK] {lang}: centrality plots saved")

    # Save CSV
    out_csv = outdir / "centrality_all.csv"
    import pandas as pd
    pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[OK] CSV: {out_csv}")


if __name__ == "__main__":
    main()
