#!/usr/bin/env python3
"""
Shuffled text null model.
Nahodne premiesat tokeny v kazdom texte a porovnat sietove metriky
s originalom. Ak je siet jazykovo vyznamna, shuffling by mal znicit
klasterizaciu a zmenit ASPL.
"""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

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


def build_undirected(tokens):
    G = nx.Graph()
    for u, v in iter_bigrams(tokens):
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def compute_metrics(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n < 2:
        return {"n_nodes": n, "n_edges": e, "avg_degree": 0, "density": 0,
                "avg_clustering": 0, "assortativity": float("nan"),
                "avg_shortest_path_lcc": float("nan")}

    degs = dict(G.degree())
    avg_deg = sum(degs.values()) / n
    density = nx.density(G)
    clust = nx.average_clustering(G)

    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = float("nan")

    comps = list(nx.connected_components(G))
    lcc = G.subgraph(max(comps, key=len)).copy() if comps else G
    aspl = float("nan")
    if 2 <= lcc.number_of_nodes() <= 5000:
        try:
            aspl = nx.average_shortest_path_length(lcc)
        except Exception:
            pass

    return {
        "n_nodes": n, "n_edges": e, "avg_degree": avg_deg,
        "density": density, "avg_clustering": clust,
        "assortativity": assort, "avg_shortest_path_lcc": aspl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--outdir", default="network_out_core/shuffled_null")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--n-shuffles", type=int, default=5, help="Kolko nahodnych shufflov na knihu")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--max-books", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_rows = []
    metrics_keys = ["n_nodes", "n_edges", "avg_degree", "density",
                    "avg_clustering", "assortativity", "avg_shortest_path_lcc"]

    for lang in langs:
        lang_dir = Path(args.token_dir) / lang
        if not lang_dir.exists():
            print(f"[WARN] {lang}: missing {lang_dir}")
            continue

        files = sorted(lang_dir.glob("*.txt"))
        if args.max_books > 0:
            files = files[:args.max_books]

        print(f"[INFO] {lang}: {len(files)} books, {args.n_shuffles} shuffles each")

        for fi, fp in enumerate(files):
            text = fp.read_text(encoding="utf-8", errors="replace")
            toks = text.split()
            if args.lowercase:
                toks = [t.lower() for t in toks]

            # Real network
            G_real = build_undirected(toks)
            m_real = compute_metrics(G_real)

            row = {"lang": lang, "book": fp.stem, "kind": "real"}
            row.update(m_real)
            all_rows.append(row)

            # Shuffled networks
            for si in range(args.n_shuffles):
                shuffled = toks.copy()
                random.shuffle(shuffled)
                G_shuf = build_undirected(shuffled)
                m_shuf = compute_metrics(G_shuf)

                row_s = {"lang": lang, "book": fp.stem, "kind": f"shuffled_{si}"}
                row_s.update(m_shuf)
                all_rows.append(row_s)

            if (fi + 1) % 20 == 0:
                print(f"  {lang}: {fi+1}/{len(files)} books")

    df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = outdir / "shuffled_vs_real_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] CSV: {csv_path}")

    # --- Boxplots: real vs shuffled ---
    plot_metrics = ["avg_clustering", "avg_degree", "density",
                    "avg_shortest_path_lcc", "assortativity"]

    for metric in plot_metrics:
        fig, axes = plt.subplots(1, len(langs), figsize=(4 * len(langs), 4), sharey=True)
        if len(langs) == 1:
            axes = [axes]

        for ax, lang in zip(axes, langs):
            sub = df[df["lang"] == lang]
            real_vals = sub[sub["kind"] == "real"][metric].dropna().values
            shuf_vals = sub[sub["kind"].str.startswith("shuffled")][metric].dropna().values

            data = []
            labels_b = []
            if len(real_vals) > 0:
                data.append(real_vals); labels_b.append("real")
            if len(shuf_vals) > 0:
                data.append(shuf_vals); labels_b.append("shuffled")

            if data:
                bp = ax.boxplot(data, tick_labels=labels_b, patch_artist=True)
                for patch, c in zip(bp["boxes"], ["C0", "C3"]):
                    patch.set_facecolor(c); patch.set_alpha(0.4)
            ax.set_title(lang.upper())

        fig.suptitle(f"{metric.replace('_', ' ')}: real vs shuffled", fontsize=13)
        fig.tight_layout()
        fig.savefig(outdir / f"boxplot_{metric}.png", dpi=300)
        plt.close(fig)

    # --- Summary ---
    summary_rows = []
    for lang in langs:
        sub = df[df["lang"] == lang]
        real = sub[sub["kind"] == "real"]
        shuf = sub[sub["kind"].str.startswith("shuffled")]
        row = {"lang": lang}
        for m in metrics_keys:
            r = real[m].dropna()
            s = shuf[m].dropna()
            row[f"real_{m}_med"] = r.median() if len(r) > 0 else float("nan")
            row[f"shuf_{m}_med"] = s.median() if len(s) > 0 else float("nan")
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(outdir / "shuffled_summary.csv", index=False, encoding="utf-8")

    print("[OK] All plots and summary saved.")


if __name__ == "__main__":
    main()
