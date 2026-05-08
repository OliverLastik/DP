#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def iter_bigrams(tokens: List[str]):
    prev = None
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if prev is not None:
            yield prev, t
        prev = t

def read_tokens_file(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.split()

def build_graph_from_tokens(tokens: List[str], directed: bool = True, weighted: bool = True):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    c = Counter(iter_bigrams(tokens))
    if weighted:
        for (u, v), w in c.items():
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)
    else:
        for (u, v) in c.keys():
            G.add_edge(u, v)
    return G

def degree_histogram_undirected(Gu: nx.Graph) -> Dict[int, int]:
    # histogram of degrees
    hist = Counter(dict(Gu.degree()).values())
    return dict(sorted(hist.items(), key=lambda x: x[0]))

def save_degree_plot(hist: Dict[int, int], out_png: Path, title: str):
    # P(k) ~ counts normalized; plot log-log (skip k=0)
    ks = [k for k in hist.keys() if k > 0]
    if not ks:
        return
    counts = [hist[k] for k in ks]
    total = sum(counts)
    pk = [c / total for c in counts]

    plt.figure()
    plt.plot(ks, pk, marker="o", linestyle="none")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("k (degree)")
    plt.ylabel("P(k)")
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def largest_component_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()

def compute_metrics_for_book(lang: str, gid: str, tokens: List[str]):
    # directed weighted graph (main)
    Gd = build_graph_from_tokens(tokens, directed=True, weighted=True)

    # undirected unweighted graph for clustering / components
    Gu = build_graph_from_tokens(tokens, directed=False, weighted=False)

    n_nodes = Gu.number_of_nodes()
    n_edges = Gu.number_of_edges()

    # degrees
    degs = dict(Gu.degree())
    avg_degree = (sum(degs.values()) / n_nodes) if n_nodes > 0 else 0.0

    # density
    density = nx.density(Gu) if n_nodes > 1 else 0.0

    # components
    n_components = nx.number_connected_components(Gu) if n_nodes > 0 else 0
    lcc = largest_component_subgraph(Gu)
    lcc_nodes = lcc.number_of_nodes()
    lcc_frac = (lcc_nodes / n_nodes) if n_nodes > 0 else 0.0

    # clustering (on undirected)
    clustering = nx.average_clustering(Gu) if n_nodes > 1 else 0.0

    # avg shortest path (only LCC; expensive if huge)
    # We'll compute only if LCC isn't enormous
    avg_spl = None
    if lcc_nodes >= 2 and lcc_nodes <= 6000:
        try:
            avg_spl = nx.average_shortest_path_length(lcc)
        except Exception:
            avg_spl = None

    return {
        "language": lang,
        "gutenberg_id": gid,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "density": density,
        "n_components": n_components,
        "lcc_nodes": lcc_nodes,
        "lcc_fraction": lcc_frac,
        "avg_clustering": clustering,
        "avg_shortest_path_lcc": avg_spl if avg_spl is not None else "",
    }, degree_histogram_undirected(Gu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-root", default="token_out_core/tokens_words_plus_punct",
                    help="Root folder with token files per language")
    ap.add_argument("--out", default="network_out_core",
                    help="Output root")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--max-books", type=int, default=0,
                    help="Limit number of books per language for testing (0=all)")
    args = ap.parse_args()

    token_root = Path(args.token_root)
    out_root = Path(args.out)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir = out_root / "plots"
    deg_dir = out_root / "degree_histograms"

    rows = []

    for lang in langs:
        lang_dir = token_root / lang
        if not lang_dir.exists():
            print(f"[WARN] Missing lang dir: {lang_dir}")
            continue

        files = sorted(lang_dir.glob("*.txt"))
        if args.max_books and args.max_books > 0:
            files = files[: args.max_books]

        # aggregate degree histogram over books (sum histograms)
        agg_hist = Counter()

        for fp in files:
            gid = fp.stem
            tokens = read_tokens_file(fp)

            metrics, hist = compute_metrics_for_book(lang, gid, tokens)
            rows.append(metrics)
            agg_hist.update(hist)

            # Save per-book degree plot (optional: comment out if too many pngs)
            # save_degree_plot(hist, plots_dir / "per_book" / lang / f"degree_pk_{gid}.png",
            #                 title=f"P(k) for {lang}:{gid} (undirected)")

        # save aggregated P(k) per language
        save_degree_plot(dict(agg_hist), plots_dir / f"degree_pk_{lang}.png",
                        title=f"Degree distribution P(k) (aggregated) - {lang}")

        # save aggregated histogram csv
        deg_dir.mkdir(parents=True, exist_ok=True)
        out_hist_csv = deg_dir / f"degree_hist_{lang}.csv"
        with out_hist_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "count"])
            for k, c in sorted(agg_hist.items(), key=lambda x: x[0]):
                w.writerow([k, c])

        print(f"[OK] {lang}: books={len(files)} saved {out_hist_csv} and degree_pk_{lang}.png")

    # Save metrics per book
    df = pd.DataFrame(rows)
    out_csv = out_root / "metrics_per_book.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved metrics: {out_csv}")

    # Summarize by language (median + basic stats)
    if not df.empty:
        summary = df.groupby("language").agg({
            "n_nodes": "median",
            "n_edges": "median",
            "avg_degree": "median",
            "density": "median",
            "n_components": "median",
            "lcc_nodes": "median",
            "lcc_fraction": "median",
            "avg_clustering": "median",
        }).reset_index()
        out_sum = out_root / "metrics_summary_by_language.csv"
        summary.to_csv(out_sum, index=False, encoding="utf-8")
        print(f"[OK] Saved summary: {out_sum}")

if __name__ == "__main__":
    main()