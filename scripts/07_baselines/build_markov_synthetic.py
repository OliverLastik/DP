#!/usr/bin/env python3
"""
Markov-generated synteticka interpunkcia.

1. Nacita natrenovanu Markov maticu P(to|from) pre kazdy jazyk
2. Pre kazdu knihu: nahradi realnu interpunkciu syntetickou
   generovanou z Markov retazca
3. Vytvori word-adjacency siet z novych tokenov
4. Porovna metriky sieti: real vs Markov-synthetic vs random-punct

Vystup: CSV s metrikami, porovnavacie ploty.
"""
import argparse
import csv
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


PUNCT_SET = {".", ",", "?", "!", ";", ":"}


def load_markov_matrix(csv_path: Path, lang: str) -> dict:
    """
    Nacita conditional probs P(to|from) pre dany jazyk.
    Vracia dict: from_symbol -> [(to_symbol, prob), ...]
    """
    transitions = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["language"] != lang:
                continue
            fr = row["from"]
            to = row["to"]
            prob = float(row["prob"])
            if fr in PUNCT_SET and to in PUNCT_SET and prob > 0:
                transitions[fr].append((to, prob))

    # normalizuj (kvoli filtrovaniu len PUNCT_SET)
    for fr in transitions:
        total = sum(p for _, p in transitions[fr])
        if total > 0:
            transitions[fr] = [(to, p / total) for to, p in transitions[fr]]

    return dict(transitions)


def markov_generate(transitions: dict, start: str, length: int, rng: random.Random) -> list:
    """Generuje sekvenciu interpunkcie dlzky `length` z Markov retazca."""
    seq = []
    current = start
    for _ in range(length):
        if current not in transitions or not transitions[current]:
            # fallback: nahodna interpunkcia
            current = rng.choice(list(PUNCT_SET))
            seq.append(current)
            continue
        targets, probs = zip(*transitions[current])
        chosen = rng.choices(targets, weights=probs, k=1)[0]
        seq.append(chosen)
        current = chosen
    return seq


def read_tokens(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="ignore").split()


def replace_punct_with_markov(tokens: list, transitions: dict, rng: random.Random) -> list:
    """Nahradi realnu interpunkciu v tokenoch syntetickou z Markov modelu."""
    # najdi pozicie a typy interpunkcie
    punct_positions = [i for i, t in enumerate(tokens) if t in PUNCT_SET]
    if not punct_positions:
        return tokens[:]

    # prvy punct ako seed
    first_punct = tokens[punct_positions[0]]
    synth_seq = markov_generate(transitions, first_punct, len(punct_positions), rng)

    result = tokens[:]
    for i, pos in enumerate(punct_positions):
        result[pos] = synth_seq[i]
    return result


def replace_punct_with_random(tokens: list, rng: random.Random) -> list:
    """Nahradi interpunkciu kompletne nahodne (uniform)."""
    punct_list = list(PUNCT_SET)
    result = tokens[:]
    for i, t in enumerate(result):
        if t in PUNCT_SET:
            result[i] = rng.choice(punct_list)
    return result


def tokens_to_graph(tokens: list) -> nx.Graph:
    """Vytvori word-adjacency graf z token listu."""
    G = nx.Graph()
    for i in range(len(tokens) - 1):
        u, v = tokens[i].lower(), tokens[i + 1].lower()
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    return G


def graph_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n == 0:
        return {"n_nodes": 0, "n_edges": 0, "avg_degree": 0,
                "avg_clustering": 0, "density": 0}

    avg_deg = 2 * e / n
    avg_cl = nx.average_clustering(G)
    density = nx.density(G)

    return {
        "n_nodes": n, "n_edges": e,
        "avg_degree": float(avg_deg),
        "avg_clustering": float(avg_cl),
        "density": float(density),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--markov-csv", default="markov_out_core/markov_conditional_probs_long.csv")
    ap.add_argument("--out-dir", default="markov_synth_out_core")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--max-books", type=int, default=20, help="Max knih na jazyk (kvoli rychlosti)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token_dir = Path(args.token_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_rows = []

    for lang in langs:
        print(f"\n{'='*50}")
        print(f"  {lang.upper()}")
        print(f"{'='*50}")

        # load Markov matrix
        transitions = load_markov_matrix(Path(args.markov_csv), lang)
        if not transitions:
            print(f"  [WARN] Ziadne transitions pre {lang}")
            continue

        lang_dir = token_dir / lang
        if not lang_dir.exists():
            print(f"  [WARN] Missing {lang_dir}")
            continue

        files = sorted(lang_dir.glob("*.txt"))[:args.max_books]
        print(f"  Spracuvam {len(files)} knih...")

        for fi, fpath in enumerate(files):
            rng = random.Random(args.seed + fi)
            tokens = read_tokens(fpath)
            gid = fpath.stem

            # 1) Real
            G_real = tokens_to_graph(tokens)
            m_real = graph_metrics(G_real)

            # 2) Markov synthetic
            tokens_markov = replace_punct_with_markov(tokens, transitions, rng)
            G_markov = tokens_to_graph(tokens_markov)
            m_markov = graph_metrics(G_markov)

            # 3) Random punct
            tokens_rand = replace_punct_with_random(tokens, rng)
            G_rand = tokens_to_graph(tokens_rand)
            m_rand = graph_metrics(G_rand)

            for kind, met in [("real", m_real), ("markov", m_markov), ("random", m_rand)]:
                row = {"language": lang, "gid": gid, "kind": kind}
                row.update(met)
                all_rows.append(row)

            if (fi + 1) % 10 == 0:
                print(f"    {fi+1}/{len(files)} done")

        print(f"  [OK] {lang}: {len(files)} knih spracovanych")

    # ── Save results ──
    out_csv = out_dir / "markov_synth_metrics.csv"
    fields = ["language", "gid", "kind", "n_nodes", "n_edges",
              "avg_degree", "avg_clustering", "density"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n[OK] Metrics saved: {out_csv}")

    # ── Summary per language and kind ──
    import pandas as pd
    df = pd.DataFrame(all_rows)

    summary = df.groupby(["language", "kind"]).agg(
        n_books=("gid", "count"),
        avg_degree_median=("avg_degree", "median"),
        avg_clustering_median=("avg_clustering", "median"),
        n_nodes_median=("n_nodes", "median"),
        n_edges_median=("n_edges", "median"),
    ).reset_index()
    summary_csv = out_dir / "markov_synth_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] Summary: {summary_csv}")

    # ── Plots ──
    metrics_to_plot = [
        ("avg_clustering", "Avg clustering coefficient"),
        ("avg_degree", "Avg degree"),
        ("n_edges", "Number of edges"),
    ]

    for metric, label in metrics_to_plot:
        fig, axes = plt.subplots(1, len(langs), figsize=(4 * len(langs), 4.5), sharey=True)
        if len(langs) == 1:
            axes = [axes]

        for ax, lang in zip(axes, langs):
            lang_df = df[df["language"] == lang]
            kinds = ["real", "markov", "random"]
            colors = {"real": "C0", "markov": "C2", "random": "C3"}

            data_per_kind = [lang_df[lang_df["kind"] == k][metric].values for k in kinds]
            bp = ax.boxplot(data_per_kind, labels=kinds, patch_artist=True)
            for patch, kind in zip(bp["boxes"], kinds):
                patch.set_facecolor(colors[kind])
                patch.set_alpha(0.7)

            ax.set_title(f"{lang.upper()}", fontsize=11)
            ax.set_ylabel(label if ax == axes[0] else "")

        fig.suptitle(f"{label}: Real vs Markov-synthetic vs Random", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"compare_{metric}.png", dpi=300)
        plt.close(fig)

    print(f"[OK] Plots saved to {out_dir / 'plots'}")
    print("Hotovo!")


if __name__ == "__main__":
    main()
