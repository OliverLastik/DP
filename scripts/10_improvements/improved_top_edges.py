#!/usr/bin/env python3
"""
Vylepseny top edges report.

Rozlisuje typy hran:
  - punct -> word (interpunkcia na zaciatku vety/frazy)
  - word -> punct (slovo pred interpunkciou)
  - word -> word (bezna kolokacia)
  - punct -> punct (dvojita interpunkcia napr. . .)

Generuje:
  - Top edges CSV s typmi a percentami
  - Vizualizacia: bar chart top hran s farebnymi typmi
  - Per-book analyza: kolko percent top hran obsahuje interpunkciu
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PUNCT_SET = {".", ",", "!", "?", ";", ":"}
LANG_COLORS = {"en": "#1f77b4", "de": "#ff7f0e", "nl": "#2ca02c"}


def classify_edge(src, dst):
    s_punct = src in PUNCT_SET
    d_punct = dst in PUNCT_SET
    if s_punct and d_punct:
        return "punct>punct"
    elif s_punct:
        return "punct>word"
    elif d_punct:
        return "word>punct"
    else:
        return "word>word"


EDGE_COLORS = {
    "punct>word": "#d62728",
    "word>punct": "#ff7f0e",
    "punct>punct": "#9467bd",
    "word>word": "#1f77b4",
}


def load_edges(csv_path: Path, top_n=30):
    """Load edge list CSV and return top N by weight."""
    df = pd.read_csv(csv_path)

    # Auto-detect columns
    cols = df.columns.tolist()
    src_col = next((c for c in cols if c.lower() in ("src", "source", "u", "from")), cols[0])
    dst_col = next((c for c in cols if c.lower() in ("dst", "target", "v", "to")), cols[1])
    w_col = next((c for c in cols if c.lower() in ("weight", "w", "count")), cols[2])

    df = df.sort_values(w_col, ascending=False).head(top_n)

    edges = []
    for _, row in df.iterrows():
        src = str(row[src_col])
        dst = str(row[dst_col])
        w = float(row[w_col])
        edges.append({
            "src": src, "dst": dst, "weight": w,
            "type": classify_edge(src, dst),
            "label": f"{src} > {dst}",
        })
    return edges


def load_edges_perbook(token_dir: Path, lang: str, top_n=20):
    """Per-book: pocet top hran ktore obsahuju interpunkciu."""
    from collections import Counter

    lang_dir = token_dir / lang
    if not lang_dir.exists():
        return []

    files = sorted(lang_dir.glob("*.txt"))
    results = []

    for fi, fp in enumerate(files):
        text = fp.read_text(encoding="utf-8", errors="replace")
        toks = text.lower().split()

        edge_counts = Counter()
        for i in range(len(toks) - 1):
            edge_counts[(toks[i], toks[i + 1])] += 1

        top_edges = edge_counts.most_common(top_n)
        n_with_punct = sum(1 for (s, d), _ in top_edges
                           if s in PUNCT_SET or d in PUNCT_SET)

        results.append({
            "gid": fp.stem,
            "n_top_edges": top_n,
            "n_with_punct": n_with_punct,
            "pct_with_punct": n_with_punct / top_n * 100 if top_n > 0 else 0,
        })

        if (fi + 1) % 20 == 0:
            print(f"  {lang}: {fi+1}/{len(files)} books")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", default="network_out_core/edges")
    ap.add_argument("--token-dir", default="token_out_core/tokens_words_plus_punct")
    ap.add_argument("--out-dir", default="improved_plots/top_edges")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--perbook-top", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_summary = []
    all_perbook = []

    for lang in langs:
        print(f"\n[INFO] {lang.upper()}")
        print("-" * 40)

        # ── Load and classify edges ──
        edges_path = Path(args.edges_dir) / f"edges_{lang}.csv"
        if not edges_path.exists():
            print(f"  [WARN] missing {edges_path}")
            continue

        edges = load_edges(edges_path, top_n=args.top_n)

        # ── Save classified CSV ──
        csv_out = out_dir / f"top_edges_classified_{lang}.csv"
        with csv_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "src", "dst", "weight", "type", "label"])
            w.writeheader()
            for i, e in enumerate(edges):
                e["rank"] = i + 1
                w.writerow(e)

        # ── Type statistics ──
        types = [e["type"] for e in edges]
        type_counts = {t: types.count(t) for t in EDGE_COLORS.keys()}
        total = len(types)
        pct_with_punct = sum(v for k, v in type_counts.items() if k != "word>word") / total * 100

        print(f"  Top {args.top_n} edges:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c} ({c/total*100:.1f}%)")
        print(f"  Edges with punct: {pct_with_punct:.1f}%")

        all_summary.append({
            "lang": lang,
            "n_top": args.top_n,
            **{f"n_{t.replace('>', '_')}": c for t, c in type_counts.items()},
            "pct_with_punct": pct_with_punct,
        })

        # ── Bar chart: top edges with types ──
        fig, ax = plt.subplots(figsize=(12, 6))

        labels = [e["label"] for e in edges]
        weights = [e["weight"] for e in edges]
        colors = [EDGE_COLORS[e["type"]] for e in edges]

        bars = ax.barh(range(len(edges) - 1, -1, -1), weights, color=colors, alpha=0.8)

        ax.set_yticks(range(len(edges) - 1, -1, -1))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Weight (frequency)", fontsize=11)
        ax.set_title(f"Top {args.top_n} edges — {lang.upper()}\n"
                     f"({pct_with_punct:.0f}% involve punctuation)", fontsize=12)

        # legenda
        legend_handles = [mpatches.Patch(facecolor=c, label=t)
                          for t, c in EDGE_COLORS.items()]
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

        fig.tight_layout()
        fig.savefig(out_dir / f"top_edges_bar_{lang}.png", dpi=300)
        plt.close(fig)
        print(f"  [OK] top_edges_bar_{lang}.png")

        # ── Per-book analysis ──
        print(f"  Per-book analysis (top {args.perbook_top} edges per book)...")
        perbook = load_edges_perbook(Path(args.token_dir), lang, top_n=args.perbook_top)
        for pb in perbook:
            pb["lang"] = lang
        all_perbook.extend(perbook)

    # ── Per-book boxplot ──
    if all_perbook:
        df_pb = pd.DataFrame(all_perbook)
        fig, ax = plt.subplots(figsize=(6, 4.5))

        data = [df_pb[df_pb["lang"] == l]["pct_with_punct"].values for l in langs]
        bp = ax.boxplot(data, tick_labels=[l.upper() for l in langs], patch_artist=True)
        for patch, l in zip(bp["boxes"], langs):
            patch.set_facecolor(LANG_COLORS[l])
            patch.set_alpha(0.6)

        ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel(f"% of top-{args.perbook_top} edges with punctuation", fontsize=11)
        ax.set_title("Per-book: punctuation in top edges", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "perbook_punct_in_top_edges.png", dpi=300)
        plt.close(fig)
        print(f"\n[OK] perbook_punct_in_top_edges.png")

        # Save per-book CSV
        df_pb.to_csv(out_dir / "perbook_top_edges.csv", index=False)

    # ── Summary CSV ──
    if all_summary:
        df_s = pd.DataFrame(all_summary)
        df_s.to_csv(out_dir / "top_edges_summary.csv", index=False)
        print(f"[OK] top_edges_summary.csv")

    # ── Cross-language comparison: edge type distribution ──
    if all_summary:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(langs))
        width = 0.2
        edge_types = ["word>word", "word>punct", "punct>word", "punct>punct"]

        for i, etype in enumerate(edge_types):
            key = f"n_{etype.replace('>', '_')}"
            vals = [s.get(key, 0) for s in all_summary]
            ax.bar(x + (i - 1.5) * width, vals, width,
                   color=EDGE_COLORS[etype], label=etype, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in langs], fontsize=12)
        ax.set_ylabel(f"Count (of top {args.top_n})", fontsize=11)
        ax.set_title("Edge type distribution in top edges", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "edge_type_distribution.png", dpi=300)
        plt.close(fig)
        print(f"[OK] edge_type_distribution.png")

    print("\n[OK] Top edges analyza hotova.")


if __name__ == "__main__":
    main()
