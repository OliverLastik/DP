#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def jsd(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return np.sum(a * np.log2(a / b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default="punct_transitions_core.csv")
    ap.add_argument("--outdir", default="markov_out_core2")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transitions, dtype={"language": str, "gutenberg_id": str, "from": str, "to": str, "count": int})

    # --- Aggregated conditional P(to|from) per language ---
    agg = df.groupby(["language", "from", "to"], as_index=False)["count"].sum()

    # all punctuation symbols appearing as states
    states_from = sorted(agg["from"].unique())
    states_to = sorted(agg["to"].unique())

    # build P(to|from) matrices per language
    P = {}
    from_counts = {}
    for lang, g in agg.groupby("language"):
        m = g.pivot_table(index="from", columns="to", values="count", aggfunc="sum", fill_value=0).reindex(index=states_from, columns=states_to, fill_value=0)
        fc = m.sum(axis=1).to_numpy()
        from_counts[lang] = fc
        p = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).to_numpy()
        P[lang] = p

    langs = sorted(P.keys())

    # --- Row-wise JSD per pair of languages ---
    rows = []
    for i, a in enumerate(langs):
        for b in langs[i+1:]:
            # per-from symbol divergence
            per_from = []
            weights = []
            for r, sym in enumerate(states_from):
                pa = P[a][r, :]
                pb = P[b][r, :]
                d = jsd(pa, pb)
                # weight by average count of that "from" in both languages
                w = 0.5 * (from_counts[a][r] + from_counts[b][r])
                per_from.append((sym, d, w))
                weights.append(w)

            W = np.array([x[2] for x in per_from], dtype=float)
            D = np.array([x[1] for x in per_from], dtype=float)
            weighted = float(np.sum(D * W) / (np.sum(W) + 1e-12))
            unweighted = float(np.mean(D))

            rows.append({"lang_a": a, "lang_b": b, "rowwise_jsd_weighted": weighted, "rowwise_jsd_mean": unweighted})

            # dump detail per pair
            det = pd.DataFrame(per_from, columns=["from_symbol", "jsd", "weight_from_count"])
            det.to_csv(outdir / f"rowwise_detail_{a}_vs_{b}.csv", index=False, encoding="utf-8")

    pd.DataFrame(rows).to_csv(outdir / "rowwise_language_distances.csv", index=False, encoding="utf-8")

    # --- Per-book conditional matrices and distances (distribution) ---
    # For each book: build conditional P(to|from) for that book
    # Then compare distributions within/between languages using mean JSD over rows
    book_rows = []
    for (lang, gid), g in df.groupby(["language", "gutenberg_id"]):
        m = g.pivot_table(index="from", columns="to", values="count", aggfunc="sum", fill_value=0).reindex(index=states_from, columns=states_to, fill_value=0)
        p = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).to_numpy()
        book_rows.append((lang, gid, p))

    # compute pairwise distances between books (sampled) would be heavy; instead:
    # compare each book to its language aggregate (rowwise JSD weighted by that book's from-counts)
    agg_cond = {}
    for lang in langs:
        agg_cond[lang] = P[lang]

    book_scores = []
    for lang, gid, p_book in book_rows:
        # weights based on book "from" counts
        # approximate weights from row sums of book matrix counts reconstructed from probs is not possible;
        # use uniform weights across from symbols present in book (row sum > 0 in original book counts)
        # We'll recover presence from transitions counts directly:
        sub = df[(df["language"] == lang) & (df["gutenberg_id"] == gid)]
        mcounts = sub.pivot_table(index="from", columns="to", values="count", aggfunc="sum", fill_value=0).reindex(index=states_from, columns=states_to, fill_value=0)
        w = mcounts.sum(axis=1).to_numpy().astype(float)

        d_rows = []
        for r in range(len(states_from)):
            d_rows.append(jsd(p_book[r, :], agg_cond[lang][r, :]))
        d_rows = np.array(d_rows, dtype=float)
        score = float(np.sum(d_rows * w) / (np.sum(w) + 1e-12))
        book_scores.append({"language": lang, "gutenberg_id": gid, "distance_to_lang_aggregate_rowwise_jsd": score})

    pd.DataFrame(book_scores).to_csv(outdir / "per_book_distance_to_language_markov.csv", index=False, encoding="utf-8")

    print("Done.")
    print(f"Output dir: {outdir.resolve()}")

if __name__ == "__main__":
    main()
