#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence (log base 2). p,q musia byť nezáporné."""
    p = p.astype(float)
    q = q.astype(float)
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
    ap.add_argument("--transitions", default="punct_transitions_core.csv", help="CSV z extract_punct_stats.py")
    ap.add_argument("--outdir", default="markov_out_core", help="Výstupný priečinok")
    ap.add_argument("--topk", type=int, default=15, help="Top K prechodov na jazyk")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transitions, dtype={"language": str, "gutenberg_id": str, "from": str, "to": str, "count": int})
    # agregácia cez všetky knihy v jazyku
    agg = df.groupby(["language", "from", "to"], as_index=False)["count"].sum()

    # --- 1) Joint distribúcia P(from,to) na jazyk ---
    # pivot per language -> vektor hrán
    agg["edge"] = agg["from"] + "->" + agg["to"]
    edges = sorted(agg["edge"].unique())

    joint = (
        agg.pivot_table(index="language", columns="edge", values="count", aggfunc="sum", fill_value=0)
        .reindex(columns=edges, fill_value=0)
        .astype(float)
    )

    # pravdepodobnosti
    joint_prob = joint.div(joint.sum(axis=1), axis=0).fillna(0.0)

    # --- 2) Conditional (Markov) P(to|from) na jazyk ---
    cond_rows = []
    for lang, g in agg.groupby("language"):
        # matica counts (from x to)
        m = g.pivot_table(index="from", columns="to", values="count", aggfunc="sum", fill_value=0).astype(float)
        row_sums = m.sum(axis=1).replace(0, np.nan)
        p = m.div(row_sums, axis=0).fillna(0.0)

        # do long form
        p_long = p.stack().reset_index()
        p_long.columns = ["from", "to", "prob"]
        # pridaj aj counts a váhy riadkov (koľko prechodov z "from")
        counts_from = m.sum(axis=1).rename("from_count").reset_index()
        p_long = p_long.merge(counts_from, on="from", how="left")
        p_long.insert(0, "language", lang)
        cond_rows.append(p_long)

    cond_long = pd.concat(cond_rows, ignore_index=True)

    # --- 3) Top prechody na jazyk (podľa joint prob alebo count) ---
    top = (
        agg.assign(joint_prob=agg["count"] / agg.groupby("language")["count"].transform("sum"))
           .sort_values(["language", "joint_prob"], ascending=[True, False])
           .groupby("language", as_index=False)
           .head(args.topk)
           .reset_index(drop=True)
    )

    # --- 4) Distance matrix medzi jazykmi (JSD na joint distribúcii) ---
    langs = list(joint_prob.index)
    dist = pd.DataFrame(index=langs, columns=langs, dtype=float)

    for a in langs:
        for b in langs:
            dist.loc[a, b] = js_divergence(joint_prob.loc[a].to_numpy(), joint_prob.loc[b].to_numpy())

    # --- 5) Uloženie ---
    agg.to_csv(outdir / "transitions_counts_aggregated.csv", index=False, encoding="utf-8")
    joint_prob.to_csv(outdir / "joint_transition_prob_matrix.csv", encoding="utf-8")
    cond_long.to_csv(outdir / "markov_conditional_probs_long.csv", index=False, encoding="utf-8")
    top.to_csv(outdir / "top_transitions_per_language.csv", index=False, encoding="utf-8")
    dist.to_csv(outdir / "language_distance_jsd.csv", encoding="utf-8")

    print("Done.")
    print(f"Output dir: {outdir.resolve()}")
    print("Files:")
    print(" - transitions_counts_aggregated.csv")
    print(" - joint_transition_prob_matrix.csv")
    print(" - markov_conditional_probs_long.csv")
    print(" - top_transitions_per_language.csv")
    print(" - language_distance_jsd.csv")

if __name__ == "__main__":
    main()
