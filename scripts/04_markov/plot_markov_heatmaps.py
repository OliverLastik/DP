#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="markov_out_core/markov_conditional_probs_long.csv")
    ap.add_argument("--outdir", default="markov_out_core/plots")
    args = ap.parse_args()

    df = pd.read_csv(args.cond)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for lang, g in df.groupby("language"):
        g = g.copy()
        valid_from = g.groupby("from")["from_count"].max()
        valid_from = valid_from[valid_from > 0].index.tolist()
        g = g[g["from"].isin(valid_from)]

        m = g.pivot_table(index="from", columns="to", values="prob", aggfunc="mean", fill_value=0.0)

        plt.figure(figsize=(10, 7))
        plt.imshow(m.to_numpy(), aspect="auto", interpolation="nearest")
        plt.xticks(range(len(m.columns)), m.columns, rotation=90)
        plt.yticks(range(len(m.index)), m.index)
        plt.title(f"Markov transition matrix P(to|from) – {lang}")
        plt.colorbar(label="probability")
        plt.tight_layout()
        plt.savefig(outdir / f"markov_heatmap_{lang}.png", dpi=200)
        plt.close()

    print(f"Saved heatmaps to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
