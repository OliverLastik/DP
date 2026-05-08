#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def bootstrap_ci(x, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    med = float(np.median(x))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        boots.append(np.median(samp))
    boots = np.array(boots)
    return (med, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="markov_out_core2/per_book_distance_to_language_markov.csv")
    ap.add_argument("--outdir", default="markov_out_core2")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    rows = []
    for lang, g in df.groupby("language"):
        med, lo, hi = bootstrap_ci(
            g["distance_to_lang_aggregate_rowwise_jsd"].to_numpy(),
            n_boot=args.boot
        )
        rows.append({
            "language": lang,
            "n_books": len(g),
            "median_distance": med,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "min": float(np.min(g["distance_to_lang_aggregate_rowwise_jsd"])),
            "max": float(np.max(g["distance_to_lang_aggregate_rowwise_jsd"])),
        })
    out = pd.DataFrame(rows).sort_values("language")
    out_path = Path(args.outdir) / "markov_stability_summary.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
