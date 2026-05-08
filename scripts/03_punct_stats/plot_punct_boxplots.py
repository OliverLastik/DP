#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SAFE_NAMES = {
    ".": "dot",
    ",": "comma",
    ";": "semicolon",
    ":": "colon",
    "?": "question",
    "!": "exclamation",
    '"': "quote",
    "'": "apostrophe",
    "…": "ellipsis",
    "—": "emdash",
    "-": "dash",
    "(": "lparen",
    ")": "rparen",
    "[": "lbracket",
    "]": "rbracket",
}

def safe_filename_for_punct(p: str) -> str:
    if p in SAFE_NAMES:
        return SAFE_NAMES[p]
    return f"u{ord(p)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", default="punct_out_core/punct_per_book_tidy.csv")
    ap.add_argument("--outdir", default="punct_out_core/plots")
    args = ap.parse_args()

    df = pd.read_csv(args.tidy)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # tu to máš natvrdo: bodka, čiarka, bodkočiarka, dvojbodka, otáznik, výkričník
    puncts = [".", ",", ";", ":", "?", "!"]

    for p in puncts:
        sub = df[df["punct"] == p].copy()
        if sub.empty:
            continue

        langs = ["en", "de", "nl"]
        data = [sub[sub["language"] == lang]["per1k"].to_numpy() for lang in langs]

        plt.figure()
        plt.boxplot(data, tick_labels=langs)  # labels -> tick_labels (matplotlib 3.9+)
        plt.title(f"Per-1000-words frequency of '{p}' (per book)")
        plt.ylabel("count per 1000 words")
        plt.xlabel("language")
        plt.tight_layout()

        safe_name = safe_filename_for_punct(p)
        out_path = outdir / f"boxplot_per1k_{safe_name}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

    print(f"Saved plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
