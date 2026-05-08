#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_rankfreq(csv_path: Path, max_rank: int | None):
    """
    Očakáva CSV so stĺpcami aspoň: rank, count
    (ak máš iné názvy, napíš mi prvý riadok head() a upravím).
    """
    df = pd.read_csv(csv_path)
    # robust: nájdi stĺpce
    cols = {c.lower(): c for c in df.columns}
    rank_col = cols.get("rank", None)
    count_col = cols.get("count", None)

    if rank_col is None or count_col is None:
        raise ValueError(
            f"{csv_path.name}: nevidím stĺpce 'rank' a 'count'. Našiel som: {list(df.columns)}"
        )

    df = df[[rank_col, count_col]].rename(columns={rank_col: "rank", count_col: "count"})
    df = df.sort_values("rank")
    if max_rank is not None:
        df = df[df["rank"] <= max_rank]
    return df


def plot_overlay(title: str, series: dict, out_path: Path):
    plt.figure(figsize=(8.8, 5.0))
    for label, df in series.items():
        plt.plot(df["rank"], df["count"], label=label, linewidth=1.2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("rank r (log)")
    plt.ylabel("frekvencia f(r) (log)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="rankfreq_out_core/csv", help="Adresár s rankfreq CSV")
    ap.add_argument("--out-dir", default="rankfreq_out_core/plots", help="Kam uložiť PNG")
    ap.add_argument("--langs", default="en,de,nl", help="Jazyky, napr. en,de,nl")
    ap.add_argument("--max-rank", type=int, default=50000, help="Orež krivku na max rank (kvôli čitateľnosti)")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    # 1) words
    words_series = {}
    for lang in langs:
        p = csv_dir / f"rankfreq_words_{lang}.csv"
        words_series[lang.upper()] = load_rankfreq(p, args.max_rank)

    plot_overlay(
        title="Rank–frequency: slová (EN/DE/NL)",
        series=words_series,
        out_path=out_dir / "rankfreq_compare_words_en_de_nl.png",
    )

    # 2) words + punct
    wpp_series = {}
    for lang in langs:
        p = csv_dir / f"rankfreq_words_plus_punct_{lang}.csv"
        wpp_series[lang.upper()] = load_rankfreq(p, args.max_rank)

    plot_overlay(
        title="Rank–frequency: slová + interpunkcia (EN/DE/NL)",
        series=wpp_series,
        out_path=out_dir / "rankfreq_compare_words_plus_punct_en_de_nl.png",
    )

    print("Done. Wrote:")
    print(" -", (out_dir / "rankfreq_compare_words_en_de_nl.png").resolve())
    print(" -", (out_dir / "rankfreq_compare_words_plus_punct_en_de_nl.png").resolve())


if __name__ == "__main__":
    main()