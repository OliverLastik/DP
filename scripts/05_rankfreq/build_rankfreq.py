#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter
import math
import csv

import matplotlib.pyplot as plt


def iter_tokens_from_file(p: Path):
    # tokeny sú oddelené medzerami
    # (v prípade, že by si mal aj prázdne, vyfiltrujeme)d
    for tok in p.read_text(encoding="utf-8", errors="ignore").split():
        tok = tok.strip()
        if tok:
            yield tok


def build_rankfreq_for_lang(lang_dir: Path) -> Counter:
    c = Counter()
    for f in lang_dir.glob("*.txt"):
        c.update(iter_tokens_from_file(f))
    return c


def write_rankfreq_csv(counter: Counter, out_csv: Path, lang: str):
    # zoradenie: najčastejšie prvé
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    total = sum(counter.values())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["language", "rank", "token", "count", "freq", "log_rank", "log_freq"])
        for i, (tok, cnt) in enumerate(items, start=1):
            freq = cnt / total if total else 0.0
            log_rank = math.log10(i)
            log_freq = math.log10(freq) if freq > 0 else float("-inf")
            w.writerow([lang, i, tok, cnt, freq, log_rank, log_freq])


def plot_rankfreq(counter: Counter, out_png: Path, title: str, max_points: int = 20000):
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    total = sum(counter.values())
    if total == 0:
        return

    ranks = []
    freqs = []

    for i, (_, cnt) in enumerate(items, start=1):
        if i > max_points:
            break
        ranks.append(i)
        freqs.append(cnt / total)

    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(ranks, freqs, linewidth=1)

    plt.xlabel("rank")
    plt.ylabel("frequency")
    plt.title(title)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-root", default="token_out_core", help="Root výstupu z tokenize_windows_v2.py")
    ap.add_argument("--out", default="rankfreq_out_core", help="Výstupný priečinok")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--max-points", type=int, default=20000)
    args = ap.parse_args()

    token_root = Path(args.token_root)
    out_root = Path(args.out)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    modes = [
        ("tokens_words", "words"),
        ("tokens_words_plus_punct", "words_plus_punct"),
    ]

    for mode_dir, mode_name in modes:
        base = token_root / mode_dir
        if not base.exists():
            print(f"[WARN] missing: {base}")
            continue

        for lang in langs:
            lang_dir = base / lang
            if not lang_dir.exists():
                print(f"[WARN] missing lang dir: {lang_dir}")
                continue

            print(f"Building rankfreq: mode={mode_name}, lang={lang}")
            c = build_rankfreq_for_lang(lang_dir)

            # CSV
            out_csv = out_root / "csv" / f"rankfreq_{mode_name}_{lang}.csv"
            write_rankfreq_csv(c, out_csv, lang)

            # Plot
            out_png = out_root / "plots" / f"rankfreq_{mode_name}_{lang}.png"
            plot_rankfreq(
                c,
                out_png,
                title=f"Rank–frequency ({mode_name}) — {lang}",
                max_points=args.max_points
            )

    print(f"Done. Output in: {out_root.resolve()}")


if __name__ == "__main__":
    main()