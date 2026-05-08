#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

def iter_bigrams(tokens: Iterable[str]):
    prev = None
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if prev is not None:
            yield prev, t
        prev = t

def read_tokens_file(path: Path):
    # token files are typically whitespace-separated
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.split()

def write_edge_csv(path: Path, counter: Counter, min_weight: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "weight"])
        for (src, dst), wt in counter.items():
            if wt >= min_weight:
                w.writerow([src, dst, wt])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-root", default="token_out_core/tokens_words_plus_punct",
                    help="Root folder with tokenized files per language")
    ap.add_argument("--out", default="network_out_core",
                    help="Output root")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--per-book", action="store_true",
                    help="Also write per-book edge CSVs (bigger output)")
    ap.add_argument("--min-weight", type=int, default=1,
                    help="Filter edges with weight < min_weight")
    ap.add_argument("--max-books", type=int, default=0,
                    help="For quick tests: limit number of books per language (0=all)")
    args = ap.parse_args()

    token_root = Path(args.token_root)
    out_root = Path(args.out)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    edges_lang_dir = out_root / "edges"
    edges_book_dir = out_root / "edges_per_book"

    total_books = 0

    for lang in langs:
        lang_dir = token_root / lang
        if not lang_dir.exists():
            print(f"[WARN] Missing lang dir: {lang_dir}")
            continue

        agg = Counter()
        book_files = sorted(lang_dir.glob("*.txt"))
        if args.max_books and args.max_books > 0:
            book_files = book_files[: args.max_books]

        for fp in book_files:
            tokens = read_tokens_file(fp)
            c = Counter(iter_bigrams(tokens))
            agg.update(c)

            if args.per_book:
                out_book = edges_book_dir / lang / f"{fp.stem}.csv"
                write_edge_csv(out_book, c, min_weight=args.min_weight)

            total_books += 1

        out_lang = edges_lang_dir / f"edges_{lang}.csv"
        write_edge_csv(out_lang, agg, min_weight=args.min_weight)
        print(f"[OK] {lang}: books={len(book_files)} edges_written={out_lang}")

    print(f"Done. Total processed books={total_books}")
    print(f"Edges in: {out_root.resolve()}")

if __name__ == "__main__":
    main()