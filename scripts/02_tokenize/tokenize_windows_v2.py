#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Iterable, List

# slová = sekvencie písmen (bez čísel a _), funguje pre latinku aj diakritiku
WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

# interpunkcia, ktorú chceme zachovať ako tokeny
PUNCT = {".", ",", "?", "!", ";", ":"}

# nájde buď slovo, alebo 1 znak z interpunkcie vyššie
TOKEN_RE = re.compile(r"[^\W\d_]+|[.,?!;:]", flags=re.UNICODE)

def iter_text_files(src_dir: Path, langs: List[str]) -> Iterable[tuple[str, Path]]:
    for lang in langs:
        lang_dir = src_dir / lang
        if not lang_dir.exists():
            continue
        for f in sorted(lang_dir.glob("*.txt")):
            yield lang, f

def tokenize(text: str, keep_punct: bool) -> List[str]:
    if keep_punct:
        return TOKEN_RE.findall(text)
    return WORD_RE.findall(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_core/windows30k", help="Vstup: okná 30k")
    ap.add_argument("--out", default="token_out_core", help="Výstupný priečinok")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    out_words = out / "tokens_words"
    out_words_punct = out / "tokens_words_plus_punct"
    for p in [out_words, out_words_punct]:
        p.mkdir(parents=True, exist_ok=True)
        for lang in langs:
            (p / lang).mkdir(parents=True, exist_ok=True)

    processed = 0
    for lang, f in iter_text_files(src, langs):
        gid = f.stem
        text = f.read_text(encoding="utf-8", errors="ignore")

        # 1) len slová
        dst1 = out_words / lang / f"{gid}.txt"
        if args.overwrite or not dst1.exists():
            toks1 = tokenize(text, keep_punct=False)
            dst1.write_text(" ".join(toks1), encoding="utf-8")
        # 2) slová + interpunkcia
        dst2 = out_words_punct / lang / f"{gid}.txt"
        if args.overwrite or not dst2.exists():
            toks2 = tokenize(text, keep_punct=True)
            dst2.write_text(" ".join(toks2), encoding="utf-8")

        processed += 1

    print(f"Done. Tokenized books={processed}")
    print(f"Tokens in: {out.resolve()}")

if __name__ == "__main__":
    main()