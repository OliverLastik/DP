#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Optional, Set, Tuple

WORD_RE = re.compile(r"\\w+", flags=re.UNICODE)

def cut_text_to_n_words(text: str, n_words: int) -> Tuple[str, int]:
    """
    Vráti substring textu, ktorý obsahuje prvých n_words slov (počítané regex \w+),
    pričom zachováva všetky znaky (interpunkciu) až po koniec n-teho slova.
    """
    count = 0
    last_end = 0
    for m in WORD_RE.finditer(text):
        count += 1
        last_end = m.end()
        if count >= n_words:
            break
    if count == 0:
        return "", 0
    if count < n_words:
        # text je kratší než n_words
        return text.strip(), count
    return text[:last_end].strip(), count

def load_accepted_ids(accepted_csv: Path, langs: Set[str]) -> Set[Tuple[str, str]]:
    """
    Vráti set (lang, id_str) pre accepted knihy, len pre vybrané jazyky.
    Podporuje accepted.csv z download_corpus.py.
    """
    out = set()
    if not accepted_csv.exists():
        return out

    import csv
    with accepted_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lang = (row.get("language") or "").strip()
            gid = (row.get("gutenberg_id") or "").strip()
            if lang in langs and gid:
                out.add((lang, gid))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_core/clean", help="Zdroj (clean) priečinok")
    ap.add_argument("--out", default="data_core/windows30k", help="Výstupný priečinok pre okná")
    ap.add_argument("--words", type=int, default=30000, help="Veľkosť okna v slovách")
    ap.add_argument("--langs", default="en,de,nl", help="Jazyky, napr. en,de,nl")
    ap.add_argument("--accepted-csv", default="data_core/logs/accepted.csv",
                    help="Ak existuje, spracuje iba accepted knihy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    langs = {x.strip() for x in args.langs.split(",") if x.strip()}

    accepted_csv = Path(args.accepted_csv)
    accepted = load_accepted_ids(accepted_csv, langs)
    use_accepted = len(accepted) > 0

    out.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for lang in sorted(langs):
        lang_src = src / lang
        if not lang_src.exists():
            continue

        lang_out = out / lang
        lang_out.mkdir(parents=True, exist_ok=True)

        for f in lang_src.glob("*.txt"):
            gid = f.stem  # "12345"
            if use_accepted and (lang, gid) not in accepted:
                continue

            out_path = lang_out / f"{gid}.txt"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            text = f.read_text(encoding="utf-8", errors="ignore")
            window, wc = cut_text_to_n_words(text, args.words)

            # ak je to extrémne krátke, preskoč (nemalo by sa stať pri accepted)
            if wc < max(1000, args.words // 10):
                continue

            out_path.write_text(window, encoding="utf-8", errors="ignore")
            processed += 1

    print(f"Done. Processed={processed}, skipped={skipped}")
    print(f"Windows in: {out.resolve()}")

if __name__ == "__main__":
    main()
