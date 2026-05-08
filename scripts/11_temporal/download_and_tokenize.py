#!/usr/bin/env python3
"""
Stiahne texty konkretnych knih z Project Gutenberg a tokenizuje ich.
Pouzitie pre temporalnu analyzu - ten isty autor, rozne obdobia.
"""
import argparse
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError


PUNCT_SET = {".", ",", "!", "?", ";", ":"}


def download_gutenberg(pg_id: int) -> str:
    """Stiahne text z Project Gutenberg."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}-0.txt",
        f"https://www.gutenberg.org/files/{pg_id}/{pg_id}.txt",
    ]
    for url in urls:
        try:
            print(f"  Trying {url}...")
            resp = urlopen(url, timeout=30)
            raw = resp.read()
            # try utf-8 first, then latin-1
            for enc in ("utf-8", "latin-1"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
        except (HTTPError, Exception) as e:
            continue
    raise RuntimeError(f"Cannot download PG {pg_id}")


def strip_gutenberg_header_footer(text: str) -> str:
    """Odstrani Project Gutenberg header a footer."""
    # Start markers
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # find next newline after marker
            nl = text.find("\n", idx)
            if nl != -1:
                start_idx = nl + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def tokenize_words_plus_punct(text: str) -> list:
    """
    Tokenizuje text na slova + interpunkcne znacky.
    Rovnaka logika ako v hlavnom pipeline.
    """
    # normalize unicode
    text = unicodedata.normalize("NFC", text)

    # lowercase
    text = text.lower()

    tokens = []
    # split on whitespace, then separate punct from words
    for raw_token in text.split():
        # strip leading/trailing non-alpha non-punct
        raw_token = raw_token.strip()
        if not raw_token:
            continue

        # extract trailing punctuation
        buf = []
        i = 0
        word_chars = []

        for ch in raw_token:
            if ch in PUNCT_SET:
                if word_chars:
                    w = "".join(word_chars)
                    if w:
                        buf.append(w)
                    word_chars = []
                buf.append(ch)
            else:
                word_chars.append(ch)

        if word_chars:
            w = "".join(word_chars)
            if w:
                buf.append(w)

        tokens.extend(buf)

    return tokens


# 3 autori, 3 obdobia kazdy.  Dickens (en), Fontane (de), Couperus (nl)
# su realistickí spoločenskí romanopisci s dlhou kariérou a velkomestskym nastavením.
CORPUS = [
    # en -- Charles Dickens (1812-1870)
    {"lang": "en", "id": 580,   "author": "dickens",  "title": "Pickwick Papers",    "year": 1836, "period": "early"},
    {"lang": "en", "id": 766,   "author": "dickens",  "title": "David Copperfield",  "year": 1850, "period": "middle"},
    {"lang": "en", "id": 883,   "author": "dickens",  "title": "Our Mutual Friend",  "year": 1865, "period": "late"},
    # de -- Theodor Fontane (1819-1898)
    {"lang": "de", "id": 52912, "author": "fontane",  "title": "L'Adultera",         "year": 1882, "period": "early"},
    {"lang": "de", "id": 5323,  "author": "fontane",  "title": "Effi Briest",        "year": 1895, "period": "middle"},
    {"lang": "de", "id": 53628, "author": "fontane",  "title": "Der Stechlin",       "year": 1898, "period": "late"},
    # nl -- Louis Couperus (1863-1923)
    {"lang": "nl", "id": 19563, "author": "couperus", "title": "Eline Vere",         "year": 1889, "period": "early"},
    {"lang": "nl", "id": 44084, "author": "couperus", "title": "Langs lijnen",       "year": 1900, "period": "middle"},
    {"lang": "nl", "id": 29814, "author": "couperus", "title": "De komedianten",     "year": 1917, "period": "late"},
]

# Couperus validation set -- 2 extra knihy na test ci je late-period Weibull
# kolaps temporalny (ciste kariera) alebo ziskovy (klasicke vs realisticke).
CORPUS_COUPERUS_VAL = [
    # Koloniálny realizmus, rovnaky rok ako Langs lijnen (kontrola middle-realist)
    {"lang": "nl", "id": 67219, "author": "couperus", "title": "De stille kracht",   "year": 1900, "period": "midreal"},
    # Neskora klasicko-satirická, rok po De komedianten (kontrola late-fantas)
    {"lang": "nl", "id": 29837, "author": "couperus", "title": "De verliefde ezel",  "year": 1918, "period": "latefan"},
]


def process_book(book, out_dir: Path) -> None:
    lang = book["lang"]
    author = book["author"]
    pg_id = book["id"]
    label = f"{lang}_{author}_{book['period']}_{book['year']}_{book['title'].replace(' ', '_')}"
    print(f"\n[INFO] {label} (PG {pg_id})")

    folder_suffix = book.get("folder_suffix", "")
    subdir_name = f"{lang}_{author}{folder_suffix}"
    subdir_tokens = out_dir / "tokens" / subdir_name
    subdir_raw = out_dir / "raw" / subdir_name
    subdir_tokens.mkdir(parents=True, exist_ok=True)
    subdir_raw.mkdir(parents=True, exist_ok=True)

    raw_path = subdir_raw / f"{label}.txt"
    if raw_path.exists():
        print(f"  Already downloaded: {raw_path}")
        text = raw_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = download_gutenberg(pg_id)
        raw_path.write_text(text, encoding="utf-8")
        print(f"  Downloaded: {len(text)} chars")

    text = strip_gutenberg_header_footer(text)
    print(f"  Stripped text: {len(text)} chars")

    tokens = tokenize_words_plus_punct(text)
    n_punct = sum(1 for t in tokens if t in PUNCT_SET)
    print(f"  Tokens: {len(tokens)}, punct {n_punct} ({n_punct/len(tokens)*100:.1f}%)")

    tok_path = subdir_tokens / f"{label}.txt"
    tok_path.write_text(" ".join(tokens), encoding="utf-8")
    print(f"  [OK] Saved: {tok_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="temporal_data")
    ap.add_argument("--only-lang", default=None, help="Only download a given language (en/de/nl)")
    ap.add_argument("--couperus-validation", action="store_true",
                    help="Download only the 2 Couperus validation books (separate folder)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.couperus_validation:
        books = [{**b, "folder_suffix": "_validation"} for b in CORPUS_COUPERUS_VAL]
    else:
        books = CORPUS
        if args.only_lang:
            books = [b for b in books if b["lang"] == args.only_lang]
            if not books:
                print(f"[ERR] No books for lang={args.only_lang}")
                return

    for book in books:
        try:
            process_book(book, out_dir)
        except Exception as e:
            print(f"  [ERR] {book['lang']}/{book['author']}/{book['title']}: {e}")

    print("\n[OK] Hotovo.")


if __name__ == "__main__":
    main()
