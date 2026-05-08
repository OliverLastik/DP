#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import csv

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

# mapovanie “variant” znakov
QUOTE_DOUBLE = set(['"', "“", "”", "„", "«", "»"])
QUOTE_SINGLE = set(["'", "’", "‘"])
DASH_CHARS = set(["—", "–"])  # em/en dash
ELLIPSIS_CHAR = "…"

# hyphen-minus '-' budeme rátať ako dash len keď NIE je medzi alfanumerikami (t.j. nie v slove)
HYPHEN_IN_WORD_RE = re.compile(r"(?<=\w)-(?=\w)", flags=re.UNICODE)

def normalize_for_tokens(text: str) -> str:
    # tri bodky -> …
    text = text.replace("...", ELLIPSIS_CHAR)
    return text

def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))

def is_punct_hyphen(text: str, i: int) -> bool:
    # '-' je punct, ak NIE je medzi \w-\w
    if text[i] != "-":
        return False
    left = text[i-1] if i-1 >= 0 else ""
    right = text[i+1] if i+1 < len(text) else ""
    return not (left.isalnum() and right.isalnum())

def punct_token_stream(text: str) -> List[str]:
    """
    Vráti postupnosť tokenov interpunkcie v poradí výskytu.
    Mapuje rôzne úvodzovky na " alebo '.
    Mapuje —/– na —, ... na …
    '-' berie len keď je "punct hyphen".
    """
    text = normalize_for_tokens(text)
    tokens: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]

        if ch == ELLIPSIS_CHAR:
            tokens.append("…")
        elif ch in [".", ",", ";", ":", "?", "!"]:
            tokens.append(ch)
        elif ch in DASH_CHARS:
            tokens.append("—")
        elif ch == "-" and is_punct_hyphen(text, i):
            tokens.append("-")
        elif ch in QUOTE_DOUBLE:
            tokens.append('"')
        elif ch in QUOTE_SINGLE:
            tokens.append("'")
        elif ch in ["(", ")", "[", "]"]:
            tokens.append(ch)
        # iné znaky ignorujeme
        i += 1
    return tokens

def punct_counts(text: str) -> Dict[str, int]:
    text = normalize_for_tokens(text)

    # základné počty
    counts = {k: 0 for k in [".", ",", ";", ":", "?", "!", "…", "—", "-", '"', "'", "(", ")", "[", "]"]}

    for i, ch in enumerate(text):
        if ch in [".", ",", ";", ":", "?", "!"]:
            counts[ch] += 1
        elif ch == ELLIPSIS_CHAR:
            counts["…"] += 1
        elif ch in DASH_CHARS:
            counts["—"] += 1
        elif ch == "-" and is_punct_hyphen(text, i):
            counts["-"] += 1
        elif ch in QUOTE_DOUBLE:
            counts['"'] += 1
        elif ch in QUOTE_SINGLE:
            counts["'"] += 1
        elif ch in ["(", ")", "[", "]"]:
            counts[ch] += 1

    return counts

def transitions(tokens: List[str]) -> Dict[Tuple[str, str], int]:
    tr: Dict[Tuple[str, str], int] = {}
    for a, b in zip(tokens, tokens[1:]):
        tr[(a, b)] = tr.get((a, b), 0) + 1
    return tr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data_core/windows30k", help="Priečinok s oknami")
    ap.add_argument("--out-stats", default="punct_stats.csv", help="Výstupné CSV štatistiky")
    ap.add_argument("--out-transitions", default="punct_transitions.csv", help="Výstupné CSV prechody")
    ap.add_argument("--langs", default="en,de,nl")
    args = ap.parse_args()

    win_dir = Path(args.windows)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    rows_stats = []
    rows_tr = []

    for lang in langs:
        lang_dir = win_dir / lang
        if not lang_dir.exists():
            continue

        for f in lang_dir.glob("*.txt"):
            gid = f.stem
            text = f.read_text(encoding="utf-8", errors="ignore")

            n_words = count_words(text)
            c = punct_counts(text)
            toks = punct_token_stream(text)
            tr = transitions(toks)

            # stats row
            row = {
                "language": lang,
                "gutenberg_id": gid,
                "words": n_words,
                "punct_tokens": len(toks),
            }
            # raw counts + per-1000-words
            for k, v in c.items():
                row[f"count_{k}"] = v
                row[f"per1k_{k}"] = (v / n_words * 1000.0) if n_words else 0.0

            rows_stats.append(row)

            # transitions rows
            for (a, b), v in tr.items():
                rows_tr.append({
                    "language": lang,
                    "gutenberg_id": gid,
                    "from": a,
                    "to": b,
                    "count": v
                })

    # write stats
    if rows_stats:
        fieldnames = list(rows_stats[0].keys())
        with open(args.out_stats, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_stats)

    # write transitions
    if rows_tr:
        fieldnames = ["language", "gutenberg_id", "from", "to", "count"]
        with open(args.out_transitions, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_tr)

    print("Done.")
    print(f"Stats: {Path(args.out_stats).resolve()}")
    print(f"Transitions: {Path(args.out_transitions).resolve()}")

if __name__ == "__main__":
    main()
