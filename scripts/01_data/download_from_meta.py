#!/usr/bin/env python3
import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://www.gutenberg.org"

HEADERS = {
    "User-Agent": "DP-Punctuation-Downloader/1.0 (educational use)"
}

HEADER_START = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
HEADER_END   = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)

def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.splitlines()
    start_idx = 0
    for i, ln in enumerate(lines[:500]):
        if HEADER_START.search(ln):
            start_idx = i + 1
            break

    end_idx = len(lines)
    for j in range(len(lines) - 1, max(len(lines) - 4000, start_idx), -1):
        if HEADER_END.search(lines[j]):
            end_idx = j
            break

    body = "\n".join(lines[start_idx:end_idx]).strip()
    return body

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))

def fetch_ebook_page(gid: int) -> str:
    url = f"{BASE}/ebooks/{gid}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def pick_plaintext_utf8_url(ebook_html: str) -> str | None:
    # veľmi jednoduché hľadanie: na stránke je link na "Plain Text UTF-8"
    # (nescrapujeme cez BS4, aby sme držali deps minimálne)
    # Hľadáme href pri "Plain Text UTF-8"
    # Väčšinou vyzerá: /files/<id>/<id>-0.txt alebo /cache/epub/<id>/pg<id>.txt
    m = re.search(r'href="([^"]+\.txt[^"]*)".{0,200}?Plain Text UTF-8', ebook_html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        href = m.group(1)
        if href.startswith("http"):
            return href
        return BASE + href

    # fallback: ak nie je "UTF-8", skús "Plain Text"
    m2 = re.search(r'href="([^"]+\.txt[^"]*)".{0,200}?Plain Text', ebook_html, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        href = m2.group(1)
        if href.startswith("http"):
            return href
        return BASE + href

    return None

def download_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    # Gutenberg txt býva väčšinou UTF-8; keď nie, necháme requests hádať a fallbackneme
    r.encoding = r.encoding or "utf-8"
    return r.text

def main(meta_path: str, out_dir: str, sleep: float, min_words: int, overwrite: bool):
    meta_path = Path(meta_path)
    out_dir = Path(out_dir)

    df = pd.read_csv(meta_path, dtype=str, encoding="utf-8")
    if "Text#" not in df.columns or "Language" not in df.columns:
        raise ValueError(f"meta musí obsahovať stĺpce Text# a Language. Našiel som: {list(df.columns)}")

    # pripraviť foldery
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "clean").mkdir(parents=True, exist_ok=True)

    # log súbor pre neúspešné downloady
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        gid = int(row["Text#"])
        lang = row["Language"]

        raw_path = out_dir / "raw" / lang / f"{gid}.txt"
        clean_path = out_dir / "clean" / lang / f"{gid}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.parent.mkdir(parents=True, exist_ok=True)

        if (raw_path.exists() and clean_path.exists()) and not overwrite:
            continue

        try:
            ebook_html = fetch_ebook_page(gid)
            txt_url = pick_plaintext_utf8_url(ebook_html)
            if not txt_url:
                failed.append((gid, lang, "no_txt_url"))
                continue

            raw_txt = download_text(txt_url)
            raw_path.write_text(raw_txt, encoding="utf-8", errors="ignore")

            clean_txt = strip_gutenberg_boilerplate(raw_txt)
            wc = word_count(clean_txt)
            if wc < min_words:
                failed.append((gid, lang, f"too_short({wc})"))
                # aj tak uložíme clean (aby si vedel čo to je), ale označíme
                clean_path.write_text(clean_txt, encoding="utf-8", errors="ignore")
                continue

            clean_path.write_text(clean_txt, encoding="utf-8", errors="ignore")

            time.sleep(sleep)

        except Exception as e:
            failed.append((gid, lang, f"error:{type(e).__name__}"))
            time.sleep(sleep)

    # uložiť failed log
    if failed:
        log_path = out_dir / "failed_downloads.csv"
        with log_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gutenberg_id", "language", "reason"])
            w.writerows(failed)
        print(f"\nSome downloads failed. See: {log_path.resolve()}")
    else:
        print("\nAll downloads OK.")

if __name__ == "__main__":
    import csv  # len pre log

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Cesta k meta_filtered.csv")
    ap.add_argument("--out", default="data", help="Výstupný priečinok (napr. data)")
    ap.add_argument("--sleep", type=float, default=0.8, help="Pauza medzi requestmi (Gutenberg etiquette)")
    ap.add_argument("--min-words", type=int, default=30000, help="Min počet slov po čistení")
    ap.add_argument("--overwrite", action="store_true", help="Prepísať existujúce súbory")
    args = ap.parse_args()

    main(args.meta, args.out, args.sleep, args.min_words, args.overwrite)
