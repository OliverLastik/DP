#!/usr/bin/env python3
import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://www.gutenberg.org"
HEADERS = {"User-Agent": "DP-Punctuation-Downloader/1.2 (educational use)"}

HEADER_START = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
HEADER_END   = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK", re.IGNORECASE)

def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.splitlines()
    start_idx = 0
    for i, ln in enumerate(lines[:700]):
        if HEADER_START.search(ln):
            start_idx = i + 1
            break

    end_idx = len(lines)
    for j in range(len(lines) - 1, max(len(lines) - 6000, start_idx), -1):
        if HEADER_END.search(lines[j]):
            end_idx = j
            break

    return "\n".join(lines[start_idx:end_idx]).strip()

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))

def fetch(url: str, timeout=60) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r

def ebook_page(gid: int) -> str:
    return fetch(f"{BASE}/ebooks/{gid}", timeout=30).text

def pick_txt_url_bs4(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    links = soup.select("table.files a.link")
    candidates: List[Tuple[str, str]] = []
    for a in links:
        label = a.get_text(" ", strip=True).lower()
        href = a.get("href")
        if not href:
            continue
        if ".txt" not in href.lower():
            continue
        full = href if href.startswith("http") else BASE + href
        candidates.append((label, full))

    # Preferuj Plain Text UTF-8
    for label, url in candidates:
        if "plain text utf-8" in label or "plain text utf8" in label:
            return url

    # Potom Plain Text
    for label, url in candidates:
        if "plain text" in label:
            return url

    return None

def fallback_txt_urls(gid: int) -> List[str]:
    return [
        f"{BASE}/cache/epub/{gid}/pg{gid}.txt",
        f"{BASE}/cache/epub/{gid}/pg{gid}.txt.utf8",
        f"{BASE}/files/{gid}/{gid}-0.txt",
        f"{BASE}/files/{gid}/{gid}.txt",
    ]

def download_text_from_url(url: str) -> str:
    r = fetch(url, timeout=60)
    if r.encoding is None:
        r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_kv_int_map(s: str) -> Dict[str, int]:
    """
    Parse map typu: "da=20000,sv=25000" -> {"da": 20000, "sv": 25000}
    """
    out: Dict[str, int] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        lang, val = part.split("=")
        out[lang.strip()] = int(val.strip())
    return out

def current_good_counts(clean_dir: Path, min_words_default: int, min_by_lang: Dict[str, int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not clean_dir.exists():
        return counts
    for lang_dir in clean_dir.glob("*"):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        mw = min_by_lang.get(lang, min_words_default)
        good = 0
        for f in lang_dir.glob("*.txt"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                if word_count(txt) >= mw:
                    good += 1
            except Exception:
                pass
        counts[lang] = good
    return counts

def load_existing_accepted_counts(accepted_csv: Path) -> Dict[Tuple[str, str], int]:
    """
    Ak už existuje logs/accepted.csv, načíta počty (lang, author_key) -> count,
    aby sa max-per-author držal aj pri opakovanom spustení.
    """
    counts: Dict[Tuple[str, str], int] = {}
    if not accepted_csv.exists():
        return counts
    try:
        with accepted_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                lang = (row.get("language") or "").strip()
                akey = (row.get("author_key") or "").strip()
                if not lang or not akey:
                    continue
                k = (lang, akey)
                counts[k] = counts.get(k, 0) + 1
    except Exception:
        pass
    return counts

def main(
    meta_path: str,
    out_dir: str,
    target_default: int,
    target_by_lang: str,
    sleep: float,
    min_words_default: int,
    min_words_by_lang: str,
    seed: int,
    max_per_author: int,
    overwrite: bool,
):
    meta_path = Path(meta_path)
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"
    logs_dir = out_dir / "logs"
    ensure_dir(raw_dir); ensure_dir(clean_dir); ensure_dir(logs_dir)

    df = pd.read_csv(meta_path, dtype=str, encoding="utf-8")
    if "Text#" not in df.columns or "Language" not in df.columns:
        raise ValueError(f"Meta musí obsahovať Text# a Language. Stĺpce: {list(df.columns)}")
    if "author_key" not in df.columns:
        raise ValueError("Meta musí obsahovať 'author_key'. Vygeneruj meta_candidates.csv cez filter_pg_catalog.py s author_key.")

    # targets per language
    targets = parse_kv_int_map(target_by_lang)
    # min words per language
    min_by_lang = parse_kv_int_map(min_words_by_lang)

    # existujúce accepted (kvôli opakovanému behu)
    accepted_csv_path = logs_dir / "accepted.csv"
    accepted_per_author = load_existing_accepted_counts(accepted_csv_path)

    # zisti aktuálne počty good textov v clean/*
    cur_good = current_good_counts(clean_dir, min_words_default, min_by_lang)
    print("Current good counts:", cur_good)

    # priprav id listy per jazyk a premapovanie gid -> author_key
    df["Text#"] = df["Text#"].astype(int)
    gid_to_author = dict(zip(df["Text#"].tolist(), df["author_key"].fillna("").tolist()))

    grouped_ids: Dict[str, List[int]] = {}
    for lang, g in df.groupby("Language"):
        ids = g["Text#"].tolist()
        # deterministic shuffle
        ids = pd.Series(ids).sample(frac=1.0, random_state=seed).tolist()
        grouped_ids[lang] = ids

    accepted_rows = []
    rejected_rows = []
    failed_rows = []

    for lang, ids in grouped_ids.items():
        lang_target = targets.get(lang, target_default)
        mw = min_by_lang.get(lang, min_words_default)

        have = cur_good.get(lang, 0)
        if have >= lang_target:
            print(f"{lang}: already have {have}/{lang_target} good texts, skipping.")
            continue

        print(f"\n== {lang}: need {lang_target - have} more (min_words={mw}, max_per_author={max_per_author}) ==")

        ensure_dir(raw_dir / lang)
        ensure_dir(clean_dir / lang)

        for gid in tqdm(ids, desc=f"{lang}"):
            if have >= lang_target:
                break

            akey = gid_to_author.get(gid, "")
            if max_per_author and akey:
                k = (lang, akey)
                if accepted_per_author.get(k, 0) >= max_per_author:
                    continue

            raw_path = raw_dir / lang / f"{gid}.txt"
            clean_path = clean_dir / lang / f"{gid}.txt"

            # ak už existuje clean a je dosť dlhý, započítaj a pokračuj
            if clean_path.exists() and not overwrite:
                txt = clean_path.read_text(encoding="utf-8", errors="ignore")
                wc = word_count(txt)
                if wc >= mw:
                    # akceptuj "existujúci" kus, ale rešpektuj max_per_author
                    if max_per_author and akey:
                        k = (lang, akey)
                        if accepted_per_author.get(k, 0) >= max_per_author:
                            continue
                        accepted_per_author[k] = accepted_per_author.get(k, 0) + 1
                    have += 1
                    continue

            try:
                html = ebook_page(gid)
                txt_url = pick_txt_url_bs4(html)

                urls_to_try = []
                if txt_url:
                    urls_to_try.append(txt_url)
                urls_to_try.extend(fallback_txt_urls(gid))

                text_raw = None
                used_url = None
                for u in urls_to_try:
                    try:
                        text_raw = download_text_from_url(u)
                        used_url = u
                        if text_raw and len(text_raw) > 500:
                            break
                    except Exception:
                        continue

                if not text_raw:
                    failed_rows.append((gid, lang, "no_txt_url"))
                    time.sleep(sleep)
                    continue

                raw_path.write_text(text_raw, encoding="utf-8", errors="ignore")

                text_clean = strip_gutenberg_boilerplate(text_raw)
                wc = word_count(text_clean)
                clean_path.write_text(text_clean, encoding="utf-8", errors="ignore")

                if wc < mw:
                    rejected_rows.append((gid, lang, akey, f"too_short({wc})", used_url or ""))
                else:
                    # final check max_per_author just before accepting
                    if max_per_author and akey:
                        k = (lang, akey)
                        if accepted_per_author.get(k, 0) >= max_per_author:
                            rejected_rows.append((gid, lang, akey, "skipped_max_per_author", used_url or ""))
                        else:
                            accepted_rows.append((gid, lang, akey, wc, used_url or ""))
                            accepted_per_author[k] = accepted_per_author.get(k, 0) + 1
                            have += 1
                    else:
                        accepted_rows.append((gid, lang, akey, wc, used_url or ""))
                        have += 1

                time.sleep(sleep)

            except Exception as e:
                failed_rows.append((gid, lang, f"error:{type(e).__name__}"))
                time.sleep(sleep)

        print(f"{lang}: collected {have}/{lang_target}")

    # uložiť logy (append aj keď už existujú)
    def append_csv(path: Path, header: List[str], rows: List[tuple]):
        if not rows:
            return
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerows(rows)

    append_csv(logs_dir / "accepted.csv", ["gutenberg_id", "language", "author_key", "words", "url"], accepted_rows)
    append_csv(logs_dir / "rejected.csv", ["gutenberg_id", "language", "author_key", "reason", "url"], rejected_rows)
    append_csv(logs_dir / "failed.csv", ["gutenberg_id", "language", "reason"], failed_rows)

    print("\nDone.")
    print(f"Logs in: {logs_dir.resolve()}")
    print("Tip: pozri accepted.csv a skontroluj rozloženie autorov (max 3 na autora by malo platiť).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="meta_candidates.csv (musí obsahovať author_key)")
    ap.add_argument("--out", default="data", help="output dir (data/...)")
    ap.add_argument("--target", type=int, default=80, help="default target per language")
    ap.add_argument("--target-by-lang", default="", help="e.g. da=80,sv=80 (inak použije --target)")
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--min-words", type=int, default=30000)
    ap.add_argument("--min-words-by-lang", default="", help="e.g. da=20000,sv=25000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-author", type=int, default=3, help="max accepted titles per author within a language")
    ap.add_argument("--overwrite", action="store_true", help="re-download and overwrite existing txt files")
    args = ap.parse_args()

    main(
        meta_path=args.meta,
        out_dir=args.out,
        target_default=args.target,
        target_by_lang=args.target_by_lang,
        sleep=args.sleep,
        min_words_default=args.min_words,
        min_words_by_lang=args.min_words_by_lang,
        seed=args.seed,
        max_per_author=args.max_per_author,
        overwrite=args.overwrite,
    )
