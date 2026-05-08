#!/usr/bin/env python3
import argparse
import re
import csv
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError

# -------------------------
# Helpers: normalizácia názvov stĺpcov
# -------------------------
def normalize_colname(x: str) -> str:
    if x is None:
        return x
    x = str(x).replace("\ufeff", "").replace("\xa0", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df

# -------------------------
# Parsovanie rokov autora
# -------------------------
YEAR_PAIR_RE = re.compile(r"(?P<birth>\d{4})\s*[-–]\s*(?P<death>\d{4})")
YEAR_BIRTH_RE = re.compile(r"(?P<birth>\d{4})\s*[-–]\s*$")
ANY_YEAR_RE = re.compile(r"\b(\d{4})\b")

def parse_author_years(authors: str):
    if not isinstance(authors, str) or not authors.strip():
        return None, None

    parts = [p.strip() for p in authors.split(";") if p.strip()]
    for p in parts:
        m = YEAR_PAIR_RE.search(p)
        if m:
            return int(m.group("birth")), int(m.group("death"))
        m2 = YEAR_BIRTH_RE.search(p)
        if m2:
            return int(m2.group("birth")), None

    m3 = ANY_YEAR_RE.findall(authors)
    if m3:
        return int(m3[0]), None

    return None, None

# -------------------------
# Žáner (MVP: próza)
# -------------------------
BAD_KEYWORDS = ["poetry", "poems", "drama", "plays", "traged", "ballads", "songs"]
GOOD_KEYWORDS = ["fiction", "novel", "stories", "short stories"]

def is_likely_prose(subjects: str, bookshelves: str, require_good=False):
    s = f"{subjects or ''} ; {bookshelves or ''}".lower()
    if any(k in s for k in BAD_KEYWORDS):
        return False
    has_good = any(k in s for k in GOOD_KEYWORDS)
    return has_good if require_good else True

# -------------------------
# Časový filter (19. + zač. 20.) cez roky autora
# -------------------------
def passes_time_window(birth, death, birth_max=1885, death_min=1830):
    if birth is None:
        return False
    if birth > birth_max:
        return False
    if death is None:
        return True
    return death >= death_min

# -------------------------
# Detekcia delimiteru
# -------------------------
def detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(16384)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        counts = {",": sample.count(","), "\t": sample.count("\t"), ";": sample.count(";"), "|": sample.count("|")}
        return max(counts, key=counts.get)

def author_key(authors: str) -> str:
    """
    Zoberie prvého autora (pred ';'), odstráni roky a normalizuje.
    Príklady:
      "Twain, Mark, 1835-1910" -> "twain, mark"
      "Shakespeare, William, 1564-1616; ..." -> "shakespeare, william"
    """
    if not isinstance(authors, str) or not authors.strip():
        return ""
    first = authors.split(";")[0].strip()
    first = re.sub(r",?\s*\d{4}\s*[-–]\s*\d{0,4}\s*$", "", first).strip()
    first = re.sub(r"\s+", " ", first).strip().lower()
    return first


def read_catalog(path: str) -> pd.DataFrame:
    delim = detect_delimiter(path)
    print(f"Detected delimiter: {repr(delim)}")

    # 1) Skús normálne CSV quoting (správne pre klasické CSV)
    try:
        df = pd.read_csv(
            path,
            sep=delim,
            dtype=str,
            encoding="utf-8-sig",
            engine="python",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip",
        )
    except ParserError:
        # 2) Fallback: ignoruj quoting (pre rozbité úvodzovky)
        df = pd.read_csv(
            path,
            sep=delim,
            dtype=str,
            encoding="utf-8-sig",
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )

    df = normalize_columns(df)
    print("Columns:", list(df.columns))
    print("Initial shape:", df.shape)
    return df

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Cesta k pg_catalog.csv")
    ap.add_argument("--out", dest="out", default="meta_filtered.csv", help="Výstupný CSV súbor")
    ap.add_argument("--langs", default="en,de,nl,sv,da")
    ap.add_argument("--per-lang", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-good-genre", action="store_true")
    ap.add_argument("--birth-max", type=int, default=1885)
    ap.add_argument("--death-min", type=int, default=1830)
    ap.add_argument("--max-per-author", type=int, default=3, help="Max titulov na autora v rámci jedného jazyka")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    df = read_catalog(args.inp)

    required_cols = ["Text#", "Type", "Title", "Language", "Authors", "Subjects", "Bookshelves"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Chýba stĺpec '{c}'. Našiel som: {list(df.columns)}")

    # --- Type filter: robustný ---
    type_norm = (
        df["Type"]
        .fillna("")
        .astype(str)
        .str.replace('"', "", regex=False)
        .str.strip()
        .str.lower()
    )

    # ukáž top hodnoty, nech vidíme čo tam reálne je
    print("\nTop Type values:")
    print(type_norm.value_counts().head(10))

    mask_text = type_norm.eq("text")
    if mask_text.sum() > 0:
        df = df.loc[mask_text].copy()
        print("After Type==Text:", df.shape)
    else:
        # Ak v Type vôbec nie je 'text', Type filter preskočíme (inak skončíš na 0).
        print("WARNING: No rows with Type=='Text' found. Skipping Type filter.")
        print("After Type filter skipped:", df.shape)

    # Len vybrané jazyky
    df = df.loc[df["Language"].isin(langs)].copy()
    print("After language filter:", df.shape)

    # Parse author years
    years = df["Authors"].apply(parse_author_years)
    df["author_birth"] = years.apply(lambda x: x[0])
    df["author_death"] = years.apply(lambda x: x[1])

    # Časové okno
    mask_time = df.apply(
        lambda r: passes_time_window(
            r["author_birth"], r["author_death"],
            birth_max=args.birth_max, death_min=args.death_min
        ),
        axis=1
    )
    df = df.loc[mask_time].copy()
    print("After time window:", df.shape)

    # Žáner / beletria
    mask_genre = df.apply(
        lambda r: is_likely_prose(
            r.get("Subjects", ""),
            r.get("Bookshelves", ""),
            require_good=args.require_good_genre
        ),
        axis=1
    )
    df = df.loc[mask_genre].copy()
    print("After genre filter:", df.shape)
    # limit na autora (per jazyk)
    df["author_key"] = df["Authors"].apply(author_key)

    # premiešaj (aby to nebralo vždy rovnaké prvé 3)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.max_per_author and args.max_per_author > 0:
        df = (df.groupby(["Language", "author_key"], group_keys=False)
              .head(args.max_per_author)
              ).copy()

    print("After max-per-author:", df.shape)

    out_cols = ["Text#", "Language", "Title", "Authors", "author_key",
                "author_birth", "author_death", "Subjects", "Bookshelves"]

    df = df[out_cols].copy()

    df = df.drop_duplicates(subset=["Text#"])

    if args.per_lang and args.per_lang > 0:
        df = (df.groupby("Language", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), args.per_lang), random_state=args.seed))
             ).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("\nPočty po filtrovaní:")
    print(df["Language"].value_counts().sort_index())
    print(f"\nUložené do: {out_path.resolve()}")

if __name__ == "__main__":
    main()
