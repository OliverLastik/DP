#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import re

# Uprav si podľa tokenizéra, ak používaš iné značky pre úvodzovky atď.
PUNCT_SET = {
    ".", ",", "!", "?", ";", ":", "—", "-", "–",
    "(", ")", "[", "]", "{", "}", "\"", "'", "„", "“", "”", "’", "‚",
    "...", "…"
}

def is_punct(tok: str) -> bool:
    return tok in PUNCT_SET

def read_edges_csv(path: Path):
    """
    Očakáva edge list CSV aspoň s columns:
    src,dst,weight   (alebo source,target,weight)
    """
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # auto-detect column names
        cols = {c.lower(): c for c in r.fieldnames or []}
        src_col = cols.get("src") or cols.get("source") or cols.get("u") or cols.get("from")
        dst_col = cols.get("dst") or cols.get("target") or cols.get("v") or cols.get("to")
        w_col   = cols.get("weight") or cols.get("w") or cols.get("count")

        if not (src_col and dst_col and w_col):
            raise ValueError(f"Neviem nájsť stĺpce src/dst/weight v {path} (mám: {r.fieldnames})")

        for row in r:
            u = (row.get(src_col) or "").strip()
            v = (row.get(dst_col) or "").strip()
            if not u or not v:
                continue
            try:
                w = float(row.get(w_col))
            except Exception:
                continue
            yield u, v, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", default="network_out_core/edges", help="Kde sú edge listy")
    ap.add_argument("--out", default="network_out_core/top_reports", help="Výstupný priečinok")
    ap.add_argument("--langs", default="en,de,nl")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--per-book", action="store_true", help="Spraviť top report aj pre každú knihu zvlášť")
    args = ap.parse_args()

    edges_dir = Path(args.edges_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    for lang in langs:
        # Skus najprv per-book priecinok, potom flat file
        lang_dir = edges_dir / lang
        flat_file = edges_dir / f"edges_{lang}.csv"

        if lang_dir.exists() and lang_dir.is_dir():
            files = sorted(lang_dir.glob("*.csv"))
        elif flat_file.exists():
            files = [flat_file]
        else:
            print(f"[WARN] {lang}: nenasiel som edges v {lang_dir} ani {flat_file}")
            continue

        # Aggregate for language
        deg = Counter()
        edge_w = Counter()   # key=(u,v)
        nodes_seen = set()

        # Optionally per book
        per_book_results = {}

        if not files:
            print(f"[WARN] {lang}: ziadne CSV edge subory")
            continue

        for f in files:
            book_deg = Counter()
            book_edge_w = Counter()
            for u, v, w in read_edges_csv(f):
                nodes_seen.add(u); nodes_seen.add(v)
                # degree: berieme neorientovane ako "adjacency" (ak máš orientované, povedz a upravím)
                deg[u] += 1
                deg[v] += 1
                book_deg[u] += 1
                book_deg[v] += 1

                edge_w[(u, v)] += w
                book_edge_w[(u, v)] += w

            if args.per_book:
                gid = f.stem  # napr. 10891
                per_book_results[gid] = (book_deg, book_edge_w)

        # --- language-level tops ---
        top_nodes = deg.most_common(args.topn)
        top_edges = edge_w.most_common(args.topn)

        punct_in_top = [tok for tok, d in top_nodes if is_punct(tok)]
        punct_ratio = (len(punct_in_top) / max(1, len(top_nodes))) * 100.0

        # Save language report (txt)
        rep_path = outdir / f"top_report_{lang}.txt"
        with rep_path.open("w", encoding="utf-8") as w:
            w.write(f"LANG={lang}\n")
            w.write(f"nodes_seen={len(nodes_seen)}\n\n")

            w.write(f"TOP {args.topn} NODES BY DEGREE:\n")
            for tok, d in top_nodes:
                w.write(f"{tok}\t{d}\t{'PUNCT' if is_punct(tok) else ''}\n")

            w.write("\n")
            w.write(f"PUNCT IN TOP NODES: {len(punct_in_top)}/{len(top_nodes)} = {punct_ratio:.1f}%\n")
            if punct_in_top:
                w.write("punct_tokens: " + ", ".join(punct_in_top) + "\n")

            w.write("\nTOP {0} EDGES BY WEIGHT:\n".format(args.topn))
            for (u, v), ww in top_edges:
                w.write(f"{u} -> {v}\t{ww}\n")

        # Save language top nodes (csv)
        nodes_csv = outdir / f"top_nodes_{lang}.csv"
        with nodes_csv.open("w", encoding="utf-8", newline="") as f:
            ww = csv.writer(f)
            ww.writerow(["token", "degree", "is_punct"])
            for tok, d in top_nodes:
                ww.writerow([tok, d, int(is_punct(tok))])

        edges_csv = outdir / f"top_edges_{lang}.csv"
        with edges_csv.open("w", encoding="utf-8", newline="") as f:
            ww = csv.writer(f)
            ww.writerow(["src", "dst", "weight"])
            for (u, v), wwgt in top_edges:
                ww.writerow([u, v, wwgt])

        print(f"[OK] {lang}: saved {rep_path} (+ CSVs)")

        # --- optional per-book reports ---
        if args.per_book:
            per_dir = outdir / f"per_book_{lang}"
            per_dir.mkdir(parents=True, exist_ok=True)
            for gid, (bd, be) in per_book_results.items():
                bn = bd.most_common(args.topn)
                be_top = be.most_common(args.topn)
                p_in = [tok for tok, d in bn if is_punct(tok)]
                p_ratio = (len(p_in) / max(1, len(bn))) * 100.0

                p = per_dir / f"{gid}.txt"
                with p.open("w", encoding="utf-8") as w:
                    w.write(f"LANG={lang} BOOK={gid}\n\n")
                    w.write(f"TOP {args.topn} NODES BY DEGREE:\n")
                    for tok, d in bn:
                        w.write(f"{tok}\t{d}\t{'PUNCT' if is_punct(tok) else ''}\n")
                    w.write(f"\nPUNCT IN TOP: {len(p_in)}/{len(bn)} = {p_ratio:.1f}%\n")
                    if p_in:
                        w.write("punct_tokens: " + ", ".join(p_in) + "\n")

                    w.write("\nTOP {0} EDGES BY WEIGHT:\n".format(args.topn))
                    for (u, v), wwgt in be_top:
                        w.write(f"{u} -> {v}\t{wwgt}\n")

            print(f"[OK] {lang}: per-book reports in {per_dir}")

if __name__ == "__main__":
    main()