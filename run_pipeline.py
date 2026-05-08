#!/usr/bin/env python3
"""
run_pipeline.py — orchestrátor analytickej pipeline pre DP
"Model distribúcie interpunkčných znamienok v rôznych textoch".

Tri módy:
  single   — jeden text (per-kniha analýza)
  corpus   — celý jazykový korpus (~80 textov, plná medzi-jazyková analýza)
  temporal — temporálna analýza 9 kníh (3 autori × 3 obdobia)

Použitie:
  python run_pipeline.py --mode single   --input <path> --input-format raw|tokenized
  python run_pipeline.py --mode corpus   --input <dir>  --input-format raw|tokenized
  python run_pipeline.py --mode temporal --input <dir>  --input-format raw|tokenized

Voliteľné flagy:
  --quick           redukovaný režim (50 bootstrap replikátov, vynechané scale-up
                    a full MFDFA — slúži IBA na overenie, že pipeline beží)
  --out-dir DIR     kam ukladať (default: závisí od módu)
  --langs en,de,nl  ktoré jazyky spracovať (corpus mód)
  --lang en         jazyk pre single-text mód (default: en)
  --seed 42         seed pre pseudonáhodné operácie (default: 42)
  --dry-run         len vypíše plán, nič nespustí
  --phases 03,04    selektívne spustenie konkrétnych fáz (override default)

Výstup:
  Plný popis v adresári <out-dir>/run_summary.txt s kľúčovými číslami a
  zoznamom úspešne / neúspešne dokončených fáz.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


@dataclass
class Phase:
    """Jedna fáza pipeline — popis a~konkrétny príkaz."""
    label: str
    script: Path
    args: list = field(default_factory=list)
    optional: bool = False  # ak True, zlyhanie nezastaví pipeline


# ---------------------------------------------------------------------------
# Definícia módov
# ---------------------------------------------------------------------------

def build_corpus_phases(input_dir: Path, out_root: Path, langs: str,
                        quick: bool, input_format: str) -> list[Phase]:
    """Plný korpusový beh (zodpovedá fázam 02--10)."""
    phases: list[Phase] = []

    if input_format == "raw":
        phases.append(Phase(
            label="02_tokenize",
            script=SCRIPTS / "02_tokenize" / "tokenize_windows_v2.py",
            args=["--src", str(input_dir), "--out", str(out_root / "token_out"),
                  "--langs", langs, "--overwrite"],
        ))
        token_root = out_root / "token_out" / "tokens_words_plus_punct"
    else:
        token_root = input_dir

    phases += [
        Phase("03_punct_stats",
              SCRIPTS / "03_punct_stats" / "extract_punct_stats.py",
              ["--src", str(token_root), "--out", str(out_root / "punct_out"),
               "--langs", langs]),
        Phase("03_punct_stats/boxplots",
              SCRIPTS / "03_punct_stats" / "plot_punct_boxplots.py",
              ["--src", str(out_root / "punct_out"),
               "--out", str(out_root / "punct_out" / "plots")],
              optional=True),
        Phase("04_markov",
              SCRIPTS / "04_markov" / "markov_rowwise_and_perbook.py",
              ["--src", str(token_root), "--out", str(out_root / "markov_out"),
               "--langs", langs]),
        Phase("04_markov/heatmaps",
              SCRIPTS / "04_markov" / "plot_markov_heatmaps_v2.py",
              ["--src", str(out_root / "markov_out"),
               "--out", str(out_root / "markov_out" / "plots"),
               "--langs", langs],
              optional=True),
        Phase("05_rankfreq/zipf",
              SCRIPTS / "05_rankfreq" / "fit_zipf_mandelbrot.py",
              ["--src", str(token_root), "--out", str(out_root / "rankfreq_out"),
               "--langs", langs]),
        Phase("05_rankfreq/heaps",
              SCRIPTS / "05_rankfreq" / "heaps_law.py",
              ["--src", str(token_root), "--out", str(out_root / "rankfreq_out"),
               "--langs", langs],
              optional=True),
        Phase("06_network/edges",
              SCRIPTS / "06_network" / "build_word_adjacency_edges.py",
              ["--src", str(token_root), "--out", str(out_root / "network_out"),
               "--langs", langs]),
        Phase("06_network/metrics",
              SCRIPTS / "06_network" / "compute_network_metrics.py",
              ["--src", str(out_root / "network_out"),
               "--out", str(out_root / "network_out")]),
        Phase("06_network/powerlaw",
              SCRIPTS / "06_network" / "fit_degree_powerlaw.py",
              ["--src", str(out_root / "network_out"),
               "--out", str(out_root / "network_out")],
              optional=True),
        Phase("07_baselines/ba",
              SCRIPTS / "07_baselines" / "build_ba_baseline.py",
              ["--src", str(out_root / "network_out"),
               "--out", str(out_root / "ba_out")]),
        Phase("07_baselines/dm",
              SCRIPTS / "07_baselines" / "build_dm_baseline.py",
              ["--src", str(out_root / "network_out"),
               "--out", str(out_root / "dm_out")],
              optional=True),
        Phase("07_baselines/markov_synth",
              SCRIPTS / "07_baselines" / "build_markov_synthetic.py",
              ["--src", str(token_root),
               "--out", str(out_root / "markov_synth_out")],
              optional=True),
    ]

    if not quick:
        phases.append(Phase(
            label="07_baselines/ba_scaleup",
            script=SCRIPTS / "07_baselines" / "build_ba_scaleup.py",
            args=["--out", str(out_root / "ba_out" / "scaleup")],
            optional=True,
        ))

    return phases


def build_temporal_phases(input_dir: Path, out_root: Path,
                          quick: bool, input_format: str, seed: int) -> list[Phase]:
    """Temporálna analýza 9 kníh (sekcia 11_temporal)."""
    phases: list[Phase] = []

    if input_format == "raw":
        phases.append(Phase(
            label="11_temporal/download_and_tokenize",
            script=SCRIPTS / "11_temporal" / "download_and_tokenize.py",
            args=["--src", str(input_dir), "--out", str(out_root / "tokens")],
        ))
        token_root = out_root / "tokens"
    else:
        token_root = input_dir

    n_boot = 50 if quick else 300
    ks_b = 50 if quick else 200

    phases += [
        Phase("11_temporal/weibull_bootstrap",
              SCRIPTS / "11_temporal" / "weibull_bootstrap.py",
              ["--token-dir", str(token_root),
               "--out-dir", str(out_root / "_bootstrap"),
               "--n-boot", str(n_boot), "--seed", str(seed)]),
        # Nasledujúce skripty NEMAJÚ --out-dir/--token-dir flagy a používajú
        # hardcoded cesty `temporal_data/tokens` -> `temporal_out/...`. Ak ich
        # chceš redirektovať, treba upraviť priamo zdrojové skripty.
        Phase("11_temporal/model_comparison_AIC",
              SCRIPTS / "11_temporal" / "model_comparison.py",
              [],
              optional=True),
        Phase("11_temporal/ks_test",
              SCRIPTS / "11_temporal" / "ks_test_weibull.py",
              ["--n-bootstrap", str(ks_b), "--seed", str(seed)],
              optional=True),
        Phase("11_temporal/independence",
              SCRIPTS / "11_temporal" / "test_ipi_punct_independence.py",
              [],
              optional=True),
        Phase("11_temporal/conditional_weibull",
              SCRIPTS / "11_temporal" / "conditional_weibull.py",
              [],
              optional=True),
        Phase("11_temporal/hazard_and_burstiness",
              SCRIPTS / "11_temporal" / "hazard_and_burstiness.py",
              [],
              optional=True),
        Phase("11_temporal/markov_punct_prediction",
              SCRIPTS / "11_temporal" / "markov_punct_prediction.py",
              ["--tok-root", str(token_root),
               "--out-dir", str(out_root / "_markov_pred"),
               "--seed", str(seed)],
              optional=True),
        Phase("11_temporal/cross_author_compare",
              SCRIPTS / "11_temporal" / "cross_author_compare.py",
              ["--token-dir", str(token_root),
               "--out-dir", str(out_root / "_cross_author")],
              optional=True),
        Phase("11_temporal/master_plot",
              SCRIPTS / "11_temporal" / "master_plot.py",
              ["--out-dir", str(out_root / "_master_plots")],
              optional=True),
    ]

    return phases


def build_single_phases(input_path: Path, out_root: Path, lang: str,
                        quick: bool, input_format: str, seed: int) -> list[Phase]:
    """Single-text analýza. Pripraví sa sandbox adresár matching očakávanej
    štruktúre (lang/file.txt) a~potom sa spúšťajú existujúce skripty."""
    sandbox = out_root / "_sandbox"
    sb_lang = sandbox / lang
    sb_lang.mkdir(parents=True, exist_ok=True)

    target = sb_lang / input_path.name
    if not target.exists():
        shutil.copy2(input_path, target)

    phases: list[Phase] = []
    if input_format == "raw":
        phases.append(Phase(
            label="02_tokenize",
            script=SCRIPTS / "02_tokenize" / "tokenize_windows_v2.py",
            args=["--src", str(sandbox), "--out", str(out_root / "token_out"),
                  "--langs", lang, "--overwrite"],
        ))
        token_root = out_root / "token_out" / "tokens_words_plus_punct"
    else:
        token_root = sandbox

    n_boot = 50 if quick else 300

    phases += [
        Phase("03_punct_stats",
              SCRIPTS / "03_punct_stats" / "extract_punct_stats.py",
              ["--src", str(token_root), "--out", str(out_root / "punct_out"),
               "--langs", lang]),
        Phase("04_markov",
              SCRIPTS / "04_markov" / "markov_rowwise_and_perbook.py",
              ["--src", str(token_root), "--out", str(out_root / "markov_out"),
               "--langs", lang]),
        Phase("06_network/edges",
              SCRIPTS / "06_network" / "build_word_adjacency_edges.py",
              ["--src", str(token_root), "--out", str(out_root / "network_out"),
               "--langs", lang]),
        Phase("06_network/metrics",
              SCRIPTS / "06_network" / "compute_network_metrics.py",
              ["--src", str(out_root / "network_out"),
               "--out", str(out_root / "network_out")]),
        Phase("11_temporal/weibull_bootstrap",
              SCRIPTS / "11_temporal" / "weibull_bootstrap.py",
              ["--tok-root", str(token_root),
               "--out-dir", str(out_root / "_bootstrap"),
               "--n-bootstrap", str(n_boot), "--seed", str(seed)],
              optional=True),
        Phase("11_temporal/model_comparison_AIC",
              SCRIPTS / "11_temporal" / "model_comparison.py",
              ["--tok-root", str(token_root),
               "--out-dir", str(out_root / "_aic")],
              optional=True),
        Phase("11_temporal/hazard_and_burstiness",
              SCRIPTS / "11_temporal" / "hazard_and_burstiness.py",
              ["--tok-root", str(token_root),
               "--out-dir", str(out_root / "_hazard")],
              optional=True),
    ]

    return phases


# ---------------------------------------------------------------------------
# Beh
# ---------------------------------------------------------------------------

def run_phase(phase: Phase, log_file) -> tuple[bool, float]:
    cmd = [sys.executable, str(phase.script)] + phase.args
    print(f"\n[{phase.label}] {' '.join(cmd)}", flush=True)
    log_file.write(f"\n=== {phase.label} ===\n{' '.join(cmd)}\n")
    log_file.flush()

    if not phase.script.exists():
        msg = f"  SKIPPED: skript neexistuje ({phase.script})"
        print(msg)
        log_file.write(msg + "\n")
        return False, 0.0

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=False, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                env=env)
    except Exception as exc:
        log_file.write(f"  EXCEPTION: {exc}\n")
        return False, time.time() - t0

    elapsed = time.time() - t0
    log_file.write(result.stdout or "")
    log_file.write(result.stderr or "")
    log_file.flush()

    if result.returncode != 0:
        print(f"  ZLYHALO (returncode={result.returncode}, čas={elapsed:.1f}s)")
        if result.stderr:
            print(result.stderr[-500:])
        return False, elapsed

    print(f"  OK ({elapsed:.1f}s)")
    return True, elapsed


def write_summary(out_root: Path, mode: str, results: list[tuple[Phase, bool, float]],
                  args) -> None:
    summary = out_root / "run_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"=== run_pipeline summary ===\n")
        f.write(f"mód: {mode}\n")
        f.write(f"vstup: {args.input}\n")
        f.write(f"vstupný formát: {args.input_format}\n")
        f.write(f"výstup: {out_root}\n")
        f.write(f"quick: {args.quick}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"\n--- fázy ---\n")
        ok = sum(1 for _, s, _ in results if s)
        total_t = sum(t for _, _, t in results)
        for phase, success, elapsed in results:
            tag = "OK" if success else ("OPT-FAIL" if phase.optional else "FAIL")
            f.write(f"[{tag:>8}] {phase.label:<40} {elapsed:>6.1f}s\n")
        f.write(f"\nÚspešné: {ok}/{len(results)}, celkový čas: {total_t:.1f}s\n")
    print(f"\nSummary uložené do: {summary}")


# ---------------------------------------------------------------------------
# Argparse a main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["single", "corpus", "temporal"],
                   help="Beh módu (single/corpus/temporal).")
    p.add_argument("--input", required=True,
                   help="Cesta k súboru (single) alebo adresáru (corpus/temporal).")
    p.add_argument("--input-format", required=True, choices=["raw", "tokenized"],
                   help="Formát vstupu: raw text z Gutenbergu/DBNL alebo už "
                        "tokenizované súbory (jeden token na riadok).")
    p.add_argument("--out-dir", default=None,
                   help="Cieľový adresár (default podľa módu).")
    p.add_argument("--langs", default="en,de,nl",
                   help="Zoznam jazykov pre corpus mód (default en,de,nl).")
    p.add_argument("--lang", default="en",
                   help="Jazyk pre single mód (default en).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Redukovaný režim (50 bootstrap, bez BA scale-up). "
                        "Iba na overenie behu, NIE pre publikovateľné výsledky.")
    p.add_argument("--dry-run", action="store_true",
                   help="Len vypíše plán, nič nespustí.")
    p.add_argument("--phases", default=None,
                   help="Selektívne spustenie (čiarkou oddelený zoznam labelov "
                        "alebo prefixov, napr. '03,04_markov').")
    return p.parse_args()


def filter_phases(phases: list[Phase], spec: str | None) -> list[Phase]:
    if not spec:
        return phases
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    return [p for p in phases if any(p.label.startswith(w) for w in wanted)]


def default_out(mode: str, quick: bool) -> Path:
    suffix = "_quick" if quick else ""
    return ROOT / f"run_{mode}{suffix}_out"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Vstup neexistuje: {input_path}", file=sys.stderr)
        return 2

    out_root = Path(args.out_dir).resolve() if args.out_dir \
        else default_out(args.mode, args.quick)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "corpus":
        if not input_path.is_dir():
            print("corpus mód vyžaduje adresár ako vstup.", file=sys.stderr)
            return 2
        phases = build_corpus_phases(input_path, out_root, args.langs,
                                     args.quick, args.input_format)
        skipped_explicit = [
            "11_temporal/* (špecifické pre temporálny korpus 9 kníh)",
        ]
    elif args.mode == "temporal":
        if not input_path.is_dir():
            print("temporal mód vyžaduje adresár ako vstup.", file=sys.stderr)
            return 2
        phases = build_temporal_phases(input_path, out_root, args.quick,
                                       args.input_format, args.seed)
        skipped_explicit = [
            "03_punct_stats/boxplots (vyžadujú väčšiu vzorku)",
            "05_rankfreq (Zipf/Heaps/MFDFA — corpus-level)",
            "07_baselines (BA/DM/Markov-synth — corpus-level porovnanie)",
        ]
    else:  # single
        if not input_path.is_file():
            print("single mód vyžaduje súbor ako vstup.", file=sys.stderr)
            return 2
        phases = build_single_phases(input_path, out_root, args.lang,
                                     args.quick, args.input_format, args.seed)
        skipped_explicit = [
            "05_rankfreq (Zipf/Heaps/MFDFA — vyžadujú dlhý korpus)",
            "07_baselines (BA/DM/Markov-synth — corpus-level porovnanie)",
            "11_temporal/{drift,independence,cond_weibull,markov_pred,cross_author} "
            "(viacknižné porovnania)",
        ]

    phases = filter_phases(phases, args.phases)

    print(f"\n[{args.mode} mode] beh fáz:")
    for p in phases:
        print(f"  - {p.label}" + (" (optional)" if p.optional else ""))
    print(f"\n[{args.mode} mode] vynechané z definície:")
    for line in skipped_explicit:
        print(f"  - {line}")
    print(f"\nVýstup: {out_root}")
    if args.quick:
        print("POZOR: QUICK rezim - redukovane bootstrap replikaty, vysledky "
              "NIE SU vhodne na publikaciu, iba na overenie behu.")

    if args.dry_run:
        print("\n[dry-run] nič sa nespúšťa.")
        return 0

    log_path = out_root / "run.log"
    results: list[tuple[Phase, bool, float]] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        for phase in phases:
            success, elapsed = run_phase(phase, log_file)
            results.append((phase, success, elapsed))
            if not success and not phase.optional:
                print(f"\nFáza {phase.label} zlyhala — pipeline preruším.")
                break

    write_summary(out_root, args.mode, results, args)
    failed_required = [p for p, s, _ in results if not s and not p.optional]
    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())
