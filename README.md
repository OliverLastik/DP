# Model distribúcie interpunkčných znamienok v rôznych textoch

Tento repozitár obsahuje elektronickú prílohu k diplomovej práci zameranej na
analýzu distribúcie interpunkčných znamienok v literárnych textoch. Projekt
spája korpusové spracovanie textov, štatistiku interpunkcie, Markovove modely,
rank-frequency analýzy, sieťové metriky a temporálnu analýzu vybraných autorov.

## Obsah repozitára

- `run_pipeline.py` - jednotný riadiaci skript celej analytickej pipeline.
- `scripts/` - zdrojové kódy jednotlivých fáz spracovania od sťahovania dát až po temporálne analýzy.
- `requirements.txt` - zoznam Python knižníc s použitými verziami.
- `temporal_data/tokens/` - tokenizované verzie kníh použitých v temporálnej časti.
- `temporal_out/` - numerické výstupy a grafy temporálnej analýzy.

## Inštalácia

Odporúčaný postup je vytvoriť samostatné Python prostredie a nainštalovať
závislosti zo súboru `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Spustenie pipeline

Hlavným vstupným bodom je skript `run_pipeline.py`.

```powershell
python run_pipeline.py --help
```

Pipeline podporuje tri režimy:

- `single` - analýza jedného textu.
- `corpus` - analýza celého jazykového korpusu.
- `temporal` - temporálna analýza vybraných kníh.

Príklad rýchlej kontroly temporálnej pipeline:

```powershell
python run_pipeline.py --mode temporal --input temporal_data/tokens --input-format tokenized --quick
```

Parameter `--quick` slúži iba na overenie behu pipeline. Pre publikovateľné
výsledky je potrebné spustiť analýzu bez tohto parametra.

## Výstupy

Výstupy temporálnej analýzy sú uložené v adresári `temporal_out/`. Obsahujú
napríklad CSV tabuľky s odhadmi parametrov, AIC porovnaniami, KS testami,
bootstrap výsledkami a vygenerované grafy.

Pri vlastnom spustení pipeline sa vytvorí aj `run_summary.txt`, ktorý sumarizuje
spustené fázy, ich úspešnosť a celkový čas behu.
