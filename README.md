# CGSM / Pajarales TSS Retrieval — Manuscript Repository

Code, data, and manuscript source for: *Physically-Constrained Calibration and
Transferability of a Semi-Analytical TSS Retrieval Model in a Tropical Coastal
Lagoon System: Ciénaga Grande de Santa Marta, Colombia* (target: MDPI *Remote
Sensing*).

## Results Summary

### Step 1 — Pajarales calibration

| Sensor | Fitting mode | $A_p$ | $C_p$ | LOSO-CV $R^2$ | RMSE |
|---|---|---|---|---|---|
| Landsat-8 | free 2-param (bootstrap-validated, CV=4.6%) | 1023.0 | 0.2732 | **0.784** | 50.5 mg/L |
| Sentinel-2 (corrected) | boundary-converged, reported as 1-param | 533.2 | 0.55 | **0.372** | 73.2 mg/L |

Sensors calibrated and reported **separately** — a pooled fit underperforms
either alone (in-sample $R^2$=0.353 vs 0.803/0.419).

**Secondary model comparison (L8 only)**: a power-law alternative
($TSS=a\rho^b$, no saturation parameter) modestly outperforms Nechad on
Landsat-8 (LOSO $R^2$=0.796 vs 0.784, bias +0.2 vs +6.2 mg/L), bootstrap-
validated to the same standard. Does NOT generalize: worse than Nechad on
S2 (0.314 vs 0.372), mixed on CGSM (0.616, within Nechad's own range).
Nechad retained as primary model for physical interpretability and
compatibility with the transfer framework; power-law reported as a real,
relevant comparison (`code/08_powerlaw_comparison.py`, §3.3 in `main.tex`).

### Step 2 — CGSM evaluation

$\kappa$=0.13, bootstrap CV($C_p$)=49.6% → genuinely unidentifiable, correctly
uses 1-param fixed-$C_p$. In-domain (Scenario B/C style) performance:
$R^2$=0.53–0.64 depending on local adaptation.

### Step 3 — Transferability, Pajarales → CGSM

| Scenario | $R^2$ | RMSE | Bias | What it means |
|---|---|---|---|---|
| A. Direct transfer (zero local data) | 0.262 | 47.9 | −22.3 | Weak — the *better* L8 source transfers worse, because its sharper curve ($C_p$=0.27) is more sensitive to cross-site offset than a flatter one would be |
| B. Local NECHAD, free $C_p$ | 0.639 | 33.5 | +3.7 | Best physical model |
| C. Partial transfer ($C_p$ fixed at source) | 0.563 | 36.8 | +8.3 | Doesn't beat B — CGSM's narrow range under-uses that $C_p$ anchor |
| D. Local GBR (non-physical ceiling) | 0.681 | 31.5 | −0.6 | Upper bound |

### Methodological corrections made during this project

1. **κ=0.35 threshold was wrong** — replaced with a validated bootstrap-CV($C_p$)
   criterion; this is the paper's now-defensible core contribution
   (`code/05_bootstrap_identifiability.py`)
2. **ACOLITE beats raw Landsat C2L2** (half the noise at matched TSS) — settled
   with data, not assumption (`code/01_pajarales_l8_raw_pipeline.py`,
   §4.3.1 in `main.tex`)
3. **S2 atmospheric correction fix** improved r: 0.118→0.622, $R^2$: −0.38→0.372
   — real but doesn't close the L8 gap
4. **Sensor pooling rejected** — persistent 50–85% reflectance offset, not
   fixable by simple ratios (§4.4 `sec:disc_pooling`)
5. **OWT stratification tested, doesn't help** (S2: 0.372→0.369; L8 has no
   variation to test, 100% Type 3; CGSM has too few stations)
6. **Station clustering tested, doesn't reliably help**
   (`code/06_station_clustering.py`) — L8's one apparent gain (k=2,
   R²=0.811 vs 0.784) is outlier-isolation (2 remote stations, 4 obs split
   off), not genuine regionalization; S2 and CGSM get worse or can't be
   tested
7. **Temporal window justification corrected** — the real sensitivity curve
   peaks tighter than originally claimed (±0.75–1.25 days, R²=0.735), then
   declines to R²=0.662 at the ±2.84-day window used, rather than
   "improving then plateauing" (`code/04_temporal_sensitivity.py`)
8. **S1–TSS dilution claim flagged** as confounded by shared long-term
   trends in both series — not yet resolved, would need the raw S1 time
   series to detrend properly (§4.6 `sec:disc_hydro`, not fabricated)
9. **Station clustering tested** (`code/06_station_clustering.py`) and
   **NIR-aware OWT tested** (`code/07_nir_owt_stratification.py`, from an
   idea in an earlier exploratory script) — neither reliably improves
   on the pooled baseline; closed
10. **Power-law alternative tested and reported** (`code/08_powerlaw_comparison.py`,
    §3.3 `sec:results_powerlaw`) — modest, real win for L8, not generalizable;
    Nechad retained as primary model

### Still open

8 of 12 figures need data never available in this session (full multi-year
satellite archive, Sentinel-1 SAR processing, GIS basemap); MDPI class file
and bibliography are not included; funding and data-availability statements
are placeholders. See "What is MISSING" below for the full breakdown.

## Status: NOT submission-ready — read this before compiling

This repo was assembled from a single working session and reflects exactly

what was verified in that session. It is **not** a complete MDPI submission
package. Specifically:

### What is real and verified
- `main.tex` — manuscript source. Every quantitative claim (R², RMSE, bias,
  MAPE, Ap, Cp, kappa values) was computed from the data in `data/` using the
  scripts in `code/`, and cross-checked against saved JSON output before being
  written into the LaTeX. See `code/*.py` to reproduce any number in the text.
- **Major revision: the κ = 0.35 identifiability threshold was found to be
  unvalidated and wrong on this study's own data.** It was inherited as a
  constant from the original project brief and never tested. Bootstrap
  resampling (50 resamples per dataset, re-fitting the free 2-parameter
  Nechad model each time) showed: Landsat-8 at Pajarales (κ=0.343, nominally
  *below* the 0.35 cutoff) actually supports a **stable, better-fitting**
  free-Cp calibration (Cp=0.2732, CV=4.6% across resamples,
  LOSO R²=0.784 — up from R²=0.662 under the old fixed-Cp assumption); CGSM
  (κ=0.129, well below 0.35) does **not** support a free fit
  (CV=49.6%, driven by only 6 of 74 points sitting in the curvature-
  informative region). κ is a single-point statistic (depends only on
  ρ_max) and doesn't capture how many points actually populate the
  curvature zone — that's what the new `n_curv` / bootstrap-CV criterion
  measures directly. **This changes the paper's primary Landsat-8
  calibration number, and required rerunning the full CGSM transfer
  framework** (Table 5): Scenario A (direct transfer) actually gets
  *worse* with the better source (R²=0.448→0.262), because a sharper,
  better-fitting curve is more sensitive to cross-site reflectance offset
  — a genuine, reportable finding, not a regression. See
  `code/05_bootstrap_identifiability.py` and the rewritten
  §2.5.2/§3.1/§3.2/§3.6/§4.1/§4.3 in `main.tex`.
- **Note on Figure 1 / §2.3 / §4.2**: the original draft text claimed
  performance "improved consistently up to ±2.5 days and plateaued."
  Running the actual sensitivity sweep (`code/04_temporal_sensitivity.py`)
  showed the opposite shape — CV R² peaks at a *tighter* window
  (±0.75–1.25 days, R²=0.735) and *declines* gradually to R²=0.662 at the
  full ±2.84-day window used throughout the study. The manuscript text was
  rewritten to match this real result and justify the ±2.5-day window on
  sample-size/fold-stability grounds instead of a false "improves then
  plateaus" claim. If you have an independently-run sensitivity analysis
  that differs from this, check it against `code/04_temporal_sensitivity.py`
  before trusting either.
- `data/` — the five matchup datasets actually used:
  - `pajarales_l8_acolite.csv` — Landsat-8, ACOLITE-corrected, n=110 (primary L8 input)
  - `pajarales_s2_acolite_corrected.csv` — Sentinel-2, ACOLITE with atmospheric-correction
    artefact fixed, n=130 (primary S2 input)
  - `pajarales_s2_acolite_original.csv` — Sentinel-2, uncorrected (r=0.118), kept for
    the before/after comparison in §4.5
  - `pajarales_l8_raw_c2l2.csv` — Landsat-8 raw Collection-2 Level-2 (DN-converted),
    used only for the ACOLITE-vs-raw comparison in §4.3.1
  - `cgsm_l8_acolite.csv` — CGSM validation set, Landsat-8, n=74, 4 stations
- `code/01_pajarales_l8_raw_pipeline.py` — raw C2 L2 Nechad pipeline (§4.3.1 comparison)
- `code/02_candidate_comparison.py` — the four-way candidate comparison (ACOLITE vs
  raw L8; original vs corrected S2) that produced Table 4's numbers
- `code/03_cgsm_transfer_scenarios.py` — Scenarios A–D (Table 5)
- `images/fig2_pajarales_cv_scatter.png` — real, generated from actual LOSO predictions
- `images/fig3_transferability_scenarios.png` — real, generated from actual scenario predictions
- `images/fig5_domain_identifiability.png` — real, generated from actual reflectance distributions and computed kappa values
- `images/methodology.png` — conceptual workflow diagram (not data-derived, built for clarity)

### What is MISSING — figures referenced in `main.tex` with no source file
These eight figures require data that was never part of this working session
(full multi-year satellite archives, Sentinel-1 SAR processing, GIS shapefiles).
**The manuscript will not compile to a complete PDF until these exist:**

| Figure | Needs |
|---|---|
| `images/fig1_temporal_sensitivity.png` | DONE — see `code/04_temporal_sensitivity.py`. Real curve peaks ±0.75-1.25d (R²=0.735), declines to R²=0.662 at ±2.84d. Manuscript text updated to match. |
| `images/map_stamarta.png` | Study area map — GIS shapefiles, basemap |
| `images/TSS_promedio_anual_mosaico_CGSM.png` | Annual median TSS applied to full 2013–2024 satellite archive (not just matchup subset) |
| `images/fig4_monthly_climatology.png` | Full 10-year in-situ time series (matchup subset only covers satellite-paired observations) |
| `images/fig6_annual_anomalies.png` | Same — full annual in-situ record |
| `images/fig7_S1_TSS_verified.png` | Sentinel-1 SAR-derived water surface area time series — no S1 processing done here |
| `images/fig8_hydroclimatic_final.png` | Same, plus ONI index alignment |
| `images/fig9_strengthening_analyses.png` | Mixed: panels (a)/(c) are reproducible from `code/03_...py` output; (b),(d),(e),(f) need additional analysis not run here |

### What also needs attention before submission
- `\funding{}` — placeholder, must be filled in
- `\dataavailability{}` — GitHub/Zenodo URLs are placeholders
- MDPI class file (`Definitions/mdpi.cls`) and `ref.bib` are **not included** —
  pull these from your existing Overleaf/local project; this repo assumes they
  sit alongside `main.tex`
- LaTeX was structurally validated (balanced braces/environments, all `\ref`
  resolve) but **not compiled to PDF** in this environment, since the class
  file and bibliography weren't available here

## Reproducing the verified numbers

```bash
cd code/
python 02_candidate_comparison.py   # → Table 4 (Pajarales calibration)
python 03_cgsm_transfer_scenarios.py  # → Table 5 (transferability)
```

Both scripts read from `../data/` and print every reported metric to stdout.

## Suggested next steps
1. Add the MDPI class files and bibliography, compile locally, fix any
   citation/reference issues that only surface at compile time
2. Generate the eight missing figures from your full archives
3. Fill in funding statement and data-availability URLs
4. Have a co-author review the rewritten Results/Discussion sections
   (§3.2, §3.6, §4.3–4.5) against this repo's `code/` output before
   final submission
