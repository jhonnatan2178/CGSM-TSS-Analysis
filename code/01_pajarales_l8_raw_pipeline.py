"""
TSS Retrieval Pipeline — Raw C2 L2 Landsat-8, Pajarales calibration
====================================================================
Implements the verified pipeline from the project specification:
  - BAND = SR_B4 throughout
  - CP_UPPER = 0.55 (not Nechad 2010 value)
  - OWT from band ratios only (no TSS stratification)
  - Identifiability: kappa = rho_max / CP_UPPER
      kappa < 0.35 → 1-param (Cp fixed at CP_UPPER)
      kappa >= 0.35 → 2-param (Ap and Cp free, bounded [0.10, 0.55])
  - FIT_MARGIN = 0.90 (exclude points within 10% of Cp)
  - TAU = 1.0 temporal weight: w = exp(-|delta_t| / TAU)
  - Leave-one-station-out CV (station = lat.round(3), lon.round(3))
  - Aerosol filter: drop rows where SR_B1 > 0.5 (all from 21/02/2024)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

BAND          = "SR_B4"
CP_UPPER      = 0.55
CP_LOWER      = 0.10
CURV_THRESH   = 0.35      # kappa threshold
FIT_MARGIN    = 0.90      # exclude x >= Cp * FIT_MARGIN during fitting
TAU           = 1.0       # temporal weight decay
RHO_MIN       = 0.001
MIN_SAMPLES   = 5

INPUT_FILE    = "/mnt/user-data/uploads/full_results_pajarales1.csv"
OUT_DIR       = "/mnt/user-data/outputs"

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD + FILTER
# ═══════════════════════════════════════════════════════════════════

df_raw = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df_raw)} rows")

# Aerosol filter: SR_B1 > 0.5 after DN→reflectance conversion
# (conversion was already applied in this file)
aerosol_mask = df_raw["SR_B1"] > 0.5
print(f"Aerosol-contaminated rows (SR_B1>0.5): {aerosol_mask.sum()} "
      f"(dates: {df_raw.loc[aerosol_mask, 'sat_date'].unique().tolist()})")

df = df_raw[~aerosol_mask].copy().reset_index(drop=True)
print(f"After aerosol filter: {len(df)} rows")

# ═══════════════════════════════════════════════════════════════════
# 2. TEMPORAL WEIGHTS
# ═══════════════════════════════════════════════════════════════════

df["weight"] = np.exp(-np.abs(df["Diferencia_dias"]) / TAU)

# ═══════════════════════════════════════════════════════════════════
# 3. OWT CLASSIFICATION (band ratios only, NEVER from TSS)
# ═══════════════════════════════════════════════════════════════════

def classify_owt(b2, b3, b4):
    """
    Type 3: b4 > b3 > b2  (NIR-dominated, very turbid)
    Type 2: b3 >= b4 > b2  (red-green transition)
    Type 1: otherwise       (clear-ish, blue-green)
    """
    if b4 > b3 and b3 > b2:
        return "Type 3"
    elif b3 >= b4 and b4 > b2:
        return "Type 2"
    else:
        return "Type 1"

df["OWT"] = df.apply(
    lambda r: classify_owt(r["SR_B2"], r["SR_B3"], r[BAND]), axis=1
)

print(f"\nOWT distribution (band-ratio only):\n{df['OWT'].value_counts()}")
print(f"\nTSS range: {df['concentracion'].min():.1f} – "
      f"{df['concentracion'].max():.1f} mg/L")
print(f"B4 range:  {df[BAND].min():.4f} – {df[BAND].max():.4f}")

# ═══════════════════════════════════════════════════════════════════
# 4. STATION ASSIGNMENT (for leave-one-station-out CV)
# ═══════════════════════════════════════════════════════════════════

def make_station_id(lat, lon):
    try:
        flat = float(lat)
        flon = float(lon)
        if np.isfinite(flat) and np.isfinite(flon):
            return f"{flat:.3f},{flon:.3f}"
    except (ValueError, TypeError):
        pass
    return "NO_LOCATION"

df["station"] = df.apply(
    lambda r: make_station_id(r["lat"], r["lon"]), axis=1
)

stations = df["station"].unique()
print(f"\nStations ({len(stations)}):")
for s in sorted(stations):
    n = (df["station"] == s).sum()
    print(f"  {s}: n={n}")

# ═══════════════════════════════════════════════════════════════════
# 5. IDENTIFIABILITY CHECK (global, computed once)
# ═══════════════════════════════════════════════════════════════════

valid_rho = df[BAND][(df[BAND] >= RHO_MIN) & np.isfinite(df[BAND])]
rho_max_global = valid_rho.max()
kappa_global   = rho_max_global / CP_UPPER

print(f"\n── Identifiability ──")
print(f"rho_max (global) = {rho_max_global:.4f}")
print(f"kappa = rho_max / CP_UPPER = {kappa_global:.4f}")
if kappa_global >= CURV_THRESH:
    print(f"kappa >= {CURV_THRESH} → 2-param fit (Ap + Cp free)")
else:
    print(f"kappa < {CURV_THRESH} → 1-param fit (Cp fixed at {CP_UPPER})")

# ═══════════════════════════════════════════════════════════════════
# 6. NECHAD MODEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def nechad_predict(rho, Ap, Cp):
    denom = 1.0 - rho / Cp
    denom = np.where(np.abs(denom) > 1e-6, denom, np.nan)
    return Ap * rho / denom


def fit_fixed_Cp(x, y, w, Cp, margin=FIT_MARGIN):
    """Closed-form weighted least squares with Cp fixed."""
    valid = (x >= RHO_MIN) & (x < Cp * margin) & np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 2:
        raise ValueError(f"Too few valid points: {valid.sum()}")
    xv, yv, wv = x[valid], y[valid], w[valid]
    feat = xv / (1.0 - xv / Cp)
    Ap = np.sum(wv * yv * feat) / np.sum(wv * feat**2)
    return float(Ap), valid


def fit_free_Cp(x, y, w, margin=FIT_MARGIN):
    """Scipy curve_fit with both Ap and Cp free."""
    valid = (x >= RHO_MIN) & (x < CP_UPPER * margin) & np.isfinite(y) & np.isfinite(x)
    if valid.sum() < MIN_SAMPLES:
        raise ValueError(f"Too few valid points: {valid.sum()}")
    xv, yv, wv = x[valid], y[valid], w[valid]
    sigma = 1.0 / (wv + 1e-9)
    popt, _ = curve_fit(
        nechad_predict, xv, yv,
        sigma=sigma,
        p0=[500.0, 0.30],
        bounds=([1e-3, CP_LOWER], [1e6, CP_UPPER]),
        maxfev=50000
    )
    return float(popt[0]), float(popt[1])


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return dict(n=len(yt), r2=np.nan, rmse=np.nan, bias=np.nan, mape=np.nan)
    return dict(
        n    = len(yt),
        r2   = r2_score(yt, yp),
        rmse = float(np.sqrt(mean_squared_error(yt, yp))),
        bias = float(np.mean(yp - yt)),
        mape = float(np.mean(np.abs((yt - yp) / yt)) * 100),
    )

# ═══════════════════════════════════════════════════════════════════
# 7. GLOBAL FIT (in-sample calibration)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("IN-SAMPLE CALIBRATION (all n=104)")
print("="*65)

x_all = df[BAND].values
y_all = df["concentracion"].values
w_all = df["weight"].values

# Choose fit mode based on global kappa
if kappa_global >= CURV_THRESH:
    Ap_global, Cp_global = fit_free_Cp(x_all, y_all, w_all)
    fit_mode = "2-param"
else:
    Ap_global, _  = fit_fixed_Cp(x_all, y_all, w_all, CP_UPPER)
    Cp_global = CP_UPPER
    fit_mode = "1-param"

print(f"Fit mode : {fit_mode}")
print(f"Ap       = {Ap_global:.2f}")
print(f"Cp       = {Cp_global:.4f}")

# Predict (apply margin in prediction too — avoid asymptote blowups)
pred_margin_mask = (x_all >= RHO_MIN) & (x_all < Cp_global * FIT_MARGIN)
df["TSS_insample"] = np.nan
df.loc[pred_margin_mask, "TSS_insample"] = nechad_predict(
    x_all[pred_margin_mask], Ap_global, Cp_global
)

m_insample = compute_metrics(
    df["concentracion"].values,
    df["TSS_insample"].values
)
print(f"\nIn-sample: R²={m_insample['r2']:.3f}  "
      f"RMSE={m_insample['rmse']:.1f} mg/L  "
      f"Bias={m_insample['bias']:.1f}  "
      f"MAPE={m_insample['mape']:.1f}%  "
      f"n={m_insample['n']}")

# ═══════════════════════════════════════════════════════════════════
# 8. LEAVE-ONE-STATION-OUT CV
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("LEAVE-ONE-STATION-OUT CROSS-VALIDATION")
print("="*65)

df["TSS_cv"] = np.nan
df["Ap_cv"]  = np.nan
df["Cp_cv"]  = np.nan

loso_results = []

# "NO_LOCATION" rows always stay in training
no_loc_mask = df["station"] == "NO_LOCATION"
real_stations = [s for s in stations if s != "NO_LOCATION"]

for held_out in real_stations:
    test_mask  = df["station"] == held_out
    train_mask = ~test_mask | no_loc_mask   # NO_LOCATION always trains
    # Correct: train = not-held-out (including NO_LOCATION)
    train_mask = (df["station"] != held_out) | no_loc_mask
    # Simpler: train = everything that is NOT the held-out station
    train_mask = df["station"] != held_out

    x_tr = df.loc[train_mask, BAND].values
    y_tr = df.loc[train_mask, "concentracion"].values
    w_tr = df.loc[train_mask, "weight"].values
    x_te = df.loc[test_mask,  BAND].values
    y_te = df.loc[test_mask,  "concentracion"].values

    n_tr = train_mask.sum()
    n_te = test_mask.sum()

    # Per-fold identifiability
    valid_tr = (x_tr >= RHO_MIN) & np.isfinite(x_tr)
    if valid_tr.sum() == 0:
        print(f"  [{held_out}] SKIP — no valid training reflectances")
        continue

    kappa_fold = x_tr[valid_tr].max() / CP_UPPER

    try:
        if kappa_fold >= CURV_THRESH:
            Ap_fold, Cp_fold = fit_free_Cp(x_tr, y_tr, w_tr)
            mode_fold = "2-param"
        else:
            Ap_fold, _ = fit_fixed_Cp(x_tr, y_tr, w_tr, CP_UPPER)
            Cp_fold = CP_UPPER
            mode_fold = "1-param"
    except Exception as e:
        print(f"  [{held_out}] FIT FAILED: {e}")
        continue

    # Predict held-out station (with margin)
    pred_mask_te = (x_te >= RHO_MIN) & (x_te < Cp_fold * FIT_MARGIN)
    pred_te = np.full(len(x_te), np.nan)
    if pred_mask_te.any():
        pred_te[pred_mask_te] = nechad_predict(
            x_te[pred_mask_te], Ap_fold, Cp_fold
        )

    df.loc[test_mask, "TSS_cv"] = pred_te
    df.loc[test_mask, "Ap_cv"]  = Ap_fold
    df.loc[test_mask, "Cp_cv"]  = Cp_fold

    m_fold = compute_metrics(y_te, pred_te)
    loso_results.append({
        "station": held_out, "n_train": n_tr, "n_test": n_te,
        "kappa": kappa_fold, "mode": mode_fold,
        "Ap": Ap_fold, "Cp": Cp_fold, **m_fold
    })
    print(f"  Hold-out [{held_out}] n_test={n_te} | "
          f"{mode_fold} | Ap={Ap_fold:.1f} Cp={Cp_fold:.4f} | "
          f"R²={m_fold['r2']:.3f} RMSE={m_fold['rmse']:.1f}")

loso_df = pd.DataFrame(loso_results)

# Overall LOSO metrics (pool all held-out predictions)
cv_valid = df["TSS_cv"].notna()
m_loso = compute_metrics(
    df.loc[cv_valid, "concentracion"].values,
    df.loc[cv_valid, "TSS_cv"].values
)

print(f"\nLOSO-CV (pooled): R²={m_loso['r2']:.3f}  "
      f"RMSE={m_loso['rmse']:.1f} mg/L  "
      f"Bias={m_loso['bias']:.1f}  "
      f"MAPE={m_loso['mape']:.1f}%  "
      f"n={m_loso['n']}")

# ═══════════════════════════════════════════════════════════════════
# 9. CORRELATION CHECK
# ═══════════════════════════════════════════════════════════════════

valid_b4 = df[BAND][(df[BAND] >= RHO_MIN)] 
r_b4_tss = df.loc[df[BAND] >= RHO_MIN, [BAND, "concentracion"]].corr().iloc[0, 1]
print(f"\nCorrelation r(B4, TSS) after aerosol filter: {r_b4_tss:.3f}")
print(f"B4 range after filter: {df[BAND].min():.4f} – {df[BAND].max():.4f}")

# ═══════════════════════════════════════════════════════════════════
# 10. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════

import os
os.makedirs(OUT_DIR, exist_ok=True)

# Full results
out_cols = ["sat_date", "fecha", "Diferencia_dias", "lat", "lon",
            "SR_B2", "SR_B3", BAND, "SR_B5",
            "concentracion", "OWT", "weight", "station",
            "TSS_insample", "TSS_cv", "Ap_cv", "Cp_cv"]
out_cols = [c for c in out_cols if c in df.columns]
df[out_cols].to_csv(f"{OUT_DIR}/pajarales_raw_results.csv", index=False)

# LOSO per-station summary
loso_df.to_csv(f"{OUT_DIR}/pajarales_loso_stations.csv", index=False)

print(f"\nFiles saved to {OUT_DIR}")

# ═══════════════════════════════════════════════════════════════════
# 11. FIGURES
# ═══════════════════════════════════════════════════════════════════

OWT_COLORS = {"Type 1": "#1E88E5", "Type 2": "#FFA000", "Type 3": "#D32F2F"}
STATION_MARKERS = ["o", "s", "^", "D", "v", "P", "*"]

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

# ── A: Reflectance vs TSS scatter (B4) ──────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
for owt, grp in df.groupby("OWT"):
    ax_a.scatter(grp[BAND], grp["concentracion"],
                 c=OWT_COLORS.get(owt, "gray"), alpha=0.55, s=28, label=owt)
rho_line = np.linspace(RHO_MIN, Cp_global * FIT_MARGIN * 0.99, 300)
ax_a.plot(rho_line, nechad_predict(rho_line, Ap_global, Cp_global),
          "k-", lw=2, label=f"Nechad fit\nAp={Ap_global:.0f}, Cp={Cp_global:.3f}")
ax_a.axvline(Cp_global * FIT_MARGIN, color="gray", ls="--", lw=1, alpha=0.6,
             label=f"Fit margin ({FIT_MARGIN*100:.0f}% Cp)")
ax_a.set_xlabel("SR_B4 (reflectance)", fontsize=9)
ax_a.set_ylabel("TSS (mg/L)", fontsize=9)
ax_a.set_title("B4 vs TSS — Nechad Fit", fontsize=10, fontweight="bold")
ax_a.legend(fontsize=7)
ax_a.set_xlim(0, 0.40)

# ── B: In-sample scatter ─────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ins_valid = df["TSS_insample"].notna()
for owt, grp in df[ins_valid].groupby("OWT"):
    ax_b.scatter(grp["concentracion"], grp["TSS_insample"],
                 c=OWT_COLORS.get(owt, "gray"), alpha=0.55, s=28, label=owt)
mx = max(df.loc[ins_valid, "concentracion"].max(),
         df.loc[ins_valid, "TSS_insample"].max()) * 1.05
ax_b.plot([0, mx], [0, mx], "k--", lw=1.2)
ax_b.set_xlabel("Measured TSS (mg/L)", fontsize=9)
ax_b.set_ylabel("Estimated TSS (mg/L)", fontsize=9)
ax_b.set_title(f"In-sample  R²={m_insample['r2']:.3f}  "
               f"RMSE={m_insample['rmse']:.1f}", fontsize=10, fontweight="bold")
ax_b.legend(fontsize=7)

# ── C: LOSO-CV scatter, colored by station ───────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
cv_rows = df[df["TSS_cv"].notna()]
station_list = sorted(cv_rows["station"].unique())
for i, st in enumerate(station_list):
    grp = cv_rows[cv_rows["station"] == st]
    mk = STATION_MARKERS[i % len(STATION_MARKERS)]
    ax_c.scatter(grp["concentracion"], grp["TSS_cv"],
                 marker=mk, alpha=0.65, s=32, label=st[:18])
mx2 = max(cv_rows["concentracion"].max(), cv_rows["TSS_cv"].max()) * 1.05
ax_c.plot([0, mx2], [0, mx2], "k--", lw=1.2)
ax_c.set_xlabel("Measured TSS (mg/L)", fontsize=9)
ax_c.set_ylabel("CV-Predicted TSS (mg/L)", fontsize=9)
ax_c.set_title(f"LOSO-CV  R²={m_loso['r2']:.3f}  "
               f"RMSE={m_loso['rmse']:.1f}", fontsize=10, fontweight="bold")
ax_c.legend(fontsize=6, loc="upper left")

# ── D: Residuals vs measured ─────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
cv_rows2 = df[df["TSS_cv"].notna()].copy()
cv_rows2["residual"] = cv_rows2["TSS_cv"] - cv_rows2["concentracion"]
for owt, grp in cv_rows2.groupby("OWT"):
    ax_d.scatter(grp["concentracion"], grp["residual"],
                 c=OWT_COLORS.get(owt, "gray"), alpha=0.55, s=28, label=owt)
ax_d.axhline(0, color="k", lw=1.2, ls="--")
ax_d.set_xlabel("Measured TSS (mg/L)", fontsize=9)
ax_d.set_ylabel("Residual (Predicted − Measured)", fontsize=9)
ax_d.set_title("LOSO-CV Residuals by OWT", fontsize=10, fontweight="bold")
ax_d.legend(fontsize=7)

# ── E: B4 histogram by OWT ───────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
for owt, grp in df.groupby("OWT"):
    ax_e.hist(grp[BAND], bins=20, alpha=0.5,
              color=OWT_COLORS.get(owt, "gray"), label=owt, density=True)
ax_e.axvline(Cp_global * FIT_MARGIN, color="k", ls="--", lw=1.2,
             label=f"Fit margin")
ax_e.set_xlabel("SR_B4", fontsize=9)
ax_e.set_ylabel("Density", fontsize=9)
ax_e.set_title("B4 Distribution by OWT", fontsize=10, fontweight="bold")
ax_e.legend(fontsize=7)

# ── F: LOSO per-station R² bar chart ─────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
if len(loso_df) > 0:
    loso_sorted = loso_df.dropna(subset=["r2"]).sort_values("r2", ascending=True)
    colors_bar  = [OWT_COLORS["Type 1"]] * len(loso_sorted)
    bars = ax_f.barh(range(len(loso_sorted)), loso_sorted["r2"],
                     color="#1E88E5", alpha=0.75)
    ax_f.set_yticks(range(len(loso_sorted)))
    ax_f.set_yticklabels(
        [f"{r['station']} (n={int(r['n'])})"
         for _, r in loso_sorted.iterrows()], fontsize=7)
    ax_f.axvline(m_loso["r2"], color="red", ls="--", lw=1.5,
                 label=f"Overall R²={m_loso['r2']:.3f}")
    ax_f.set_xlabel("R²", fontsize=9)
    ax_f.set_title("LOSO-CV R² by Station", fontsize=10, fontweight="bold")
    ax_f.legend(fontsize=8)

# ── G: Nechad curves — sensitivity to Ap variation ───────────────────
ax_g = fig.add_subplot(gs[2, 0:2])
rho_g = np.linspace(RHO_MIN, Cp_global * FIT_MARGIN * 0.98, 400)
for frac, alpha in [(0.5, 0.3), (0.75, 0.5), (1.0, 1.0), (1.25, 0.5), (1.5, 0.3)]:
    Ap_test = Ap_global * frac
    label = f"Ap={Ap_test:.0f}" if frac == 1.0 else f"Ap={Ap_test:.0f} ({frac:.2f}×)"
    ax_g.plot(rho_g, nechad_predict(rho_g, Ap_test, Cp_global),
              lw=2 if frac == 1.0 else 1.2,
              ls="-" if frac == 1.0 else "--",
              alpha=alpha, label=label)
ax_g.scatter(df[BAND], df["concentracion"], c="lightgray", s=18, alpha=0.4, zorder=0)
ax_g.set_xlabel("SR_B4 (reflectance)", fontsize=9)
ax_g.set_ylabel("TSS (mg/L)", fontsize=9)
ax_g.set_title("Nechad Sensitivity to Ap  (Cp fixed)", fontsize=10, fontweight="bold")
ax_g.legend(fontsize=7)
ax_g.set_xlim(0, 0.40)
ax_g.set_ylim(0, df["concentracion"].max() * 1.1)

# ── H: Summary text box ───────────────────────────────────────────────
ax_h = fig.add_subplot(gs[2, 2])
ax_h.axis("off")
summary = (
    "PIPELINE SUMMARY\n"
    "─────────────────────────────\n"
    f"Input:       Raw C2 L2 (DN→ρ)\n"
    f"Band:        SR_B4\n"
    f"Aerosol flt: SR_B1 > 0.5 (n=6)\n"
    f"n final:     {len(df)}\n"
    f"CP_UPPER:    {CP_UPPER}\n"
    f"FIT_MARGIN:  {FIT_MARGIN}\n"
    f"TAU:         {TAU} day\n\n"
    f"kappa:       {kappa_global:.3f}\n"
    f"Fit mode:    {fit_mode}\n"
    f"Ap:          {Ap_global:.1f}\n"
    f"Cp:          {Cp_global:.4f}\n\n"
    f"IN-SAMPLE\n"
    f"  R²   = {m_insample['r2']:.3f}\n"
    f"  RMSE = {m_insample['rmse']:.1f} mg/L\n"
    f"  Bias = {m_insample['bias']:.1f}\n\n"
    f"LOSO-CV (pooled)\n"
    f"  R²   = {m_loso['r2']:.3f}\n"
    f"  RMSE = {m_loso['rmse']:.1f} mg/L\n"
    f"  Bias = {m_loso['bias']:.1f}\n"
    f"  MAPE = {m_loso['mape']:.1f}%\n"
    f"  n    = {m_loso['n']}\n\n"
    f"r(B4,TSS) = {r_b4_tss:.3f}\n"
    f"(after aerosol filter)"
)
ax_h.text(0.05, 0.97, summary, transform=ax_h.transAxes,
          fontsize=8.5, va="top", fontfamily="monospace",
          bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle(
    "Pajarales TSS Calibration — Raw Landsat-8 C2 L2\n"
    "Nechad Semi-analytical Model | Leave-One-Station-Out CV",
    fontsize=13, fontweight="bold"
)

fig.savefig(f"{OUT_DIR}/pajarales_raw_calibration.png",
            dpi=160, bbox_inches="tight")
plt.close()
print("Figure saved.")

# ═══════════════════════════════════════════════════════════════════
# 12. PRINT FINAL TABLE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("FINAL RESULTS TABLE")
print("="*65)
print(f"{'Metric':<25} {'In-sample':>12} {'LOSO-CV':>12}")
print("-"*50)
for key, label in [("r2","R²"), ("rmse","RMSE (mg/L)"),
                   ("bias","Bias (mg/L)"), ("mape","MAPE (%)")]:
    print(f"{label:<25} {m_insample[key]:>12.3f} {m_loso[key]:>12.3f}")
print(f"{'n':<25} {m_insample['n']:>12} {m_loso['n']:>12}")
print("\nGlobal parameters:")
print(f"  Ap = {Ap_global:.2f}   Cp = {Cp_global:.4f}   kappa = {kappa_global:.3f}")
print(f"  Fit mode: {fit_mode}")

print("\nPer-station LOSO breakdown:")
print(loso_df[["station","n_test","Ap","Cp","r2","rmse","bias"]].to_string(index=False))
