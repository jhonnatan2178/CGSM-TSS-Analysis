"""
Temporal Matchup Window Sensitivity Analysis
==============================================
Reproduces the analysis behind Figure 1: sweeps the maximum allowed
|delta_days| threshold from a tight same-day-ish cutoff up to the full
dataset's range, and reports leave-one-station-out CV R² at each
threshold, using the same fixed-Cp Nechad pipeline as the main
calibration (Landsat-8, Pajarales, B4, Cp=0.55).

This is an honest re-use of the existing Pajarales L8 ACOLITE matchup
dataset (data/pajarales_l8_acolite.csv) -- it does not require new data,
only re-running LOSO-CV at each of several |delta_days| cutoffs.

NOTE: the underlying dataset only contains matchups already filtered to
|delta_days| <= 2.84 days (the matchup window used to build this
dataset in the first place). This script can show how CV R2 behaves
as that cutoff is tightened from the existing maximum down to a small
value -- it CANNOT show what would happen with a wider window than the
data already used, since no observations beyond 2.84 days exist in
this file. This matches the manuscript's claim ("all matchups in the
final dataset fall within ±2.5 days, maximum observed |delta_days| =
2.84") but means the upper end of the sweep is necessarily flat by
construction, not independently re-verified out to 3.0 days.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

CP_UPPER = 0.55
FIT_MARGIN = 0.90
RHO_MIN = 0.001
TAU = 1.0
BAND = "SR_B4"

df_full = pd.read_csv('../data/pajarales_l8_acolite.csv', sep=';')
print(f"Full Pajarales L8 dataset: n={len(df_full)}")
print(f"delta_days range: {df_full['delta_days'].min():.3f} - {df_full['delta_days'].max():.3f}")

def nechad(rho, Ap, Cp=CP_UPPER):
    denom = 1.0 - rho/Cp
    denom = np.where(np.abs(denom) > 1e-6, denom, np.nan)
    return Ap*rho/denom

def fit_fixed_Cp(x, y, w, Cp=CP_UPPER, margin=FIT_MARGIN):
    valid = (x >= RHO_MIN) & (x < Cp*margin) & np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 2:
        raise ValueError("too few points")
    xv, yv, wv = x[valid], y[valid], w[valid]
    feat = xv/(1.0 - xv/Cp)
    Ap = float(np.sum(wv*yv*feat)/np.sum(wv*feat**2))
    return Ap, Cp

def metrics(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
    if m.sum() < 2:
        return dict(n=int(m.sum()), r2=np.nan, rmse=np.nan)
    a, b = yt[m], yp[m]
    return dict(n=int(m.sum()), r2=float(r2_score(a, b)),
                rmse=float(np.sqrt(mean_squared_error(a, b))))

def station_id(lat, lon):
    try:
        la, lo = float(lat), float(lon)
        if np.isfinite(la) and np.isfinite(lo):
            return f"{la:.3f},{lo:.3f}"
    except Exception:
        pass
    return "NO_LOCATION"

def run_loso(df):
    df = df.copy()
    if len(df) < 15:
        return None
    df['station'] = df.apply(lambda r: station_id(r['lat'], r['lon']), axis=1)
    df['weight'] = np.exp(-np.abs(df['delta_days'])/TAU)
    x_all = df[BAND].values
    y_all = df['concentracion'].values
    w_all = df['weight'].values
    stations = [s for s in df['station'].unique() if s != "NO_LOCATION"]
    if len(stations) < 3:
        return None
    pred_cv = np.full(len(df), np.nan)
    for held in stations:
        te = (df['station']==held).values
        tr = ~te
        if tr.sum() < 5:
            continue
        try:
            Ap_f, Cp_f = fit_fixed_Cp(x_all[tr], y_all[tr], w_all[tr])
        except Exception:
            continue
        pm = (x_all[te] >= RHO_MIN) & (x_all[te] < Cp_f*FIT_MARGIN)
        pte = np.full(te.sum(), np.nan)
        if pm.any():
            pte[pm] = nechad(x_all[te][pm], Ap_f, Cp_f)
        pred_cv[te] = pte
    return metrics(y_all, pred_cv)

# Sweep thresholds. Upper bound is necessarily capped at the data's own
# max |delta_days| (2.84) -- see module docstring.
thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.84]
results = []
print("\nSweeping |delta_days| <= threshold:")
for t in thresholds:
    sub = df_full[df_full['delta_days'] <= t]
    m = run_loso(sub)
    if m is None:
        print(f"  threshold={t:.2f}: n={len(sub)} -- too few stations/points, skipped")
        continue
    results.append({"threshold": t, "n": len(sub), **m})
    print(f"  threshold={t:.2f}: n={len(sub):3d}  LOSO R²={m['r2']:.3f}  RMSE={m['rmse']:.1f}  (n_scored={m['n']})")

res_df = pd.DataFrame(results)
res_df.to_csv('../data/temporal_sensitivity_results.csv', index=False)

# Figure
fig, ax1 = plt.subplots(figsize=(7.5, 5))
ax2 = ax1.twinx()

ax1.plot(res_df['threshold'], res_df['r2'], 'o-', color='#1565C0', lw=2, markersize=7, label='LOSO-CV R²')
ax1.set_xlabel("Maximum matchup window |Δt| (days)", fontsize=11)
ax1.set_ylabel("LOSO-CV R²", fontsize=11, color='#1565C0')
ax1.tick_params(axis='y', labelcolor='#1565C0')
ax1.axvline(2.5, color='gray', ls='--', lw=1.2, alpha=0.7)
ax1.text(2.55, ax1.get_ylim()[0]+0.02, "selected\nwindow", fontsize=8.5, color='gray')

ax2.bar(res_df['threshold'], res_df['n'], width=0.15, alpha=0.25, color='#9E9E9E', label='n matchups')
ax2.set_ylabel("n matchups", fontsize=11, color='#757575')
ax2.tick_params(axis='y', labelcolor='#757575')

ax1.set_title("Landsat-8 LOSO-CV performance vs.\nmaximum satellite–\\textit{in situ} matchup lag", fontsize=11)
fig.tight_layout()
fig.savefig('../images/fig1_temporal_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure saved to ../images/fig1_temporal_sensitivity.png")
print("\nIMPORTANT: see module docstring -- this sweep narrows the EXISTING")
print("matchup set, it does not independently verify performance at windows")
print("wider than what the dataset already used (max 2.84 days).")
