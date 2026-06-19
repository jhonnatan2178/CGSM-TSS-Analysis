"""
Transferability Analysis: Pajarales -> CGSM
=============================================
Source: L8 ACOLITE calibration at Pajarales (Ap=1476.2, Cp=0.55), our verified winner.
Target: CGSM, n=74, Landsat-8 only, 4 stations.

Scenario A - Direct transfer: apply Pajarales Ap, Cp unchanged. No local fitting.
Scenario B - Local NECHAD, free Cp: both Ap and Cp fit on CGSM data, LOSO per station.
Scenario C - Partial transfer: Cp fixed at CP_UPPER (same anchor), Ap refit on CGSM, LOSO.
Scenario D - Local GBR: ML benchmark trained on CGSM's 4 raw bands, LOSO.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

CP_UPPER, CP_LOWER = 0.55, 0.10
FIT_MARGIN = 0.90
RHO_MIN = 0.001
TAU = 1.0

# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────
cgsm = pd.read_csv('data/cgsm_landsat.csv', sep=';')
cgsm['weight'] = np.exp(-np.abs(cgsm['delta_days'])/TAU)
cgsm['station'] = cgsm['lat'].round(3).astype(str) + "," + cgsm['lon'].round(3).astype(str)

# Source (Pajarales L8 ACOLITE) calibration, from our verified pipeline
AP_SOURCE = 1476.2
CP_SOURCE = 0.55

print(f"CGSM: n={len(cgsm)}, stations={cgsm['station'].nunique()}")
print(cgsm['station'].value_counts())

# ─────────────────────────────────────────────────────────────────
# MACHINERY
# ─────────────────────────────────────────────────────────────────
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

def fit_free_Cp(x, y, w, margin=FIT_MARGIN):
    valid = (x >= RHO_MIN) & (x < CP_UPPER*margin) & np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 5:
        raise ValueError("too few points")
    xv, yv, wv = x[valid], y[valid], w[valid]
    best_r2, best_Ap, best_Cp = -np.inf, None, None
    for p0_Cp in [0.15, 0.20, 0.30, 0.40, 0.50]:
        for p0_Ap in [300, 1000, 3000]:
            try:
                popt, _ = curve_fit(
                    nechad, xv, yv, sigma=1.0/(wv+1e-9),
                    p0=[p0_Ap, p0_Cp],
                    bounds=([1.0, CP_LOWER], [1e6, CP_UPPER]),
                    maxfev=30000
                )
                if popt[0] < 1.0: continue
                pred = nechad(xv, *popt)
                fin = np.isfinite(pred)
                if fin.sum() < 2: continue
                r2 = r2_score(yv[fin], pred[fin])
                if r2 > best_r2:
                    best_r2, best_Ap, best_Cp = r2, float(popt[0]), float(popt[1])
            except Exception:
                pass
    if best_Ap is None:
        raise ValueError("2-param fit failed entirely")
    return best_Ap, best_Cp

def metrics(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
    if m.sum() < 2:
        return dict(n=int(m.sum()), r2=np.nan, rmse=np.nan, bias=np.nan, mape=np.nan)
    a, b = yt[m], yp[m]
    return dict(n=int(m.sum()), r2=float(r2_score(a, b)),
                rmse=float(np.sqrt(mean_squared_error(a, b))),
                bias=float(np.mean(b-a)),
                mape=float(np.mean(np.abs((a-b)/a))*100))

x_all = cgsm['b4'].values
y_all = cgsm['concentracion'].values
w_all = cgsm['weight'].values
stations = cgsm['station'].unique()

print(f"\nrho_max={x_all.max():.4f}, kappa={x_all.max()/CP_UPPER:.3f} -- confirms 1-param-only regime for CGSM")

# ─────────────────────────────────────────────────────────────────
# SCENARIO A: Direct transfer, no local adjustment
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO A: Direct transfer (Pajarales Ap, Cp -> CGSM)")
print("="*60)
pm = (x_all >= RHO_MIN) & (x_all < CP_SOURCE*FIT_MARGIN)
pred_A = np.full(len(x_all), np.nan)
pred_A[pm] = nechad(x_all[pm], AP_SOURCE, CP_SOURCE)
m_A = metrics(y_all, pred_A)
print(f"Ap={AP_SOURCE} (from Pajarales, unchanged)  Cp={CP_SOURCE}")
print(f"R²={m_A['r2']:.3f}  RMSE={m_A['rmse']:.1f}  Bias={m_A['bias']:.1f}  MAPE={m_A['mape']:.1f}%  n={m_A['n']}")

# ─────────────────────────────────────────────────────────────────
# SCENARIO B: Local NECHAD, free Cp, LOSO
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO B: Local NECHAD (Cp free), LOSO-CV")
print("="*60)
pred_B = np.full(len(cgsm), np.nan)
for held in stations:
    te = (cgsm['station']==held).values
    tr = ~te
    try:
        Ap_f, Cp_f = fit_free_Cp(x_all[tr], y_all[tr], w_all[tr])
    except Exception as e:
        print(f"  [{held}] free-Cp failed ({e}), falling back to fixed-Cp")
        Ap_f, Cp_f = fit_fixed_Cp(x_all[tr], y_all[tr], w_all[tr])
    pm_te = (x_all[te] >= RHO_MIN) & (x_all[te] < Cp_f*FIT_MARGIN)
    pte = np.full(te.sum(), np.nan)
    if pm_te.any():
        pte[pm_te] = nechad(x_all[te][pm_te], Ap_f, Cp_f)
    pred_B[te] = pte
    print(f"  Held out {held}: Ap={Ap_f:.1f} Cp={Cp_f:.4f} n_test={te.sum()}")
m_B = metrics(y_all, pred_B)
print(f"\nPooled LOSO: R²={m_B['r2']:.3f}  RMSE={m_B['rmse']:.1f}  Bias={m_B['bias']:.1f}  MAPE={m_B['mape']:.1f}%  n={m_B['n']}")

# ─────────────────────────────────────────────────────────────────
# SCENARIO C: Partial transfer, Cp fixed (same anchor), Ap refit locally, LOSO
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO C: Partial transfer (Cp=0.55 fixed, Ap refit on CGSM), LOSO-CV")
print("="*60)
pred_C = np.full(len(cgsm), np.nan)
for held in stations:
    te = (cgsm['station']==held).values
    tr = ~te
    Ap_f, Cp_f = fit_fixed_Cp(x_all[tr], y_all[tr], w_all[tr])
    pm_te = (x_all[te] >= RHO_MIN) & (x_all[te] < Cp_f*FIT_MARGIN)
    pte = np.full(te.sum(), np.nan)
    if pm_te.any():
        pte[pm_te] = nechad(x_all[te][pm_te], Ap_f, Cp_f)
    pred_C[te] = pte
    print(f"  Held out {held}: Ap={Ap_f:.1f} Cp={Cp_f} n_test={te.sum()}")
m_C = metrics(y_all, pred_C)
print(f"\nPooled LOSO: R²={m_C['r2']:.3f}  RMSE={m_C['rmse']:.1f}  Bias={m_C['bias']:.1f}  MAPE={m_C['mape']:.1f}%  n={m_C['n']}")

# ─────────────────────────────────────────────────────────────────
# SCENARIO D: Local GBR, 4 raw bands, LOSO
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO D: Local GBR (4 raw bands), LOSO-CV")
print("="*60)
pred_D = np.full(len(cgsm), np.nan)
features = ['b2','b3','b4','b5']
for held in stations:
    te = (cgsm['station']==held).values
    tr = ~te
    model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(cgsm.loc[tr, features], cgsm.loc[tr, 'concentracion'])
    pred_D[te] = model.predict(cgsm.loc[te, features])
m_D = metrics(y_all, pred_D)
print(f"Pooled LOSO: R²={m_D['r2']:.3f}  RMSE={m_D['rmse']:.1f}  Bias={m_D['bias']:.1f}  MAPE={m_D['mape']:.1f}%  n={m_D['n']}")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("TRANSFERABILITY SUMMARY (n=74, 4 stations, LOSO-CV)")
print("="*72)
print(f"{'Scenario':<45}{'R2':>8}{'RMSE':>9}{'Bias':>9}{'MAPE':>9}")
print(f"{'A. Direct transfer (no refit)':<45}{m_A['r2']:>8.3f}{m_A['rmse']:>9.1f}{m_A['bias']:>9.1f}{m_A['mape']:>9.1f}")
print(f"{'B. Local NECHAD (Cp free)':<45}{m_B['r2']:>8.3f}{m_B['rmse']:>9.1f}{m_B['bias']:>9.1f}{m_B['mape']:>9.1f}")
print(f"{'C. Partial transfer (Cp fixed, Ap local)':<45}{m_C['r2']:>8.3f}{m_C['rmse']:>9.1f}{m_C['bias']:>9.1f}{m_C['mape']:>9.1f}")
print(f"{'D. Local GBR (non-physical ceiling)':<45}{m_D['r2']:>8.3f}{m_D['rmse']:>9.1f}{m_D['bias']:>9.1f}{m_D['mape']:>9.1f}")

# Save
cgsm['TSS_scenA'] = pred_A
cgsm['TSS_scenB'] = pred_B
cgsm['TSS_scenC'] = pred_C
cgsm['TSS_scenD'] = pred_D
cgsm.to_csv('/mnt/user-data/outputs/CGSM_transferability_results.csv', index=False)

import json
summary = {
    "source_calibration": {"Ap": AP_SOURCE, "Cp": CP_SOURCE, "source_loso_r2": 0.662},
    "A_direct_transfer": m_A,
    "B_local_free_Cp": m_B,
    "C_partial_transfer": m_C,
    "D_local_GBR": m_D,
}
with open('cgsm_transfer_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved CGSM_transferability_results.csv and cgsm_transfer_summary.json")
