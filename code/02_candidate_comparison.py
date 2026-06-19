"""
Pajarales TSS — Candidate Pipeline Comparison
================================================
Goal: find the best-performing, defensible Nechad calibration approach
given all data variants now available:

  L8 candidates:
    - ACOLITE  (doc3_landsat.csv)            n=110
    - Raw C2 L2 (full_results_pajarales1.csv) n=110 -> 104 after aerosol filter

  S2 candidates:
    - ACOLITE original  (doc4_sentinel.csv)            n=130, r=0.118
    - ACOLITE corrected (sentinel2_pajarales_flagged)  n=130, r=0.622

Fixed throughout: BAND=SR_B4, CP_UPPER=0.55, FIT_MARGIN=0.90, TAU=1.0,
identifiability kappa=rho_max/CP_UPPER >=0.35 -> 2-param else 1-param,
leave-one-station-out CV.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

CP_UPPER, CP_LOWER = 0.55, 0.10
CURV_THRESH = 0.35
FIT_MARGIN = 0.90
TAU = 1.0
RHO_MIN = 0.001
MIN_SAMPLES = 5
BAND = "SR_B4"

# ─────────────────────────────────────────────────────────────────
# LOAD ALL VARIANTS
# ─────────────────────────────────────────────────────────────────

# L8 ACOLITE
l8_acolite = pd.read_csv('data/doc3_landsat.csv', sep=';')
l8_acolite['weight'] = np.exp(-np.abs(l8_acolite['delta_days'])/TAU)

# L8 raw C2 L2 (apply aerosol filter)
l8_raw_full = pd.read_csv('/mnt/user-data/uploads/full_results_pajarales1.csv')
l8_raw = l8_raw_full[l8_raw_full['SR_B1'] <= 0.5].copy().reset_index(drop=True)
l8_raw['weight'] = np.exp(-np.abs(l8_raw['Diferencia_dias'])/TAU)

# S2 ACOLITE original
s2_orig = pd.read_csv('data/doc4_sentinel.csv', sep=';')
s2_orig['weight'] = np.exp(-np.abs(s2_orig['delta_days'])/TAU)

# S2 ACOLITE corrected (user-fixed)
s2_corr = pd.read_csv('/mnt/user-data/uploads/sentinel2_pajarales_flagged.csv', sep=';')
s2_corr['weight'] = np.exp(-np.abs(s2_corr['delta_days'])/TAU)

print("Dataset sizes:")
print(f"  L8 ACOLITE:      n={len(l8_acolite)}")
print(f"  L8 raw C2 L2:    n={len(l8_raw)} (after aerosol filter)")
print(f"  S2 original:     n={len(s2_orig)}")
print(f"  S2 corrected:    n={len(s2_corr)}")

# ─────────────────────────────────────────────────────────────────
# NECHAD MACHINERY
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
    return Ap, Cp, valid

def fit_free_Cp(x, y, w, margin=FIT_MARGIN):
    """Robust 2-param fit: multi-start, reject degenerate (Ap<1) results."""
    valid = (x >= RHO_MIN) & (x < CP_UPPER*margin) & np.isfinite(y) & np.isfinite(x)
    if valid.sum() < MIN_SAMPLES:
        raise ValueError("too few points")
    xv, yv, wv = x[valid], y[valid], w[valid]
    best_r2, best_Ap, best_Cp = -np.inf, None, None
    for p0_Cp in [0.40, 0.45, 0.50, 0.30, 0.35, 0.20]:
        for p0_Ap in [300, 500, 1000, 2000]:
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
    return best_Ap, best_Cp, valid

def metrics(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
    if m.sum() < 2:
        return dict(n=int(m.sum()), r2=np.nan, rmse=np.nan, bias=np.nan, mape=np.nan)
    a, b = yt[m], yp[m]
    return dict(n=int(m.sum()), r2=float(r2_score(a, b)),
                rmse=float(np.sqrt(mean_squared_error(a, b))),
                bias=float(np.mean(b-a)),
                mape=float(np.mean(np.abs((a-b)/a))*100))

def station_id(lat, lon):
    try:
        la, lo = float(lat), float(lon)
        if np.isfinite(la) and np.isfinite(lo):
            return f"{la:.3f},{lo:.3f}"
    except Exception:
        pass
    return "NO_LOCATION"

def run_loso(df, band_col, lat_col, lon_col, tss_col='concentracion'):
    """Full leave-one-station-out pipeline for a single dataset."""
    df = df.copy()
    df['station'] = df.apply(lambda r: station_id(r[lat_col], r[lon_col]), axis=1)
    x_all = df[band_col].values
    y_all = df[tss_col].values
    w_all = df['weight'].values

    valid_rho = x_all[(x_all >= RHO_MIN) & np.isfinite(x_all)]
    if len(valid_rho) == 0:
        return None
    rho_max = valid_rho.max()
    kappa = rho_max/CP_UPPER

    # in-sample global fit
    try:
        if kappa >= CURV_THRESH:
            try:
                Ap_g, Cp_g, _ = fit_free_Cp(x_all, y_all, w_all)
                mode_g = "2-param"
            except Exception:
                Ap_g, Cp_g, _ = fit_fixed_Cp(x_all, y_all, w_all)
                mode_g = "1-param(fb)"
        else:
            Ap_g, Cp_g, _ = fit_fixed_Cp(x_all, y_all, w_all)
            mode_g = "1-param"
    except Exception as e:
        return None

    pm = (x_all >= RHO_MIN) & (x_all < Cp_g*FIT_MARGIN)
    pred_ins = np.full(len(x_all), np.nan)
    pred_ins[pm] = nechad(x_all[pm], Ap_g, Cp_g)
    m_ins = metrics(y_all, pred_ins)

    # LOSO
    stations = df['station'].unique()
    real_stations = [s for s in stations if s != "NO_LOCATION"]
    pred_cv = np.full(len(df), np.nan)
    n_stations_fit = 0
    for held in real_stations:
        te_mask = (df['station'] == held).values
        tr_mask = ~te_mask
        x_tr, y_tr, w_tr = x_all[tr_mask], y_all[tr_mask], w_all[tr_mask]
        x_te = x_all[te_mask]

        vr = x_tr[(x_tr >= RHO_MIN) & np.isfinite(x_tr)]
        if len(vr) == 0: continue
        kf = vr.max()/CP_UPPER
        try:
            if kf >= CURV_THRESH:
                try:
                    Ap_f, Cp_f, _ = fit_free_Cp(x_tr, y_tr, w_tr)
                except Exception:
                    Ap_f, Cp_f, _ = fit_fixed_Cp(x_tr, y_tr, w_tr)
            else:
                Ap_f, Cp_f, _ = fit_fixed_Cp(x_tr, y_tr, w_tr)
        except Exception:
            continue
        n_stations_fit += 1
        pm_te = (x_te >= RHO_MIN) & (x_te < Cp_f*FIT_MARGIN)
        pred_te = np.full(len(x_te), np.nan)
        if pm_te.any():
            pred_te[pm_te] = nechad(x_te[pm_te], Ap_f, Cp_f)
        pred_cv[te_mask] = pred_te

    m_cv = metrics(y_all, pred_cv)
    r_corr = float(np.corrcoef(x_all, y_all)[0,1])

    return dict(n=len(df), rho_max=rho_max, kappa=kappa, mode=mode_g,
                Ap=Ap_g, Cp=Cp_g, r_corr=r_corr,
                m_insample=m_ins, m_loso=m_cv, n_stations=len(real_stations),
                n_stations_fit=n_stations_fit)

# ─────────────────────────────────────────────────────────────────
# RUN ALL CANDIDATES
# ─────────────────────────────────────────────────────────────────

results = {}

print("\n" + "="*70)
print("CANDIDATE 1: L8 ACOLITE alone")
print("="*70)
r = run_loso(l8_acolite, 'SR_B4', 'lat', 'lon')
results['L8_acolite'] = r
print(f"n={r['n']} kappa={r['kappa']:.3f} mode={r['mode']} r(B4,TSS)={r['r_corr']:.3f}")
print(f"In-sample: R²={r['m_insample']['r2']:.3f} RMSE={r['m_insample']['rmse']:.1f}")
print(f"LOSO-CV:   R²={r['m_loso']['r2']:.3f} RMSE={r['m_loso']['rmse']:.1f} n={r['m_loso']['n']}")

print("\n" + "="*70)
print("CANDIDATE 2: L8 raw C2 L2 (aerosol-filtered)")
print("="*70)
r = run_loso(l8_raw, 'SR_B4', 'lat', 'lon')
results['L8_raw'] = r
print(f"n={r['n']} kappa={r['kappa']:.3f} mode={r['mode']} r(B4,TSS)={r['r_corr']:.3f}")
print(f"In-sample: R²={r['m_insample']['r2']:.3f} RMSE={r['m_insample']['rmse']:.1f}")
print(f"LOSO-CV:   R²={r['m_loso']['r2']:.3f} RMSE={r['m_loso']['rmse']:.1f} n={r['m_loso']['n']}")

print("\n" + "="*70)
print("CANDIDATE 3: S2 ACOLITE original (uncorrected)")
print("="*70)
r = run_loso(s2_orig, 'SR_B4', 'latitud', 'longitud')
results['S2_orig'] = r
print(f"n={r['n']} kappa={r['kappa']:.3f} mode={r['mode']} r(B4,TSS)={r['r_corr']:.3f}")
print(f"In-sample: R²={r['m_insample']['r2']:.3f} RMSE={r['m_insample']['rmse']:.1f}")
print(f"LOSO-CV:   R²={r['m_loso']['r2']:.3f} RMSE={r['m_loso']['rmse']:.1f} n={r['m_loso']['n']}")

print("\n" + "="*70)
print("CANDIDATE 4: S2 ACOLITE corrected (user fix)")
print("="*70)
r = run_loso(s2_corr, 'SR_B4', 'latitud', 'longitud')
results['S2_corr'] = r
print(f"n={r['n']} kappa={r['kappa']:.3f} mode={r['mode']} r(B4,TSS)={r['r_corr']:.3f}")
print(f"In-sample: R²={r['m_insample']['r2']:.3f} RMSE={r['m_insample']['rmse']:.1f}")
print(f"LOSO-CV:   R²={r['m_loso']['r2']:.3f} RMSE={r['m_loso']['rmse']:.1f} n={r['m_loso']['n']}")

import pickle
with open('candidates_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'l8_acolite': l8_acolite, 'l8_raw': l8_raw,
                 's2_orig': s2_orig, 's2_corr': s2_corr}, f)
print("\nSaved intermediate results.")
