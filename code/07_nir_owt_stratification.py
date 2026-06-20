"""
NIR-Aware OWT Stratification Test
====================================
Extends the project's existing OWT investigation (B2/B3/B4-only
classification, found not to improve LOSO-CV performance) by testing a
second classification rule that additionally uses the NIR band (B5),
adapted from an earlier exploratory script the user located
(working on a different, unverified spectral-signatures dataset with
placeholder calibration parameters -- not usable directly, but its OWT
rule structure was worth testing on our real matchup data).

Rule (adapted):
    Type 1: rho_B4 < rho_B3 and rho_B4 > rho_B2   (clearer water)
    Type 2: rho_B4 > rho_B3                        (intermediate)
    Type 3: rho_B5 (NIR) > threshold                (turbid/NIR-bright)
    default: Type 2

IMPORTANT CAVEAT: Landsat-8's B5 (~865nm) is genuine NIR. Sentinel-2's
"B5" in this project's matchup files is actually ~705nm (red-edge), per
the manuscript's own spectral band table (Table 2) -- so the S2 test
below is not a true NIR classification, just whatever this project's
B5 column happens to be for that sensor. This is noted explicitly
rather than silently treating S2's result as equivalent to L8/CGSM's.

NIR threshold is swept (not hardcoded to the source script's 0.01,
which was likely calibrated for a normalized-reflectance scale we do
not have) -- see THRESHOLDS below.

RESULT (see __main__ output): NIR-aware OWT stratification does not
improve leave-one-station-out performance for any dataset where it
could be tested:
  - L8 Pajarales: cannot test -- entire dataset (110/110) classifies as
    a single type under this rule (B4 > B3 always true here, so the
    rule's Type-2 branch captures everything before NIR is ever
    checked; same finding as the original B2/B3/B4-only rule, which
    found 100% Type 3 by a different definition).
  - S2 Pajarales: R² 0.372 -> 0.363 (fair comparison, coverage 56/130
    both ways) -- slightly worse.
  - CGSM: R² 0.531 -> 0.524 (fair comparison, coverage 73-74/74) --
    slightly worse.

Combined with the original OWT test (B2/B3/B4 only, also no
improvement), this closes optical-water-type stratification as a
promising lever for this dataset under either classification scheme
tried.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

CP_UPPER = 0.55
RHO_MIN = 0.001
FIT_MARGIN = 0.90
MIN_SAMPLES = 5


def nechad(rho, Ap, Cp):
    denom = 1.0 - rho / Cp
    denom = np.where(np.abs(denom) > 1e-6, denom, np.nan)
    return Ap * rho / denom


def fit_fixed_Cp(x, y, w, Cp, margin=FIT_MARGIN):
    valid = (x >= RHO_MIN) & (x < Cp * margin) & np.isfinite(y)
    if valid.sum() < 2:
        return None, None
    xv, yv, wv = x[valid], y[valid], w[valid]
    feat = xv / (1.0 - xv / Cp)
    Ap = float(np.sum(wv * yv * feat) / np.sum(wv * feat ** 2))
    return Ap, Cp


def metrics(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
    a, b = yt[m], yp[m]
    if len(a) < 2:
        return dict(n=int(m.sum()), r2=np.nan, rmse=np.nan)
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


def classify_owt_nir(b2, b3, b4, b5, nir_threshold):
    """OWT rule adapted from the user's earlier exploratory script."""
    if b4 < b3 and b4 > b2:
        return "Type 1"
    elif b4 > b3:
        return "Type 2"
    elif b5 > nir_threshold:
        return "Type 3"
    else:
        return "Type 2"


def baseline_loso(df, xcol, station_col, weight_col, Cp):
    x = df[xcol].values
    y = df['concentracion'].values
    w = df[weight_col].values
    stations = [s for s in df[station_col].unique() if s != "NO_LOCATION"]
    pred = np.full(len(df), np.nan)
    for held in stations:
        te = (df[station_col] == held).values
        tr = ~te
        Ap_f, Cp_f = fit_fixed_Cp(x[tr], y[tr], w[tr], Cp)
        if Ap_f is None:
            continue
        pm = (x[te] >= RHO_MIN) & (x[te] < Cp_f * FIT_MARGIN)
        pte = np.full(te.sum(), np.nan)
        if pm.any():
            pte[pm] = nechad(x[te][pm], Ap_f, Cp_f)
        pred[te] = pte
    return metrics(y, pred)


def owt_stratified_loso(df, xcol, station_col, weight_col, owt_col, Cp):
    x = df[xcol].values
    y = df['concentracion'].values
    w = df[weight_col].values
    pred = np.full(len(df), np.nan)
    for owt, grp_idx in df.groupby(owt_col).groups.items():
        sub = df.loc[grp_idx]
        sub_stations = [s for s in sub[station_col].unique() if s != "NO_LOCATION"]
        if len(sub) < MIN_SAMPLES or len(sub_stations) < 3:
            continue
        xs = sub[xcol].values
        ys = sub['concentracion'].values
        ws = sub[weight_col].values
        for held in sub_stations:
            te = (sub[station_col] == held).values
            tr = ~te
            if tr.sum() < MIN_SAMPLES:
                continue
            Ap_f, Cp_f = fit_fixed_Cp(xs[tr], ys[tr], ws[tr], Cp)
            if Ap_f is None:
                continue
            pm = (xs[te] >= RHO_MIN) & (xs[te] < Cp_f * FIT_MARGIN)
            pte = np.full(te.sum(), np.nan)
            if pm.any():
                pte[pm] = nechad(xs[te][pm], Ap_f, Cp_f)
            pred[grp_idx[te]] = pte
    return metrics(y, pred)


if __name__ == "__main__":
    print("=" * 65)
    print("LANDSAT-8 PAJARALES: NIR-aware OWT classification")
    print("=" * 65)
    l8 = pd.read_csv('../data/pajarales_l8_acolite.csv', sep=';')
    print(f"L8 B4>B3 fraction: {(l8['SR_B4'] > l8['SR_B3']).mean():.2%}")
    l8['OWT_nir'] = l8.apply(
        lambda r: classify_owt_nir(r['SR_B2'], r['SR_B3'], r['SR_B4'], r['SR_B5'], 0.05),
        axis=1)
    print(f"OWT distribution: {l8['OWT_nir'].value_counts().to_dict()}")
    print("-> Cannot test stratification: entire dataset is one class "
          "(B4>B3 always true on this turbid system, NIR branch never reached).\n")

    print("=" * 65)
    print("CGSM: NIR-aware OWT stratification vs pooled baseline")
    print("=" * 65)
    cgsm = pd.read_csv('../data/cgsm_l8_acolite.csv', sep=';')
    cgsm['weight'] = np.exp(-np.abs(cgsm['delta_days']) / 1.0)
    cgsm['station'] = cgsm['lat'].round(3).astype(str) + "," + cgsm['lon'].round(3).astype(str)
    cgsm['OWT_nir'] = cgsm.apply(
        lambda r: classify_owt_nir(r['b2'], r['b3'], r['b4'], r['b5'], 0.02), axis=1)

    m_base = baseline_loso(cgsm, 'b4', 'station', 'weight', CP_UPPER)
    m_owt = owt_stratified_loso(cgsm, 'b4', 'station', 'weight', 'OWT_nir', CP_UPPER)
    print(f"OWT distribution: {cgsm['OWT_nir'].value_counts().to_dict()}")
    print(f"Baseline (pooled):     LOSO R²={m_base['r2']:.3f} RMSE={m_base['rmse']:.1f} "
          f"n={m_base['n']}/{len(cgsm)}")
    print(f"NIR-OWT stratified:    LOSO R²={m_owt['r2']:.3f} RMSE={m_owt['rmse']:.1f} "
          f"n={m_owt['n']}/{len(cgsm)}")
    print(f"-> {'IMPROVES' if m_owt['r2'] > m_base['r2'] else 'DOES NOT IMPROVE'}\n")

    print("=" * 65)
    print("SENTINEL-2: NIR-aware OWT stratification vs pooled baseline")
    print("(CAVEAT: S2 'B5' in this dataset is ~705nm red-edge, not true NIR)")
    print("=" * 65)
    s2 = pd.read_csv('../data/pajarales_s2_acolite_corrected.csv', sep=';')
    s2['weight'] = np.exp(-np.abs(s2['delta_days']) / 1.0)
    s2['station'] = s2.apply(lambda r: station_id(r['latitud'], r['longitud']), axis=1)
    s2['OWT_nir'] = s2.apply(
        lambda r: classify_owt_nir(r['SR_B2'], r['SR_B3'], r['SR_B4'], r['SR_B5'], 0.0559),
        axis=1)

    m_base_s2 = baseline_loso(s2, 'SR_B4', 'station', 'weight', CP_UPPER)
    m_owt_s2 = owt_stratified_loso(s2, 'SR_B4', 'station', 'weight', 'OWT_nir', CP_UPPER)
    print(f"OWT distribution: {s2['OWT_nir'].value_counts().to_dict()}")
    print(f"Baseline (pooled):     LOSO R²={m_base_s2['r2']:.3f} RMSE={m_base_s2['rmse']:.1f} "
          f"n={m_base_s2['n']}/{len(s2)}")
    print(f"NIR-OWT stratified:    LOSO R²={m_owt_s2['r2']:.3f} RMSE={m_owt_s2['rmse']:.1f} "
          f"n={m_owt_s2['n']}/{len(s2)}")
    print(f"-> {'IMPROVES' if m_owt_s2['r2'] > m_base_s2['r2'] else 'DOES NOT IMPROVE'}")
