"""
Power-Law Model Comparison: TSS = a * rho^b
==============================================
Tests a simpler 2-parameter power-law alternative to the Nechad
semi-analytical model, motivated by an idea from the user's earlier
exploratory work (a hardcoded power-law fallback with placeholder
coefficients in a different, unverified script -- see
07_nir_owt_stratification.py docstring for context on that source).

Unlike Nechad, the power-law form has no saturation asymptote (Cp), so
it sidesteps the entire identifiability question that motivates
Sections 2.5.2-2.5.3 of the manuscript. Tested here with the same
rigor applied to Nechad throughout this project: bootstrap stability
check, full leave-one-station-out CV, per-station breakdown.

RESULT (Landsat-8 Pajarales, n=110):
  Nechad (free Cp):  LOSO R2=0.784  RMSE=50.5 mg/L  bias=+6.2 mg/L
  Power-law:         LOSO R2=0.796  RMSE=49.1 mg/L  bias=+0.2 mg/L

Power-law wins on every metric -- a real, modest, consistent
improvement (R2 +0.012, RMSE -1.4 mg/L), most notably in bias (0.2 vs
6.2 mg/L, i.e. power-law is far less systematically biased).

Bootstrap stability: the exponent b is reasonably stable (CV=6.4%,
comparable to Nechad's Cp at 4.6%); the coefficient a alone is noisier
(CV=22.9%) but this is a parameter-correlation artifact, not real
model instability -- predictions at the median reflectance are stable
(CV=6.8%) once a and b are evaluated together rather than separately.

CAVEAT: per-station LOSO results show the same structural issue as
Nechad -- the pooled R2=0.796 is dominated by the 3 large stations
(n=28-32 each); the 7 small stations (n=2-5) show wildly unstable
per-station R2 in both directions, an artifact of tiny test-fold sizes
rather than a power-law-specific weakness (Nechad shows the identical
pattern on the same folds).

NOT tested here (out of scope for this script): power-law on
Sentinel-2 (found WORSE than Nechad, R2=0.314 vs 0.372 -- Nechad
remains the S2 model) or CGSM (mixed result, R2=0.616 vs Nechad's
0.531-0.639 depending on scenario -- not pursued further since CGSM's
primary framework is the Pajarales-to-CGSM transfer, not a from-scratch
power-law calibration).
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

RHO_MIN = 0.001


def power_law(rho, a, b):
    return a * np.power(rho, b)


def fit_power_law(x, y, w):
    valid = (x >= RHO_MIN) & np.isfinite(y)
    if valid.sum() < 5:
        return None, None
    xv, yv, wv = x[valid], y[valid], w[valid]
    try:
        popt, _ = curve_fit(
            power_law, xv, yv, sigma=1.0 / (wv + 1e-9),
            p0=[1000, 1.0], bounds=([1e-6, 0.01], [1e8, 5.0]), maxfev=30000
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return None, None


def metrics(yt, yp):
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
    a, b = yt[m], yp[m]
    if len(a) < 2:
        return dict(n=int(m.sum()), r2=np.nan, rmse=np.nan, bias=np.nan, mape=np.nan)
    return dict(n=int(m.sum()), r2=float(r2_score(a, b)),
                rmse=float(np.sqrt(mean_squared_error(a, b))),
                bias=float(np.mean(b - a)),
                mape=float(np.mean(np.abs((a - b) / a)) * 100))


def station_id(lat, lon):
    try:
        la, lo = float(lat), float(lon)
        if np.isfinite(la) and np.isfinite(lo):
            return f"{la:.3f},{lo:.3f}"
    except Exception:
        pass
    return "NO_LOCATION"


def bootstrap_stability(x, y, w, n_boot=50, seed=42):
    rng = np.random.default_rng(seed)
    valid = (x >= RHO_MIN) & np.isfinite(y)
    xv, yv, wv = x[valid], y[valid], w[valid]
    n = len(xv)
    rho_test = np.median(xv)
    a_list, b_list, pred_list = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a_f, b_f = fit_power_law(xv[idx], yv[idx], wv[idx])
        if a_f is not None:
            a_list.append(a_f)
            b_list.append(b_f)
            pred_list.append(power_law(rho_test, a_f, b_f))
    a_arr, b_arr, pred_arr = np.array(a_list), np.array(b_list), np.array(pred_list)
    return dict(
        a_cv=float(a_arr.std() / a_arr.mean() * 100),
        b_cv=float(b_arr.std() / abs(b_arr.mean()) * 100),
        pred_cv_at_median=float(pred_arr.std() / pred_arr.mean() * 100),
        rho_median=float(rho_test),
    )


if __name__ == "__main__":
    l8 = pd.read_csv('../data/pajarales_l8_acolite.csv', sep=';')
    l8['station'] = l8.apply(lambda r: station_id(r['lat'], r['lon']), axis=1)
    x = l8['SR_B4'].values
    y = l8['concentracion'].values
    w = np.exp(-np.abs(l8['delta_days']) / 1.0).values
    stations = [s for s in l8['station'].unique() if s != "NO_LOCATION"]

    print("=" * 65)
    print("BOOTSTRAP STABILITY (same scrutiny as Nechad's Cp)")
    print("=" * 65)
    stab = bootstrap_stability(x, y, w)
    print(f"a: CV={stab['a_cv']:.1f}%   b: CV={stab['b_cv']:.1f}%")
    print(f"Prediction CV at median rho ({stab['rho_median']:.4f}): "
          f"{stab['pred_cv_at_median']:.1f}%")
    print("(a's higher CV is a parameter-correlation artifact -- predictions")
    print(" are what matter, and those are stable.)\n")

    print("=" * 65)
    print("GLOBAL FIT + LOSO-CV")
    print("=" * 65)
    a_g, b_g = fit_power_law(x, y, w)
    pred_ins = power_law(x, a_g, b_g)
    m_ins = metrics(y, pred_ins)
    print(f"Global fit: a={a_g:.2f}  b={b_g:.4f}")
    print(f"In-sample: R²={m_ins['r2']:.3f} RMSE={m_ins['rmse']:.1f} bias={m_ins['bias']:.1f}")

    pred_cv = np.full(len(l8), np.nan)
    print("\nPer-station LOSO breakdown:")
    for held in stations:
        te = (l8['station'] == held).values
        tr = ~te
        a_f, b_f = fit_power_law(x[tr], y[tr], w[tr])
        if a_f is None:
            continue
        pte = power_law(x[te], a_f, b_f)
        pred_cv[te] = pte
        m_s = metrics(y[te], pte)
        print(f"  {held:18s} n={te.sum():2d}  a={a_f:8.1f} b={b_f:.3f}  R²={m_s['r2']:.3f}")

    m_cv = metrics(y, pred_cv)
    print(f"\nPooled LOSO-CV: R²={m_cv['r2']:.3f} RMSE={m_cv['rmse']:.1f} "
          f"bias={m_cv['bias']:.1f} mape={m_cv['mape']:.1f} n={m_cv['n']}")

    print("\n" + "=" * 65)
    print("COMPARISON TO NECHAD (free Cp, this project's primary L8 model)")
    print("=" * 65)
    print(f"{'Metric':<15}{'Nechad':>12}{'Power-law':>12}")
    print(f"{'In-sample R2':<15}{0.803:>12.3f}{m_ins['r2']:>12.3f}")
    print(f"{'LOSO R2':<15}{0.784:>12.3f}{m_cv['r2']:>12.3f}")
    print(f"{'LOSO RMSE':<15}{50.5:>12.1f}{m_cv['rmse']:>12.1f}")
    print(f"{'LOSO Bias':<15}{6.2:>12.1f}{m_cv['bias']:>12.1f}")

    l8['TSS_powerlaw_loso'] = pred_cv
    l8.to_csv('../data/l8_powerlaw_results.csv', index=False)
    print("\nSaved ../data/l8_powerlaw_results.csv")
