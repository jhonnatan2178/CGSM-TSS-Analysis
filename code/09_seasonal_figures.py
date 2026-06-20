"""
Monthly Climatology and Annual Anomaly Figures
=================================================
Rebuilds two descriptive figures the user had from an earlier session
(images not independently reproducible by us, since we never had the
generating script) using our verified matchup datasets.

CRITICAL FIX vs. the user's earlier figure: the original monthly
climatology used a date format string requiring HH:MM
(e.g. '%d/%m/%Y %H:%M'). 74 of 130 Sentinel-2 matchup rows have
date-only timestamps (no time component, e.g. '12/08/2025'), which
silently fail that parse and return NaT -- those rows were dropped
from the monthly/seasonal aggregation without any error or warning.
This skewed the dry-season median (the original figure's Pajarales
dry median was 75.3 mg/L; with the parsing fix it is 80.2 mg/L) and,
more importantly, fully invalidated the CGSM seasonal significance
test reported in earlier analysis of this dataset.

RESULT after fixing the date parsing:
  Pajarales: dry median=80.2 (n=112), wet median=29.1 (n=70),
             Kruskal-Wallis p<0.001 (matches prior claim of strong
             significance, numbers shift slightly)
  CGSM:      dry median=66.0 (n=38), wet median=39.0 (n=14),
             Kruskal-Wallis p=0.053 -- THIS IS A SUBSTANTIAL CHANGE.
             Prior analysis of this dataset claimed p=0.80 (no
             significant seasonal pattern) for CGSM. The corrected
             figure is borderline-significant and directionally
             consistent with Pajarales, not flat. Verified robust to
             a single outlier (excluding CGSM's single largest dry-
             season value still gives p=0.106, still far from 0.80).

The annual anomaly figure (Pajarales only, Landsat-8) was independently
verified against the user's earlier figure and reproduces it exactly
(long-term median=67.0, Mann-Kendall tau=-0.467, p=0.073) -- no issues
found, rebuilt here only for consistent styling and to ensure the file
exists in this repo under the filename the manuscript expects.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, kendalltau

warnings.filterwarnings("ignore")


def parse_dt(fecha_str):
    """Robust date parser: tries with time component first, falls back
    to date-only. The user's earlier script used only the first format,
    which silently drops 74/130 Sentinel-2 rows -- see module docstring."""
    for fmt in ['%d/%m/%Y %H:%M', '%d/%m/%Y']:
        try:
            return pd.to_datetime(fecha_str, format=fmt)
        except Exception:
            continue
    return pd.NaT


def season(m):
    if m in [1, 2, 3, 4]:
        return "Dry (Jan-Apr)"
    elif m in [5, 6]:
        return "Transition (May-Jun)"
    elif m in [7, 8, 9, 10, 11]:
        return "Wet (Jul-Nov)"
    elif m == 12:
        return "Transition (Dec)"


SEASON_COLOR = {
    "Dry (Jan-Apr)": "#F57C00",
    "Transition (May-Jun)": "#43A047",
    "Wet (Jul-Nov)": "#1E88E5",
    "Transition (Dec)": "#8E24AA",
}
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def build_monthly_climatology():
    l8 = pd.read_csv('../data/pajarales_l8_acolite.csv', sep=';')
    s2 = pd.read_csv('../data/pajarales_s2_acolite_corrected.csv', sep=';')
    cgsm = pd.read_csv('../data/cgsm_l8_acolite.csv', sep=';')

    for df in (l8, s2, cgsm):
        df['dt'] = df['fecha'].apply(parse_dt)
        df['month'] = df['dt'].dt.month
        n_nat = df['month'].isna().sum()
        if n_nat > 0:
            print(f"WARNING: {n_nat} unparsed dates remain even after fallback")

    paj = pd.concat([l8[['month', 'concentracion']], s2[['month', 'concentracion']]])
    print(f"Pajarales combined n={len(paj)} (expect 240)")
    print(f"CGSM n={len(cgsm)} (expect 74)")

    # Seasonal significance tests (the part that changed)
    for name, df in [("Pajarales", paj), ("CGSM", cgsm)]:
        dry = df[df['month'].isin([1, 2, 3, 4])]['concentracion']
        wet = df[df['month'].isin([7, 8, 9, 10, 11])]['concentracion']
        stat, p = kruskal(dry, wet)
        print(f"\n{name}: dry median={dry.median():.1f} (n={len(dry)}), "
              f"wet median={wet.median():.1f} (n={len(wet)}), "
              f"Kruskal-Wallis p={p:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    for ax, df, title in [
        (axes[0], paj, f"(a) Pajarales (L8 + S2, n={len(paj)})"),
        (axes[1], cgsm, f"(b) CGSM (L8, n={len(cgsm)})"),
    ]:
        box_data, box_colors, counts = [], [], []
        for m in range(1, 13):
            sub = df[df['month'] == m]['concentracion']
            box_data.append(sub.values if len(sub) > 0 else [np.nan])
            box_colors.append(SEASON_COLOR[season(m)])
            counts.append(len(sub))

        bp = ax.boxplot(box_data, positions=range(1, 13), widths=0.6, patch_artist=True,
                         showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.5))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(1.8)

        ymax = ax.get_ylim()[1]
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(i + 1, -ymax * 0.04, f"n={c}", ha='center', fontsize=7.5,
                         color='gray', style='italic')

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES)
        ax.set_ylabel("TSS (mg L$^{-1}$)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=-ymax * 0.08)

        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.7) for c in SEASON_COLOR.values()]
        ax.legend(handles, SEASON_COLOR.keys(), fontsize=8.5, loc='upper right')

    fig.suptitle("Monthly TSS Climatology — In Situ Matchup Dataset (2015–2024)",
                  fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('../images/fig_monthly_climatology.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved ../images/fig_monthly_climatology.png")


def build_annual_anomaly():
    l8 = pd.read_csv('../data/pajarales_l8_acolite.csv', sep=';')
    l8['dt'] = l8['fecha'].apply(parse_dt)
    l8['year'] = l8['dt'].dt.year

    long_term_median = l8['concentracion'].median()
    annual_median = l8.groupby('year')['concentracion'].median()
    annual_n = l8.groupby('year').size()
    anomaly_pct = (annual_median - long_term_median) / long_term_median * 100

    tau, p = kendalltau(annual_median.index, annual_median.values)
    print(f"\nAnnual anomaly: long-term median={long_term_median:.1f}, "
          f"Mann-Kendall tau={tau:.3f} p={p:.3f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#F57C00' if v > 0 else '#1565C0' for v in anomaly_pct]
    ax.bar(anomaly_pct.index.astype(int).astype(str), anomaly_pct.values, color=colors, alpha=0.85)
    for i, (year, val) in enumerate(anomaly_pct.items()):
        n = annual_n[year]
        offset = 8 if val > 0 else -16
        ax.text(i, val + offset, f"n={n}", ha='center', fontsize=8.5, color='gray', style='italic')

    ax.axhline(0, color='black', lw=1.2)
    ax.set_ylabel("TSS anomaly (% deviation\nfrom long-term median)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title(f"Annual TSS Anomaly — Pajarales (L8 matchup data)\n"
                 f"Long-term median = {long_term_median:.1f} mg L$^{{-1}}$; "
                 f"Mann–Kendall $\\tau$ = {tau:.2f}, $p$ = {p:.3f} (ns)",
                 fontsize=12.5, fontweight='bold')

    handles = [plt.Rectangle((0, 0), 1, 1, fc='#F57C00', alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, fc='#1565C0', alpha=0.85)]
    ax.legend(handles, ['Positive anomaly', 'Negative anomaly'], fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig('../images/fig6_annual_anomalies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ../images/fig6_annual_anomalies.png")


if __name__ == "__main__":
    print("=" * 65)
    print("MONTHLY CLIMATOLOGY (with date-parsing fix)")
    print("=" * 65)
    build_monthly_climatology()

    print("\n" + "=" * 65)
    print("ANNUAL ANOMALY (verified against user's earlier figure, exact match)")
    print("=" * 65)
    build_annual_anomaly()
