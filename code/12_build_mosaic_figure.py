"""
Annual TSS Mosaic Figure Assembly
===================================
Takes the annual GeoTIFF outputs from 11_annual_tss_mosaic.py and builds
the multi-panel annual median TSS figure for the manuscript
(TSS_promedio_anual_mosaico_CGSM.png).

Run AFTER 11_annual_tss_mosaic.py has completed successfully.

Usage:
    python 12_build_mosaic_figure.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─── EDIT THESE ──────────────────────────────────────────────────────────────

# Directory containing annual_TSS_nechad_YYYY.tif files
TIFF_DIR = r"E:\output_mosaics"

# Which years to include in the figure (leave empty [] to auto-detect)
YEARS = []  # e.g. [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]

# Which model to plot: 'nechad' for our calibrated model, 'acolite' for ACOLITE SPM
MODEL = 'nechad'

# Colombian water quality reference threshold (Resolución 883 de 2018)
ALERT_THRESHOLD = 150  # mg/L

# TSS color scale limits (mg/L)
TSS_VMIN = 0
TSS_VMAX = 200

# Output path for the manuscript figure
OUTPUT_FIG = r"E:\output_mosaics\TSS_promedio_anual_mosaico_CGSM.png"
# ─────────────────────────────────────────────────────────────────────────────

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("rasterio not found: pip install rasterio")


def read_tiff(path):
    """Read a GeoTIFF as (array, transform, crs)."""
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            bounds = src.bounds
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data <= 0] = np.nan
        return data, transform, bounds
    else:
        arr = np.load(path.replace('.tif', '.npy'))
        return arr, None, None


def find_year_files(tiff_dir, model, years):
    """Find annual TSS tiff files."""
    pattern = os.path.join(tiff_dir, f"annual_TSS_{model}_*.tif")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try .npy fallback
        files = sorted(glob.glob(pattern.replace('.tif', '.npy')))
    year_files = {}
    for f in files:
        try:
            yr = int(os.path.basename(f).split('_')[-1].split('.')[0])
            if not years or yr in years:
                year_files[yr] = f
        except ValueError:
            pass
    return year_files


def build_figure(year_files, output_path, model_label, alert_threshold, vmin, vmax):
    """Build the multi-panel annual TSS mosaic figure."""
    years = sorted(year_files.keys())
    n = len(years)
    if n == 0:
        print("No annual TSS files found. Run 11_annual_tss_mosaic.py first.")
        return

    # Layout: 2 rows x ceil(n/2) cols, with space for shared colorbar
    ncols = min(5, n)  # max 5 per row
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.5))
    axes = np.array(axes).flatten()

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # TSS colorbar reference threshold
    threshold_color = '#800026'

    print(f"Building figure: {n} years, {nrows}x{ncols} panels")

    for i, yr in enumerate(years):
        ax = axes[i]
        data, transform, bounds = read_tiff(year_files[yr])

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        ax.set_title(str(yr), fontsize=9, fontweight='bold', pad=3)
        ax.set_xticks([]); ax.set_yticks([])

        # Stats overlay
        valid = data[np.isfinite(data)]
        if len(valid) > 0:
            median_val = np.nanmedian(valid)
            n_scenes_file = year_files[yr].replace(
                f'TSS_{MODEL}', 'scene_count').replace(
                'scene_count_', 'scene_count_')
            ax.text(0.04, 0.97, f"Median: {median_val:.0f} mg/L",
                    transform=ax.transAxes, fontsize=6.5, va='top',
                    color='black', bbox=dict(fc='white', alpha=0.65, pad=1.5, ec='none'))

            # Flag pixels above alert threshold
            n_alert = np.sum(valid > alert_threshold)
            if n_alert > 0:
                ax.text(0.04, 0.05, f">{alert_threshold}: {n_alert/valid.size*100:.0f}%",
                        transform=ax.transAxes, fontsize=6.5, va='bottom',
                        color=threshold_color,
                        bbox=dict(fc='white', alpha=0.65, pad=1.5, ec='none'))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("TSS (mg L$^{-1}$)", fontsize=10)
    cbar.ax.axhline(alert_threshold, color=threshold_color, lw=1.5, ls='--')
    cbar.ax.text(1.3, alert_threshold / vmax, f"{alert_threshold} mg/L\n(Res.883)",
                 transform=cbar.ax.transAxes, fontsize=7, va='center', color=threshold_color)

    fig.suptitle(f"Annual Median Satellite-Derived TSS (mg L$^{{-1}}$)\n"
                 f"CGSM + Pajarales Complex, Colombia  [{model_label}]",
                 fontsize=12, fontweight='bold', y=0.98)

    plt.subplots_adjust(left=0.02, right=0.90, top=0.93, bottom=0.02,
                        hspace=0.15, wspace=0.08)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {output_path}")
    print(f"  Years: {years}")
    print(f"  Model: {model_label}")


def compare_nechad_vs_acolite(tiff_dir, year):
    """
    For a single year, produce a side-by-side comparison of our Nechad
    model vs ACOLITE's built-in SPM. Useful for validation.
    """
    f_nechad  = os.path.join(tiff_dir, f"annual_TSS_nechad_{year}.tif")
    f_acolite = os.path.join(tiff_dir, f"annual_TSS_acolite_{year}.tif")

    if not os.path.exists(f_nechad) or not os.path.exists(f_acolite):
        print(f"Missing files for year {year} comparison")
        return

    nechad_arr, _, _ = read_tiff(f_nechad)
    acolite_arr, _, _ = read_tiff(f_acolite)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    norm = mcolors.Normalize(vmin=0, vmax=200)
    cmap = plt.cm.YlOrRd

    axes[0].imshow(nechad_arr, cmap=cmap, norm=norm, aspect='auto')
    axes[0].set_title(f"Our Nechad Model\n(Ap=533.2, Cp=0.55)", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(acolite_arr, cmap=cmap, norm=norm, aspect='auto')
    axes[1].set_title(f"ACOLITE Built-in SPM\n(Nechad 2010 default)", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Difference map
    diff = nechad_arr - acolite_arr
    lim = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95)
    im3 = axes[2].imshow(diff, cmap='RdBu_r',
                          norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                          aspect='auto')
    axes[2].set_title("Difference\n(Nechad − ACOLITE)", fontsize=10)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.colorbar(im3, ax=axes[2], shrink=0.8, label="Δ TSS (mg/L)")

    # Stats
    valid_mask = np.isfinite(nechad_arr) & np.isfinite(acolite_arr)
    if valid_mask.sum() > 0:
        from scipy.stats import pearsonr
        r, p = pearsonr(nechad_arr[valid_mask], acolite_arr[valid_mask])
        bias = np.mean(nechad_arr[valid_mask] - acolite_arr[valid_mask])
        rmse = np.sqrt(np.mean((nechad_arr[valid_mask] - acolite_arr[valid_mask])**2))
        fig.text(0.5, 0.02, f"Pixel-level (n={valid_mask.sum():,}): "
                 f"r={r:.3f}, bias={bias:.1f} mg/L, RMSE={rmse:.1f} mg/L",
                 ha='center', fontsize=10, color='gray')

    fig.suptitle(f"Annual Median TSS Comparison — Year {year}", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0,0.05,1,0.95])
    out = os.path.join(tiff_dir, f"comparison_nechad_vs_acolite_{year}.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("ANNUAL TSS MOSAIC FIGURE ASSEMBLY")
    print("=" * 60)

    # Auto-detect years if not specified
    year_files = find_year_files(TIFF_DIR, MODEL, YEARS)
    if not year_files:
        print(f"No files found in: {TIFF_DIR}")
        print(f"Looking for: annual_TSS_{MODEL}_*.tif")
        print("Run 11_annual_tss_mosaic.py first to generate the annual composites.")
    else:
        print(f"Found {len(year_files)} annual files: {sorted(year_files.keys())}")
        model_label = "Calibrated Nechad (Ap=533.2, Cp=0.55)" if MODEL == 'nechad' \
                      else "ACOLITE SPM (Nechad 2010 default)"
        build_figure(year_files, OUTPUT_FIG, model_label,
                     ALERT_THRESHOLD, TSS_VMIN, TSS_VMAX)

        # Generate comparison figures for all available years
        acolite_years = find_year_files(TIFF_DIR, 'acolite', YEARS)
        for yr in sorted(set(year_files.keys()) & set(acolite_years.keys())):
            print(f"\nGenerating Nechad vs ACOLITE comparison for {yr}...")
            compare_nechad_vs_acolite(TIFF_DIR, yr)
