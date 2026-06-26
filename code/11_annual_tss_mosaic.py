"""
Annual TSS Mosaic Pipeline
============================
Produces annual median TSS maps for the CGSM + Pajarales study area by:

1. Scanning all S2A/S2B L2W NetCDF files across both tiles (T18PWS, T18PWT)
2. Grouping by date (scenes acquired on the same day are the same overpass)
3. For each scene date: reading rhow_665 from both tiles, mosaicking them
   into a single grid, applying the calibrated Nechad model to get TSS
4. Optionally comparing against ACOLITE's built-in SPM (if present)
5. Computing annual median composites and saving as GeoTIFF

BEFORE RUNNING:
  - Run 10_inspect_netcdf.py first and check that the variable names below
    match what your files actually contain (especially RHOW_VAR, SPM_VAR).
  - Install requirements: pip install netCDF4 numpy rasterio tqdm

OUTPUTS (in OUTPUT_DIR):
  annual_TSS_nechad_YYYY.tif    -- annual median from our calibrated model
  annual_TSS_acolite_YYYY.tif   -- annual median from ACOLITE SPM (if available)
  scene_count_YYYY.tif          -- number of valid scenes per pixel per year
  mosaic_log.csv                 -- per-scene metadata and stats

Author: generated for manuscript "Physically-Constrained Calibration and
Transferability of a Semi-Analytical TSS Retrieval Model in a Tropical
Coastal Lagoon System: CGSM, Colombia"
"""
import os
import re
import glob
import logging
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION -- EDIT THESE
# ─────────────────────────────────────────────────────────────────────────────

# Root folder -- ALL NetCDF files are in this single flat folder
# (both T18PWS and T18PWT tiles together, no subfolders needed)
INPUT_FOLDER = r"E:\images_filtradas\lista_archivos_carpeta1\output_carpeta1"

# Output directory for annual GeoTIFFs
OUTPUT_DIR = r"E:\output_mosaics"

# Tile names to include (both needed for full CGSM coverage)
TILES = ["T18PWS", "T18PWT"]

# Variable name for red-band water-leaving reflectance (~665nm)
# Common ACOLITE names: 'rhow_665', 'rhow_664', 'rhow_660', 'Rrs_665'
# Run 10_inspect_netcdf.py to confirm which one your files use
RHOW_VAR = "rhow_665"
RHOW_VAR_FALLBACKS = ["rhow_664", "rhow_660", "rhow_B04", "Rrs_665", "rrs_665"]

# ACOLITE SPM variable name (may or may not be present)
# Common names: 'SPM_Nechad2010', 'SPM', 'tsm', 'spm_nechad2016'
SPM_VAR = "SPM_Nechad2010"
SPM_VAR_FALLBACKS = ["SPM", "spm", "tsm", "SPM_nechad", "SPM_Nechad2016"]

# Flag variable (to mask clouds/land -- set None if not present)
FLAG_VAR = "l2_flags"
FLAG_INVALID_BITS = [1, 2]  # bit 1=cloud, bit 2=land (adjust after inspecting)
FLAG_USE = True  # set False to skip flag masking if flags aren't reliable

# Calibrated Nechad model parameters (from this project's bootstrap-validated fit)
# -- Sentinel-2 uses fixed Cp since boundary-converged (see manuscript §3.2)
NECHAD_S2 = {"Ap": 533.2, "Cp": 0.55}

# TSS thresholds for output range clipping (physical limits, mg/L)
TSS_MIN = 0.0
TSS_MAX = 1000.0

# Minimum valid reflectance to treat a pixel as water (not land/cloud)
RHOW_MIN = 0.001
RHOW_MAX = 0.30    # above this, likely cloud or adjacency artifact

# Output coordinate reference system (WGS84 geographic -- ACOLITE default)
# If your files are projected (UTM), adjust accordingly
OUTPUT_CRS = "EPSG:4326"

# Target output resolution in degrees (approx 10m S2 -> ~0.0001 deg)
# If mosaicking to a common grid, this sets the grid spacing
TARGET_RES = 0.0001  # degrees per pixel (~11m at equator)

# ─────────────────────────────────────────────────────────────────────────────

try:
    import netCDF4 as nc4
    HAS_NC4 = True
except ImportError:
    HAS_NC4 = False
    log.warning("netCDF4 not found; trying xarray")

try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    from rasterio.merge import merge as rio_merge
    from rasterio import MemoryFile
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    log.error("rasterio not found: pip install rasterio")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kw: x  # no-op fallback


# ─── Helpers ─────────────────────────────────────────────────────────────────

def nechad_tss(rhow, Ap, Cp):
    """Apply Nechad semi-analytical model: TSS = Ap * rhow / (1 - rhow/Cp)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = 1.0 - rhow / Cp
        denom = np.where(np.abs(denom) > 1e-6, denom, np.nan)
        tss = Ap * rhow / denom
    tss = np.where(rhow < RHOW_MIN, np.nan, tss)
    tss = np.where(rhow > RHOW_MAX, np.nan, tss)
    tss = np.where(tss < TSS_MIN, np.nan, tss)
    tss = np.where(tss > TSS_MAX, np.nan, tss)
    return tss.astype(np.float32)


def parse_filename(path):
    """
    Parse ACOLITE L2W filename: S2A_MSI_2015_12_11_15_30_17_T18PWS_CGSM_L2W.nc
    Returns dict with keys: satellite, sensor, date, time, tile, site
    Returns None if filename doesn't match expected pattern.
    """
    name = Path(path).stem
    pattern = r"(S2[AB]|LC08|LC09)_(\w+?)_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(T18PW[ST])_(\w+?)_L2W"
    m = re.match(pattern, name)
    if not m:
        return None
    return {
        "path": str(path),
        "satellite": m.group(1),
        "sensor": m.group(2),
        "date": f"{m.group(3)}-{m.group(4)}-{m.group(5)}",
        "time": f"{m.group(6)}:{m.group(7)}:{m.group(8)}",
        "tile": m.group(9),
        "site": m.group(10),
        "year": int(m.group(3)),
    }


def find_variable(ds_vars, primary, fallbacks):
    """Find a variable by primary name, then try fallbacks."""
    if primary in ds_vars:
        return primary
    for fb in fallbacks:
        if fb in ds_vars:
            log.debug(f"  Using fallback variable '{fb}' (primary '{primary}' not found)")
            return fb
    return None


def read_nc_arrays(path, rhow_var, spm_var):
    """
    Read rhow and SPM arrays from an ACOLITE L2W NetCDF file.
    Returns: rhow (2D array), spm (2D array or None), lat, lon
    """
    if HAS_NC4:
        with nc4.Dataset(path, 'r') as ds:
            # Resolve variable names
            rv = find_variable(ds.variables, rhow_var, RHOW_VAR_FALLBACKS)
            sv = find_variable(ds.variables, spm_var, SPM_VAR_FALLBACKS) if spm_var else None

            if rv is None:
                log.warning(f"  No reflectance band found in {Path(path).name}")
                return None, None, None, None

            rhow = np.array(ds.variables[rv][:], dtype=np.float32)
            # Handle fill values
            fill = getattr(ds.variables[rv], '_FillValue', None)
            if fill is not None:
                rhow[rhow == fill] = np.nan

            spm = None
            if sv:
                spm = np.array(ds.variables[sv][:], dtype=np.float32)
                fill_s = getattr(ds.variables[sv], '_FillValue', None)
                if fill_s is not None:
                    spm[spm == fill_s] = np.nan

            # Apply quality flags
            if FLAG_USE and FLAG_VAR in ds.variables:
                flags = np.array(ds.variables[FLAG_VAR][:], dtype=np.int32)
                invalid = np.zeros(flags.shape, dtype=bool)
                for bit in FLAG_INVALID_BITS:
                    invalid |= ((flags & (1 << bit)) != 0)
                rhow[invalid] = np.nan
                if spm is not None:
                    spm[invalid] = np.nan

            # Get coordinates
            lat = lon = None
            for lv in ['lat', 'latitude']:
                if lv in ds.variables:
                    lat = np.array(ds.variables[lv][:], dtype=np.float64)
                    break
            for lv in ['lon', 'longitude']:
                if lv in ds.variables:
                    lon = np.array(ds.variables[lv][:], dtype=np.float64)
                    break

            return rhow, spm, lat, lon
    else:
        ds = xr.open_dataset(path)
        rv = find_variable(list(ds.data_vars), rhow_var, RHOW_VAR_FALLBACKS)
        sv = find_variable(list(ds.data_vars), spm_var, SPM_VAR_FALLBACKS) if spm_var else None
        if rv is None:
            ds.close(); return None, None, None, None
        rhow = ds[rv].values.astype(np.float32)
        spm = ds[sv].values.astype(np.float32) if sv else None
        lat = ds['lat'].values if 'lat' in ds.coords else None
        lon = ds['lon'].values if 'lon' in ds.coords else None
        ds.close()
        return rhow, spm, lat, lon


def array_to_memfile(data, lat, lon):
    """
    Convert a 2D numpy array + lat/lon grids into a rasterio MemoryFile
    for use with rasterio.merge.
    Assumes lat/lon are 2D arrays (same shape as data).
    """
    if not HAS_RASTERIO:
        return None
    if lat is None or lon is None:
        return None

    # Determine bounds
    lat_f, lon_f = lat.astype(float), lon.astype(float)
    south, north = np.nanmin(lat_f), np.nanmax(lat_f)
    west, east   = np.nanmin(lon_f), np.nanmax(lon_f)
    nrows, ncols = data.shape

    transform = from_bounds(west, south, east, north, ncols, nrows)
    crs = CRS.from_epsg(4326)

    mf = MemoryFile()
    with mf.open(driver='GTiff', height=nrows, width=ncols, count=1,
                  dtype=data.dtype, crs=crs, transform=transform,
                  nodata=np.nan) as ds:
        ds.write(data[np.newaxis, :, :])
    return mf


def merge_tiles(arrays_latlons, output_res=TARGET_RES):
    """
    Merge multiple (data, lat, lon) arrays onto a common grid using
    rasterio.merge (last array wins for overlapping pixels, or use 'first').
    Returns merged 2D array and its extent.
    """
    if not HAS_RASTERIO:
        # Fallback: return the first non-None array
        for data, lat, lon in arrays_latlons:
            if data is not None:
                return data, lat, lon
        return None, None, None

    memfiles = []
    opened = []
    for data, lat, lon in arrays_latlons:
        if data is None or lat is None:
            continue
        mf = array_to_memfile(data, lat, lon)
        if mf is not None:
            memfiles.append(mf)
            opened.append(mf.open())

    if not opened:
        return None, None, None

    try:
        merged, transform = rio_merge(opened, method='first', nodata=np.nan)
        merged_2d = merged[0]
        # Reconstruct lat/lon arrays for output
        height, width = merged_2d.shape
        west = transform.c; north = transform.f
        res = transform.a  # pixel width in degrees
        lons = west + (np.arange(width) + 0.5) * res
        lats = north + (np.arange(height) + 0.5) * transform.e  # transform.e is negative
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        return merged_2d.astype(np.float32), lat_grid, lon_grid
    finally:
        for ds in opened:
            ds.close()
        for mf in memfiles:
            mf.close()


def save_geotiff(array, lat, lon, output_path, nodata=np.nan):
    """Save a 2D array to GeoTIFF using rasterio."""
    if not HAS_RASTERIO:
        # Fallback: save as numpy binary
        np.save(output_path.replace('.tif', '.npy'), array)
        log.warning(f"  rasterio not available; saved as .npy instead of .tif")
        return

    nrows, ncols = array.shape
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    west, east   = float(np.nanmin(lon)), float(np.nanmax(lon))
    transform = from_bounds(west, south, east, north, ncols, nrows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', driver='GTiff',
                        height=nrows, width=ncols, count=1,
                        dtype=rasterio.float32,
                        crs=CRS.from_epsg(4326),
                        transform=transform,
                        nodata=nodata,
                        compress='lzw') as dst:
        dst.write(array[np.newaxis, :, :])
    log.info(f"  Saved: {output_path}")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def scan_files(folder):
    """
    Find all L2W NetCDF files in a single flat folder (no subfolders needed).
    Both T18PWS and T18PWT files are expected in the same directory.
    Single-tile scenes (only one tile for a given date) are handled normally --
    the mosaic step simply returns that one tile's data directly.
    """
    log.info(f"Scanning folder: {folder}")
    # Non-recursive: all files are in one flat folder
    all_nc = glob.glob(os.path.join(folder, "*L2W.nc"))
    log.info(f"Found {len(all_nc)} *L2W.nc files")

    records = []
    for path in sorted(all_nc):
        meta = parse_filename(path)
        if meta and meta['tile'] in TILES:
            records.append(meta)
        elif meta is None:
            log.debug(f"  Skipped (unexpected name pattern): {Path(path).name}")

    df = pd.DataFrame(records)
    if df.empty:
        log.error("No valid L2W files found. Check INPUT_FOLDER and TILES settings.")
        return df

    # Count dates with one vs both tiles
    date_tiles = df.groupby('date')['tile'].apply(set)
    both = (date_tiles.apply(len) == 2).sum()
    single = (date_tiles.apply(len) == 1).sum()

    log.info(f"Valid L2W files: {len(df)}")
    log.info(f"  Satellites: {df['satellite'].unique()}")
    log.info(f"  Tiles:      {df['tile'].unique()}")
    log.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    log.info(f"  Years:      {sorted(df['year'].unique())}")
    log.info(f"  Dates with both tiles: {both}  |  single-tile only: {single}")
    log.info(f"  (single-tile scenes processed normally -- no pairing required)")
    return df


def process_year(year, df_year, output_dir, rhow_var, spm_var):
    """
    Process all scenes for one year: mosaic tiles, compute TSS, accumulate
    for annual median, return stack summary.
    """
    dates = sorted(df_year['date'].unique())
    log.info(f"\n  Year {year}: {len(dates)} scene dates, "
             f"{len(df_year)} total files")

    # Accumulators for annual median computation
    tss_stack   = []  # list of 2D arrays (TSS from Nechad)
    spm_stack   = []  # list of 2D arrays (ACOLITE SPM)
    ref_lat = ref_lon = None  # will be set from first successful mosaic

    scene_logs = []

    for date in (tqdm(dates, desc=f"  {year}") if HAS_TQDM else dates):
        day_df = df_year[df_year['date'] == date]

        # Read each tile for this date
        tile_data = {}
        for _, row in day_df.iterrows():
            tile = row['tile']
            rhow, spm_ac, lat, lon = read_nc_arrays(row['path'], rhow_var, spm_var)
            if rhow is None:
                log.warning(f"  Skipped {Path(row['path']).name} (unreadable)")
                continue
            # Squeeze if 3D (single band, may have leading dim)
            if rhow.ndim == 3:
                rhow = rhow[0]
            if spm_ac is not None and spm_ac.ndim == 3:
                spm_ac = spm_ac[0]

            # Compute TSS from our calibrated Nechad model
            tss = nechad_tss(rhow, **NECHAD_S2)

            tile_data[tile] = {
                'rhow': rhow, 'tss': tss, 'spm_ac': spm_ac, 'lat': lat, 'lon': lon
            }

        if not tile_data:
            log.warning(f"  {date}: no valid tiles read, skipping")
            continue

        # Mosaic tiles for this date
        tss_pairs  = [(d['tss'],    d['lat'], d['lon']) for d in tile_data.values()]
        spm_pairs  = [(d['spm_ac'], d['lat'], d['lon']) for d in tile_data.values()
                       if d['spm_ac'] is not None]

        tss_merged, lat_m, lon_m = merge_tiles(tss_pairs)
        if tss_merged is None:
            log.warning(f"  {date}: tile merge failed, skipping")
            continue

        spm_merged = None
        if spm_pairs:
            spm_merged, _, _ = merge_tiles(spm_pairs)

        # Store reference grid from first successful scene
        if ref_lat is None:
            ref_lat, ref_lon = lat_m, lon_m

        # Reproject merged to reference grid if shapes differ
        if tss_merged.shape != ref_lat.shape:
            # Simple approach: skip if shapes differ significantly
            # For production: use rasterio.reproject for exact alignment
            if abs(tss_merged.size - ref_lat.size) / ref_lat.size > 0.01:
                log.warning(f"  {date}: shape mismatch {tss_merged.shape} vs "
                            f"{ref_lat.shape}, skipping")
                continue

        tss_stack.append(tss_merged)
        if spm_merged is not None:
            spm_stack.append(spm_merged)

        valid_pct = float(np.isfinite(tss_merged).sum() / tss_merged.size * 100)
        tss_valid = tss_merged[np.isfinite(tss_merged)]
        scene_logs.append({
            "date": date, "year": year,
            "tiles": list(tile_data.keys()),
            "shape": str(tss_merged.shape),
            "valid_pct": round(valid_pct, 1),
            "tss_median": round(float(np.nanmedian(tss_valid)), 1) if len(tss_valid) else np.nan,
            "tss_p90":    round(float(np.nanpercentile(tss_valid, 90)), 1) if len(tss_valid) else np.nan,
        })
        log.debug(f"  {date}: valid={valid_pct:.0f}% TSS_med={scene_logs[-1]['tss_median']:.1f}")

    if not tss_stack:
        log.warning(f"  Year {year}: no valid scenes processed")
        return scene_logs

    # Annual median composite
    log.info(f"  Year {year}: computing annual median from {len(tss_stack)} scenes")
    tss_cube = np.stack(tss_stack, axis=0)         # (n_scenes, rows, cols)
    tss_annual = np.nanmedian(tss_cube, axis=0).astype(np.float32)
    count_map  = np.sum(np.isfinite(tss_cube), axis=0).astype(np.int16)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    save_geotiff(tss_annual, ref_lat, ref_lon,
                 os.path.join(output_dir, f"annual_TSS_nechad_{year}.tif"))
    save_geotiff(count_map.astype(np.float32), ref_lat, ref_lon,
                 os.path.join(output_dir, f"scene_count_{year}.tif"))

    if spm_stack:
        spm_cube = np.stack(spm_stack, axis=0)
        spm_annual = np.nanmedian(spm_cube, axis=0).astype(np.float32)
        save_geotiff(spm_annual, ref_lat, ref_lon,
                     os.path.join(output_dir, f"annual_TSS_acolite_{year}.tif"))

    # Summary stats
    v = tss_annual[np.isfinite(tss_annual)]
    log.info(f"  Year {year} done: "
             f"median={np.nanmedian(v):.1f} mg/L, "
             f"scenes={len(tss_stack)}, "
             f"coverage={np.isfinite(tss_annual).sum()/tss_annual.size*100:.0f}%")
    return scene_logs


def main():
    log.info("=" * 70)
    log.info("ANNUAL TSS MOSAIC PIPELINE")
    log.info("=" * 70)
    log.info(f"Input:       {INPUT_FOLDER}")
    log.info(f"Output:      {OUTPUT_DIR}")
    log.info(f"Tiles:       {TILES}")
    log.info(f"Nechad S2:   Ap={NECHAD_S2['Ap']}, Cp={NECHAD_S2['Cp']}")
    log.info(f"rhow band:   {RHOW_VAR}")
    log.info(f"ACOLITE SPM: {SPM_VAR}")

    # Scan all files
    df = scan_files(INPUT_FOLDER)
    if df.empty:
        return

    # Process year by year
    all_logs = []
    years = sorted(df['year'].unique())
    log.info(f"\nProcessing {len(years)} years: {years}")

    for year in years:
        df_yr = df[df['year'] == year]
        scene_logs = process_year(year, df_yr, OUTPUT_DIR, RHOW_VAR, SPM_VAR)
        all_logs.extend(scene_logs)

    # Save scene log
    if all_logs:
        log_df = pd.DataFrame(all_logs)
        log_path = os.path.join(OUTPUT_DIR, "mosaic_log.csv")
        log_df.to_csv(log_path, index=False)
        log.info(f"\nScene log saved: {log_path}")
        log.info(f"Total scenes processed: {len(log_df)}")
        log.info(f"Years completed: {sorted(log_df['year'].unique())}")

    log.info("\nPipeline complete.")
    log.info(f"Output files in: {OUTPUT_DIR}")
    log.info("\nNEXT STEPS:")
    log.info("  1. Open annual_TSS_nechad_YYYY.tif files in QGIS")
    log.info("  2. Compare with annual_TSS_acolite_YYYY.tif (if ACOLITE SPM was present)")
    log.info("  3. Use scene_count_YYYY.tif to mask pixels with fewer than 2-3 valid scenes")
    log.info("  4. The annual_TSS_nechad*.tif files become TSS_promedio_anual_mosaico_CGSM.png")
    log.info("     by assembling a multi-panel figure in QGIS or matplotlib")


if __name__ == "__main__":
    main()
