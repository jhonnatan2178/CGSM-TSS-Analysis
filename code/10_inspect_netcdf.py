"""
Step 1: NetCDF File Inspector + Folder Inventory
==================================================
Scans the entire input folder for all L2W NetCDF files, prints a complete
inventory (dates, tiles, satellite), then inspects one representative file
from each tile to confirm variable names before running the mosaic pipeline.

Key things confirmed from your example filenames:
  - All files are in ONE flat folder (not subfolders per tile)
  - Same date does NOT always have both tiles -- single-tile scenes are normal
    e.g. 2015-12-11 has only T18PWS; 2016-02-19 has only T18PWT
  - Pattern: S2A/S2B_MSI_YYYY_MM_DD_HH_MM_SS_T18PW[ST]_CGSM_L2W.nc

Usage:
    python 10_inspect_netcdf.py
"""
import os
import re
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path

# ── EDIT THIS ONE PATH ────────────────────────────────────────────────────────
INPUT_FOLDER = r"E:\images_filtradas\lista_archivos_carpeta1\output_carpeta1"
# ─────────────────────────────────────────────────────────────────────────────

FILENAME_PATTERN = re.compile(
    r"(S2[AB])_MSI_(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}_(T18PW[ST])_CGSM_L2W\.nc$"
)

try:
    import netCDF4 as nc
    USE_NETCDF4 = True
except ImportError:
    try:
        import xarray as xr
        USE_NETCDF4 = False
        print("netCDF4 not found, using xarray")
    except ImportError:
        print("ERROR: pip install netCDF4   OR   pip install xarray")
        exit(1)


# ── 1. Inventory scan ─────────────────────────────────────────────────────────

def scan_folder(folder):
    all_nc = glob.glob(os.path.join(folder, "*L2W.nc"))
    print(f"Found {len(all_nc)} *L2W.nc files in:\n  {folder}\n")

    records = []
    for path in sorted(all_nc):
        name = Path(path).name
        m = FILENAME_PATTERN.match(name)
        if not m:
            print(f"  [SKIP] Unrecognised pattern: {name}")
            continue
        sat, yr, mo, dy, tile = m.groups()
        records.append({
            "path": path, "satellite": sat,
            "date": f"{yr}-{mo}-{dy}", "year": int(yr),
            "month": int(mo), "tile": tile
        })

    if not records:
        print("No files matched the expected naming pattern.")
        return []

    # Summary table
    from collections import Counter
    dates_per_tile = defaultdict(set)
    for r in records:
        dates_per_tile[r['tile']].add(r['date'])

    years = sorted(set(r['year'] for r in records))
    sats  = sorted(set(r['satellite'] for r in records))
    tiles = sorted(set(r['tile'] for r in records))

    print(f"{'─'*60}")
    print(f"INVENTORY SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total files : {len(records)}")
    print(f"  Satellites  : {sats}")
    print(f"  Tiles       : {tiles}")
    print(f"  Years       : {years}")
    print(f"  Unique dates per tile:")
    for tile in tiles:
        print(f"    {tile}: {len(dates_per_tile[tile])} dates")

    # Dates with BOTH tiles vs single-tile only
    if len(tiles) == 2:
        t0, t1 = tiles
        both  = dates_per_tile[t0] & dates_per_tile[t1]
        only0 = dates_per_tile[t0] - dates_per_tile[t1]
        only1 = dates_per_tile[t1] - dates_per_tile[t0]
        print(f"\n  Dates with BOTH {t0}+{t1}: {len(both)}")
        print(f"  Dates with {t0} only      : {len(only0)}")
        print(f"  Dates with {t1} only      : {len(only1)}")
        print(f"  (single-tile scenes are fine -- pipeline handles them)")

    # Per-year breakdown
    print(f"\n  Files per year per tile:")
    year_tile = defaultdict(lambda: defaultdict(int))
    for r in records:
        year_tile[r['year']][r['tile']] += 1
    print(f"  {'Year':<6}" + "".join(f"  {t:<8}" for t in tiles) + "  Total")
    for yr in years:
        row = f"  {yr:<6}"
        total = 0
        for t in tiles:
            n = year_tile[yr][t]
            row += f"  {n:<8}"
            total += n
        print(row + f"  {total}")

    return records


# ── 2. Detailed file inspection ───────────────────────────────────────────────

def inspect_file(path, label):
    print(f"\n{'='*70}")
    print(f"INSPECTING: {label}")
    print(f"  {Path(path).name}")
    print('='*70)

    if USE_NETCDF4:
        with nc.Dataset(path, 'r') as ds:
            # Dimensions
            print(f"\nDimensions:")
            for name, dim in ds.dimensions.items():
                print(f"  {name}: {len(dim)}")

            # All variables
            print(f"\nAll variables ({len(ds.variables)}):")
            for name, var in ds.variables.items():
                fill = getattr(var, '_FillValue', None)
                units = getattr(var, 'units', '')
                lname = getattr(var, 'long_name', '')
                print(f"  {name:32s} shape={str(var.shape):<18} dtype={str(var.dtype):<10}"
                      f"  fill={str(fill):<10}  units={units}  {lname}")

            # Categorised variable check
            print(f"\nKey variable categories:")
            rhow_bands = [v for v in ds.variables if 'rhow' in v.lower()]
            rrs_bands  = [v for v in ds.variables if 'rrs' in v.lower()]
            spm_vars   = [v for v in ds.variables
                          if 'spm' in v.lower() or 'tsm' in v.lower()]
            flag_vars  = [v for v in ds.variables if 'flag' in v.lower()]
            coord_vars = [v for v in ds.variables
                          if v in ('lat','lon','x','y','longitude','latitude')]
            print(f"  rhow bands : {rhow_bands}")
            print(f"  rrs  bands : {rrs_bands}")
            print(f"  SPM / TSM  : {spm_vars}")
            print(f"  Flags      : {flag_vars}")
            print(f"  Coords     : {coord_vars}")

            # Red band check (what our Nechad model uses)
            print(f"\nRed-band (rhow_665) check:")
            target = None
            for cand in ['rhow_665','rhow_664','rhow_660','rhow_B04','Rrs_665','rrs_665']:
                if cand in ds.variables:
                    target = cand
                    break
            if target:
                arr = np.array(ds.variables[target][:], dtype=np.float32)
                fill = getattr(ds.variables[target], '_FillValue', None)
                if fill is not None:
                    arr[arr == fill] = np.nan
                valid = arr[np.isfinite(arr) & (arr > 0)]
                print(f"  Variable   : {target}  ✓")
                print(f"  Shape      : {arr.shape}")
                print(f"  Valid px   : {len(valid):,} / {arr.size:,} "
                      f"({len(valid)/arr.size*100:.1f}%)")
                if len(valid) > 0:
                    print(f"  Range      : {valid.min():.4f} – {valid.max():.4f}")
                    print(f"  Median     : {np.median(valid):.4f}")
            else:
                print(f"  *** NO rhow_665 equivalent found! ***")
                print(f"  Available rhow bands: {rhow_bands}")
                print(f"  -> Update RHOW_VAR in 11_annual_tss_mosaic.py")

            # SPM check
            print(f"\nACOLITE SPM check:")
            if spm_vars:
                for sv in spm_vars[:3]:
                    arr = np.array(ds.variables[sv][:], dtype=np.float32)
                    fill = getattr(ds.variables[sv], '_FillValue', None)
                    if fill is not None: arr[arr==fill] = np.nan
                    valid = arr[np.isfinite(arr) & (arr > 0)]
                    print(f"  {sv}: shape={ds.variables[sv].shape}  "
                          f"valid={len(valid):,}  "
                          f"range=[{valid.min():.1f},{valid.max():.1f}]"
                          if len(valid) > 0 else
                          f"  {sv}: shape={ds.variables[sv].shape}  (no valid pixels)")
            else:
                print(f"  No SPM/TSM variable found -- ACOLITE didn't compute it")
                print(f"  -> Only our calibrated Nechad model will produce TSS")

            # Spatial extent
            print(f"\nSpatial extent:")
            lat_found = False
            for lv in ['lat','latitude']:
                if lv in ds.variables:
                    lat = np.array(ds.variables[lv][:])
                    lnv = 'lon' if 'lon' in ds.variables else 'longitude'
                    lon = np.array(ds.variables[lnv][:])
                    print(f"  Lat : {np.nanmin(lat):.4f} to {np.nanmax(lat):.4f}")
                    print(f"  Lon : {np.nanmin(lon):.4f} to {np.nanmax(lon):.4f}")
                    if lat.ndim == 2:
                        print(f"  Grid: {lat.shape[0]} rows × {lat.shape[1]} cols (2D lat/lon)")
                    else:
                        print(f"  Grid: {lat.shape} × {lon.shape} (1D lat/lon)")
                    lat_found = True
                    break
            if not lat_found:
                print(f"  No lat/lon found. May use x/y projected coords.")
                for v in ['x','y']:
                    if v in ds.variables:
                        arr = ds.variables[v][:]
                        print(f"  {v}: {arr.min():.1f} to {arr.max():.1f} (n={len(arr)})")

    else:
        ds = xr.open_dataset(path)
        print(f"\nDimensions : {dict(ds.dims)}")
        print(f"Variables  : {list(ds.data_vars)}")
        print(f"Coordinates: {list(ds.coords)}")
        ds.close()


# ── Main ─────────────────────────────────────────────────────────────────────

records = scan_folder(INPUT_FOLDER)

if records:
    # Inspect one representative file from each tile
    by_tile = defaultdict(list)
    for r in records:
        by_tile[r['tile']].append(r)

    print(f"\n\n{'─'*70}")
    print("DETAILED INSPECTION (one file per tile)")
    print('─'*70)

    for tile, tile_records in sorted(by_tile.items()):
        # Pick the file with the most recent date to maximise chance of
        # having full ACOLITE outputs (older files may have fewer variables)
        sample = sorted(tile_records, key=lambda r: r['date'])[-1]
        inspect_file(sample['path'], tile)

print(f"""

{'='*70}
WHAT TO DO NEXT
{'='*70}

1. Copy the output above and note:
   a) What is the red-band variable called?
      (rhow_665, rhow_664, etc — needed for RHOW_VAR in 11_annual_tss_mosaic.py)
   b) Is there an SPM variable?
      (if yes, its exact name goes in SPM_VAR)
   c) Are there flag variables? What are they called?
   d) Are lat/lon 1D or 2D arrays?

2. Edit 11_annual_tss_mosaic.py:
   - Set INPUT_FOLDER = r"{INPUT_FOLDER}"
   - Update RHOW_VAR, SPM_VAR, FLAG_VAR based on (a-c) above

3. Run: python 11_annual_tss_mosaic.py

4. When done, run: python 12_build_mosaic_figure.py
{'='*70}
""")
