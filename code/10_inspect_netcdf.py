"""
Step 1: NetCDF File Inspector
==============================
Run this FIRST on one file from each tile (T18PWS and T18PWT) to understand
what variables are available, the coordinate system, spatial resolution,
and whether ACOLITE has already computed SPM.

Usage:
    python 01_inspect_netcdf.py

Edit the two paths at the top to point to one file from each tile.
"""
import numpy as np

# ── EDIT THESE TWO PATHS ─────────────────────────────────────────────────────
FILE_PWS = r"E:\images_filtradas\lista_archivos_carpeta1\output_carpeta1\S2A_MSI_2015_12_11_15_30_17_T18PWS_CGSM_L2W.nc"
FILE_PWT = r"E:\images_filtradas\lista_archivos_carpeta1\output_carpeta1\S2A_MSI_2015_12_11_15_30_17_T18PWT_CGSM_L2W.nc"
# Replace FILE_PWT with an actual T18PWT file path from your folder
# ─────────────────────────────────────────────────────────────────────────────

try:
    import netCDF4 as nc
    USE_NETCDF4 = True
except ImportError:
    try:
        import xarray as xr
        USE_NETCDF4 = False
        print("Using xarray (netCDF4 not found)")
    except ImportError:
        print("ERROR: install either netCDF4 or xarray:")
        print("  pip install netCDF4   OR   pip install xarray")
        exit(1)


def inspect_file(path, label):
    print(f"\n{'='*70}")
    print(f"FILE: {label}")
    print(f"PATH: {path}")
    print('='*70)

    if USE_NETCDF4:
        with nc.Dataset(path, 'r') as ds:
            print(f"\nGlobal attributes:")
            for attr in ds.ncattrs():
                val = getattr(ds, attr)
                if isinstance(val, str) and len(val) > 80:
                    val = val[:77] + '...'
                print(f"  {attr}: {val}")

            print(f"\nDimensions:")
            for name, dim in ds.dimensions.items():
                print(f"  {name}: {len(dim)}")

            print(f"\nVariables ({len(ds.variables)} total):")
            for name, var in ds.variables.items():
                shape = var.shape
                dtype = var.dtype
                attrs = {a: getattr(var, a) for a in var.ncattrs()
                         if a in ('long_name', 'units', 'valid_min', 'valid_max',
                                  'flag_meanings', '_FillValue')}
                print(f"  {name:30s} shape={str(shape):20s} dtype={dtype}")
                for k, v in attrs.items():
                    vstr = str(v)
                    if len(vstr) > 60: vstr = vstr[:57] + '...'
                    print(f"    {k}: {vstr}")

            # Check specifically for rhow/rrs bands and SPM
            print(f"\nKey variables check:")
            rhow_bands = [v for v in ds.variables if 'rhow' in v.lower()]
            rrs_bands  = [v for v in ds.variables if 'rrs' in v.lower()]
            spm_vars   = [v for v in ds.variables if 'spm' in v.lower() or 'tsm' in v.lower()]
            flag_vars  = [v for v in ds.variables if 'flag' in v.lower()]
            coord_vars = [v for v in ds.variables if v in ('lat','lon','x','y','longitude','latitude')]

            print(f"  rhow bands: {rhow_bands}")
            print(f"  rrs bands:  {rrs_bands}")
            print(f"  SPM/TSM:    {spm_vars}")
            print(f"  Flags:      {flag_vars}")
            print(f"  Coords:     {coord_vars}")

            # Check for rhow_665 specifically (what our model uses)
            target = None
            for candidate in ['rhow_665', 'rhow_664', 'rhow_660', 'rhow_B04']:
                if candidate in ds.variables:
                    target = candidate
                    break
            if target:
                v = ds.variables[target]
                arr = v[:]
                valid = arr[np.isfinite(arr) & (arr > 0)]
                print(f"\n  Red band ({target}): shape={arr.shape}")
                print(f"    valid pixels: {len(valid)}/{arr.size} ({len(valid)/arr.size*100:.1f}%)")
                print(f"    range: {valid.min():.4f} - {valid.max():.4f}")
                print(f"    median: {np.median(valid):.4f}")
            else:
                print(f"\n  WARNING: no rhow_665 (or similar) found!")
                print(f"  Available bands: {rhow_bands[:10]}")

            # Check SPM values if present
            if spm_vars:
                for sv in spm_vars[:2]:
                    arr = ds.variables[sv][:]
                    valid = arr[np.isfinite(arr) & (arr > 0)]
                    print(f"\n  {sv}: shape={arr.shape}")
                    if len(valid) > 0:
                        print(f"    valid pixels: {len(valid)}/{arr.size}")
                        print(f"    range: {valid.min():.2f} - {valid.max():.2f}")
                        print(f"    median: {np.median(valid):.2f}")

            # Spatial extent
            for coord in ['lat', 'latitude']:
                if coord in ds.variables:
                    lat = ds.variables[coord][:]
                    lon_var = 'lon' if 'lon' in ds.variables else 'longitude'
                    lon = ds.variables[lon_var][:]
                    print(f"\n  Spatial extent:")
                    print(f"    Lat: {np.nanmin(lat):.4f} to {np.nanmax(lat):.4f}")
                    print(f"    Lon: {np.nanmin(lon):.4f} to {np.nanmax(lon):.4f}")
                    break

    else:
        ds = xr.open_dataset(path)
        print(f"\nDimensions: {dict(ds.dims)}")
        print(f"\nVariables: {list(ds.data_vars)}")
        print(f"\nCoordinates: {list(ds.coords)}")
        print(f"\nAttributes: {dict(list(ds.attrs.items())[:10])}")
        ds.close()


inspect_file(FILE_PWS, "T18PWS tile")

import os
if os.path.exists(FILE_PWT):
    inspect_file(FILE_PWT, "T18PWT tile")
else:
    print(f"\nT18PWT file not found at: {FILE_PWT}")
    print("Edit FILE_PWT at the top of this script to point to a real T18PWT file.")

print("\n\nINSTRUCTIONS FOR NEXT STEP:")
print("Copy the output of this script (especially 'rhow bands', 'SPM/TSM',")
print("and 'Key variables check') and share it so the mosaic pipeline")
print("can be built with the exact variable names your files use.")
