# CGSM-TSS-Analysis

Repository accompanying the manuscript

**Physically-Constrained Calibration and Transferability of a Semi-Analytical TSS Retrieval Model in a Tropical Coastal Lagoon System: Ciénaga Grande de Santa Marta, Colombia**

submitted to *Remote Sensing*.

---

## Overview

This repository contains the datasets, Python scripts, and manuscript source used to reproduce the analyses presented in the paper.

The workflow includes:

- preprocessing of Landsat-8 and Sentinel-2 matchup datasets;
- calibration of the semi-analytical Nechad TSS model;
- bootstrap-based parameter identifiability assessment;
- transferability experiments between Pajarales and the Ciénaga Grande de Santa Marta;
- comparison with machine-learning models;
- generation of manuscript figures.

---

## Repository structure

```
code/
    Python scripts

data/
    Input and processed datasets

images/
    Figures used in the manuscript

main.tex
    Manuscript source

README.md
```

---

## Requirements

Python 3.11+

Required packages include:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- xarray

---

## Reproducing the analyses

Examples:

```bash
python code/02_candidate_comparison.py

python code/03_cgsm_transfer_scenarios.py

python code/05_bootstrap_identifiability.py
```

Each script reproduces the corresponding analyses described in the manuscript.

---

## Data

The repository includes the processed satellite–in situ matchup datasets used in this study.

Original satellite imagery can be obtained from:

- USGS EarthExplorer (Landsat-8)
- Copernicus Data Space Ecosystem (Sentinel-2)

---

## Citation

If you use this repository, please cite:

Rodríguez J., et al.

*Physically-Constrained Calibration and Transferability of a Semi-Analytical TSS Retrieval Model in a Tropical Coastal Lagoon System: Ciénaga Grande de Santa Marta, Colombia.*

(Remote Sensing, under review)

---

## License

MIT License
