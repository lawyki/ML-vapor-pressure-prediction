# Vapor Pressure Prediction from Molecular Fingerprints

Predicting **log₁₀(saturation vapor pressure)** of atmospheric molecules from 
Morgan fingerprints (ECFP4) alone — no thermodynamic features, no hand-crafted 
descriptors.

## Why this is hard

Vapor pressure is physically linked to enthalpy of vaporization via 
Clausius-Clapeyron:

    ln(pSat) ∝ −ΔHvap / RT

Features like `HeatOfVap`, `ChemPot`, and `FreeEnergy` are excluded — they are 
rarely known experimentally and would trivialize the regression. The model must 
learn implicit thermodynamic structure from molecular topology alone.

## Dataset

~31,000 atmospheric molecules from:

> L. Lind, H. Sandström, P. Rinke; *An interpretable molecular descriptor for 
> machine learning predictions in atmospheric science.*  
> J. Chem. Phys. 164, 084115 (2026).  
> https://doi.org/10.1063/5.0308548

Data: https://zenodo.org/records/18669279

## Features

ECFP4 Morgan fingerprints (radius=2, 2048 bits) generated directly from SMILES 
strings. No RDKit descriptor preprocessing — structure only.

## Experiments

| Version | Model | MAE (log₁₀) |
|---------|-------|-------------|
| v2 | MLP baseline (AdamW, dropout, early stopping) | 0.8425 |
| v3 | 5-seed MLP ensemble + SWA | 0.7252 |
| v4 | Optuna XGB ensemble (5-seed bagging) | **0.6490** |

MAE of 0.65 in log₁₀ space ≈ factor of 4.5 error in raw pSat.

## Finding

MLPs struggle with sparse binary fingerprints even with ensembling. 
Gradient boosting handles them natively — the v3→v4 gap (0.73→0.65) comes 
from model family, not hyperparameter tuning.

## Limitations

- Single train/test split (not cross-validated) — numbers carry ~±0.01 split variance
- Random row-level split — if duplicate SMILES exist, absolute MAE may be optimistic
- Thermodynamic ceiling: without ΔHvap and related features, MAE ~0.65 appears 
  to be the structural floor for this feature set

## Stack

Python 3.14, PyTorch, XGBoost, Optuna, RDKit, scikit-learn, pandas, numpy

## Status

Active — last updated 2026-05-19.
