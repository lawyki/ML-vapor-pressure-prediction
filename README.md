# Vapor Pressure Prediction from Molecular Fingerprints

Predicting the **saturation vapor pressure** of atmospheric molecules from their
structure alone, using machine learning on Morgan fingerprints (ECFP4). No
thermodynamic inputs and no hand-crafted descriptors.

## In plain terms

Vapor pressure tells you how easily a molecule escapes into the air. Molecules
with very low vapor pressure condense onto tiny airborne particles (aerosols),
which scatter sunlight and help clouds form. So knowing it matters for air
quality and climate.

It has only been measured for a few compounds, and computing it with quantum
chemistry is slow. This project asks: **can a model learn vapor pressure just
from a molecule's structure?**

The best model here is off by a factor of about 4.5 on average, across values
spanning more than ten orders of magnitude. That matches a published graph
neural network on the same dataset, with a much simpler representation.

## How this started

During TET, the Finnish upper secondary school work-life orientation, a friend
and I asked Aalto University if we could spend two days with researchers. A
computer science PhD student showed us his work: predicting vapor pressure from
a molecule's skeleton to find which molecules could help form clouds. It was a
short introduction and demo.

I built everything in this repository myself afterwards. When my neural
networks reached about 0.73 (all errors here are mean absolute error in log₁₀
units), I told the PhD student. He shared the full dataset and pointed me to
better approaches, and the v4 gradient boosting models reached about 0.65,
roughly where published models on this dataset level off.

## Why this is hard

Vapor pressure is physically tied to the energy needed to vaporize a molecule
(the Clausius-Clapeyron relation):

    ln(pSat) ∝ −ΔHvap / RT

The dataset includes features like `HeatOfVap`, `ChemPot` and `FreeEnergy`, but
I deliberately left them out. They are rarely known for real molecules, and
including them would make the problem close to trivial. The model has to learn
that thermodynamic behaviour implicitly, from molecular topology only.

## Dataset

About 31,000 atmospheric molecules from:

> L. Lind, H. Sandström, P. Rinke; *An interpretable molecular descriptor for
> machine learning predictions in atmospheric science.*
> J. Chem. Phys. 164, 084115 (2026).
> https://doi.org/10.1063/5.0308548

Data: https://zenodo.org/records/18669279

The target is log₁₀(pSat), which compresses the huge range of values into
something a model can work with.

## Features

ECFP4 Morgan fingerprints (radius 2, 2048 bits) generated directly from SMILES
strings. Each bit marks whether a small structural fragment appears in the
molecule. Structure only, no RDKit descriptor preprocessing.

## Experiments

| Version | Model | MAE (log₁₀) |
|---------|-------|-------------|
| v2 | MLP baseline (AdamW, dropout, early stopping) | 0.8425 |
| v3 | 5-seed MLP ensemble + SWA | 0.7252 |
| v4 | Optuna-tuned XGBoost ensemble (5-seed bagging) | **0.6490** |

An MAE of 0.65 in log₁₀ space means the typical prediction is off by a factor of
about 4.5 in raw pSat.

For comparison, Krüger et al. (2025) used a graph convolutional neural network
on this same dataset and reached an MAE of roughly 0.66.

## Findings

**Model family mattered more than tuning.** Neural networks struggled with
sparse binary fingerprints even when ensembled. Gradient boosting handles them
natively, and the v3 to v4 improvement (0.73 to 0.65) comes from switching
model family, not from hyperparameter search.

**The remaining error looks like it is mostly in the data.** The
`v4_diagnostics` notebook checks whether the leftover 0.65 is model error (fixable
with better features) or noise in the reference values (not fixable). It looks
at the residual distribution, calibration, error across the pSat range,
correlation of errors with molecular properties, and the 20 worst predictions.
The pSat values in the dataset come from COSMOtherm calculations that average
over molecular conformations, which is a likely source of irreducible noise.

## Notebooks

| File | What it does |
|------|--------------|
| `ML_pytorch_analysis_v2.ipynb` | MLP baseline in PyTorch |
| `ML_pytorch_analysis_v3.ipynb` | MLP seed ensemble with stochastic weight averaging |
| `ML_pytorch_analysis_v4.ipynb` | Optuna search and XGBoost seed ensemble (best result) |
| `ML_pytorch_analysis_v4_diagnostics.ipynb` | Residual analysis: model error or data noise? |

## Running it

1. Install the dependencies: `pip install -r requirements.txt`
2. Download the dataset from Zenodo (link above).
3. Change the `pd.read_csv(...)` path in the notebook to point at your copy.
4. Run the notebook top to bottom.

All versions use the same split (`random_state=42`), so results are directly
comparable.

## Limitations

- Single train/test split, not cross-validated, so the numbers carry about
  ±0.01 of split variance.
- Random row-level split. If the dataset contains duplicate SMILES, the
  absolute MAE may be slightly optimistic.
- Without ΔHvap and related features, an MAE of about 0.65 appears to be the
  floor for this feature set.

## Stack

Python 3.14, PyTorch, XGBoost, Optuna, RDKit, scikit-learn, pandas, NumPy,
SciPy, Matplotlib

## Status

Active. Last updated 2026-11-23.
