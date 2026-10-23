## Core Python source code for the Sto-XAJ-WaveML framework.


## File Organization

| File | Purpose |
|---|---|
| `Det_XAJ.py` | Deterministic Xin'anjiang model used as the baseline. |
| `Sto_XAJ_W.py` | Stochastic XAJ configuration associated with the soil moisture storage state variable \(W\). |
| `Sto_XAJ_S.py` | Stochastic XAJ configuration associated with the state variable \(S\). |
| `Sto_XAJ_Qg.py` | Stochastic XAJ configuration associated with the state variable \(Qg\). |
| `Sto_XAJ_Qi.py` | Stochastic XAJ configuration associated with the state variable \(Qi\). |
| `Sto_XAJ_Qs.py` | Stochastic XAJ configuration associated with the state variable \(Qs\). |
| `Sto_XAJ_GRU_1.py` | GRU-based machine learning configuration 1. |
| `Sto_XAJ_GRU_2.py` | GRU-based machine learning configuration 2. |
| `Sto_XAJ_XGBoost_1.py` | XGBoost-based machine learning configuration 1. |
| `Sto_XAJ_XGBoost_2.py` | XGBoost-based machine learning configuration 2. |

The exact role of each configuration should be interpreted together with its implementation and experimental settings.


## Modeling Workflow

The source code supports the following general research workflow:

1. Run the deterministic XAJ model as a baseline.
2. Evaluate stochastic XAJ configurations with noise introduced into alternative hydrological state variables.
3. Identify the most effective stochastic perturbation node.
4. Generate ensemble discharge trajectories from the selected Sto-XAJ configuration.
5. Decompose discharge trajectory features using wavelet transform.
6. Train and evaluate machine learning configurations using the specified input schemes.


## Data Requirements

The scripts require basin-level precipitation, potential evapotranspiration, observed discharge, and any static attributes required by the implementation.

The example basin data format is:

```csv
date,plui,etp,discharge
```

Potential evapotranspiration is calculated using the FAO Penman equation.

The input paths, basin identifiers, date ranges, and output directories must match the settings used by each script.
