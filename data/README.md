# Data Directory

This directory documents the input data required for the Sto_XAJ_WaveML rainfall-runoff simulation framework.

The study evaluates the proposed framework using 182 selected basins from the 671-basin US CAMELS dataset.

## Directory Structure

```text
data/
├── README.md
├── static_data.csv
└── basins/
    ├── README.md
    ├── selected_basin_ids.txt
    └── 14301000.csv
```

The repository includes a list of the selected basin identifiers and one example basin time series. The complete time series for all 182 selected basins are not included because of storage limitations.

## Data Source

The original CAMELS dataset is available from the official provider:

https://ral.ucar.edu/solutions/products/camels

Users should consult the official dataset documentation and comply with its terms of use.

## Static Basin Attributes

The file `static_data.csv` contains static attributes associated with the selected basins.

Expected columns:

```csv
gauge_id,elev_mean,slope_mean,area_gages2
```

| Column | Description |
|---|---|
| `gauge_id` | Basin or gauge identifier |
| `elev_mean` | Mean basin elevation |
| `slope_mean` | Mean basin slope |
| `area_gages2` | Basin drainage area |

The column names and identifier format should remain consistent with the source code.

## Basin Time Series

Each basin time series is stored as a separate CSV file in the `basins/` directory.

Expected columns:

```csv
date,plui,etp,discharge
```

| Column | Description |
|---|---|
| `date` | Observation date |
| `plui` | Precipitation |
| `etp` | Potential evapotranspiration |
| `discharge` | Observed streamflow |

Potential evapotranspiration (`etp`) is calculated using the FAO Penman equation.

The exact units, temporal resolution, missing-value handling, and date conventions should be consistent across all basin files and with the model implementation.

## Basin Selection and Preparation

The research uses 182 basins selected from the 671-basin US CAMELS dataset.

To reproduce the experiments:

1. Obtain the required CAMELS data from the official provider.
2. Identify the 182 basins used in the study.
3. Prepare the corresponding precipitation, potential evapotranspiration, and observed discharge time series.
4. Calculate potential evapotranspiration using the FAO Penman equation, as required by the workflow.
5. Format the basin files using the expected column names.
6. Prepare `static_data.csv` and ensure that basin identifiers match the time-series file names.
7. Configure the data paths in the model scripts.

## Data Availability

The repository provides basins/selected_basin_ids.txt and one example basin file, basins/14301000.csv, for documentation and format verification.

The complete time series for all 182 selected basins are not distributed here because of storage limitations. Users must obtain the required data from the original CAMELS source and prepare the selected basin files independently.

Redistribution of original or processed data should comply with the applicable data access and licensing terms.
