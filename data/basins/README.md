# Basin Time Series Data

This directory stores the basin-level time series used by the Sto_XAJ_WaveML rainfall-runoff simulation framework.

The study evaluates the model using **182 selected basins from the 671-basin US CAMELS dataset**.

## Directory Contents

```text
basins/
├── README.md
├── selected_basin_ids.txt
└── 14301000.csv
```

- `selected_basin_ids.txt`: Lists the identifiers of the 182 selected basins.
- `14301000.csv`: Provides an example of the required basin time-series format.

The complete time series for all 182 selected basins are not included because of storage limitations.

## Selected Basin Identifiers

The file `selected_basin_ids.txt` records the basin identifiers selected for the experiments from the 671-basin US CAMELS dataset.

Use this list to determine which basin time series need to be prepared. The basin identifiers should be consistent with the filenames of the corresponding CSV files and with the identifiers in `data/static_data.csv`.

The list contains identifiers only; it does not include precipitation, potential evapotranspiration, or observed discharge data.

## Required CSV Format

Each basin CSV file should contain the following columns:

```csv
date,plui,etp,discharge
```

| Column | Description |
|---|---|
| `date` | Observation date |
| `plui` | Precipitation |
| `etp` | Potential evapotranspiration calculated using the FAO Penman equation |
| `discharge` | Observed streamflow |

Use the same column names and compatible units, temporal resolution, and date conventions across all basin files.

## File Naming

Name each basin file according to the basin identifier convention expected by the model scripts.

For example:

```text
14301000.csv
```

The basin identifier in the filename must be consistent with the identifier in `selected_basin_ids.txt`, the static attribute table, and the data-loading code.

## Preparing the Full Dataset

To reproduce the experiments:

1. Obtain the required US CAMELS data from the official source.
2. Refer to `selected_basin_ids.txt` to identify the 182 selected basins.
3. Prepare the precipitation and observed discharge time series for those basins.
4. Calculate potential evapotranspiration using the FAO Penman equation.
5. Export each basin's time series to a CSV file with the required column names.
6. Place the files in this directory.
7. Verify that the filenames and basin identifiers match `selected_basin_ids.txt`, `data/static_data.csv`, and the model configuration.

Official CAMELS dataset page:

https://ral.ucar.edu/solutions/products/camels

Please follow the original data provider's terms of use.

## Data Availability Note

This repository provides the selected basin identifiers and one example time-series file for documentation and format verification. It does not contain the complete time series for all 182 basins.

Users must obtain or prepare the remaining basin data independently to reproduce the complete experiments.
