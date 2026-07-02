# Energy Poverty Early-Warning Framework

This repository contains the final implementation developed for an MSc dissertation titled:

**An Explainable Energy Poverty Early-Warning Framework for England Using Postcode-Level Electricity Consumption and Deprivation: Forecasting, Uncertainty-Aware Anomaly Detection, and Risk Scoring**

## Development environment

The project was developed and executed entirely in Google Colab using Python.

The final implementation is provided as a Jupyter/Google Colab notebook:

`notebooks/Energy_Poverty_Early_Warning_Final.ipynb`

## Project functions

The notebook performs:

- ingestion of DESNZ postcode-level electricity data for 2019–2024;
- postcode and geographic standardisation;
- linkage with ONSPD and IMD 2019;
- leakage-free feature engineering;
- forecasting model comparison;
- Random Forest validation and deployment;
- calibrated 90% prediction intervals;
- drop and spike anomaly classification;
- explainable 0–100 risk scoring;
- sensitivity analysis;
- permutation importance; and
- nationwide 2024 postcode deployment.

## How to run

1. Open the final notebook in Google Colab.
2. Mount Google Drive when requested.
3. Place the required DESNZ, ONSPD and IMD source files in the expected Google Drive raw-data directory.
4. Run the notebook cells sequentially from top to bottom.

## Data availability

Large raw and processed datasets are not stored in this repository because of their size.

The source datasets are publicly available from:

- Department for Energy Security and Net Zero;
- Office for National Statistics; and
- English Indices of Deprivation 2019.

## Important interpretation

The output is a postcode-level screening and prioritisation framework. It does not identify individual households and does not provide an official fuel-poverty diagnosis.
