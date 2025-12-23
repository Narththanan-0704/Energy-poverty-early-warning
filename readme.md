## How to run
pip install -r requirements.txt
python src/01_build_panel.py
python src/02_add_deprivation.py
python src/03_train_forecast.py
python src/04_anomaly_risk.py
python src/05_make_figures.py

## Outputs
- data/processed/panel.parquet
- data/processed/panel_with_imd.parquet
- output/tables/anomaly_drop_all_years.csv
- output/tables/forecast_metrics.csv
- output/tables/top50_risk_latest_year.csv
- output/figures/*.png