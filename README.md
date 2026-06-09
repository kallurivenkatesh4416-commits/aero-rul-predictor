# Aero RUL Predictor

Production-style machine learning project for predicting the Remaining Useful Life
(RUL) of aircraft turbofan engines from NASA C-MAPSS sensor data.

The project trains an XGBoost regression model, saves a reusable model artifact,
and exposes predictions through a FastAPI service. In addition to a numeric RUL
estimate, the API converts each prediction into a maintenance risk category:
`Critical`, `Warning`, or `Healthy`.

## Highlights

- Predictive maintenance use case based on multivariate engine time-series data.
- Leakage-safe train/validation split by engine ID.
- RUL target capping for more stable model training.
- Feature selection that removes constant sensors, engine IDs, cycle counters,
  and target columns.
- Tuned XGBoost model saved with feature metadata and validation metrics.
- FastAPI inference API with request/response schemas.
- Dockerized API for local or containerized serving.
- Experiment, final evaluation, and interpretation notebooks.

## Business Problem

Aircraft engines generate continuous operational and sensor data. Maintenance
teams need to estimate how many operating cycles remain before an engine reaches
failure so they can plan inspections, repairs, and replacements.

This project predicts engine RUL from operational settings and sensor readings.
The model output is also mapped into practical risk bands:

| Risk category | Predicted RUL |
| --- | --- |
| `Critical` | `<= 30` cycles |
| `Warning` | `31-70` cycles |
| `Healthy` | `> 70` cycles |

## Dataset

This project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation
dataset, specifically the FD001 subset:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

FD001 contains one operating condition, one fault mode, 100 training engines,
and 100 test engines. Training engines run until failure, allowing the RUL target
to be calculated. Test engines stop before failure, and NASA provides the true
remaining RUL values separately.

Download the dataset from the
[NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data),
then place the extracted files here:

```text
data/raw/cmapss/CMAPSSData/
```

The expected raw data path for FD001 is:

```text
data/raw/cmapss/CMAPSSData/train_FD001.txt
data/raw/cmapss/CMAPSSData/test_FD001.txt
data/raw/cmapss/CMAPSSData/RUL_FD001.txt
```

## Project Structure

```text
aero-rul-predictor/
|-- api/
|   `-- app.py                  # FastAPI application
|-- data/
|   `-- raw/cmapss/CMAPSSData/  # NASA C-MAPSS files
|-- models/
|   `-- xgboost_rul_model.joblib
|-- notebooks/
|   |-- 01_experiments.ipynb
|   |-- 02_final_evaluation.ipynb
|   `-- 03_results_interpretation.ipynb
|-- reports/
|   |-- official_test_predictions.csv
|   |-- official_test_predictions_with_risk_categories.csv
|   `-- threshold_comparison.csv
|-- src/
|   |-- config.py
|   |-- data_processing.py
|   |-- features.py
|   |-- inference.py
|   `-- train.py
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Setup

Python 3.11 is recommended. The Docker image uses `python:3.11-slim`.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the Model

After the FD001 files are in `data/raw/cmapss/CMAPSSData/`, run:

```bash
python -m src.train
```

The training pipeline:

1. Loads the raw FD001 training data.
2. Creates a capped RUL target.
3. Splits training and validation data by engine ID.
4. Selects non-constant model features.
5. Trains the tuned XGBoost regressor.
6. Evaluates validation performance.
7. Saves the model artifact to `models/xgboost_rul_model.joblib`.

The saved artifact contains:

- Trained XGBoost model.
- Feature column list.
- Model name.
- Validation metrics.

Current saved model validation metrics:

| Metric | Value |
| --- | ---: |
| MAE | 12.12 |
| RMSE | 16.76 |
| R2 score | 0.839 |

During training, RUL values above 125 cycles are capped at 125. This helps the
model focus on the maintenance-critical degradation region instead of trying to
distinguish between very healthy engines with long remaining life.

## Run the API Locally

Start the FastAPI server:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/
```

Prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "operational_setting_1": 0.0023,
    "operational_setting_2": 0.0003,
    "sensor_2": 642.15,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_6": 21.61,
    "sensor_7": 554.36,
    "sensor_8": 2388.06,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 392.0,
    "sensor_20": 39.06,
    "sensor_21": 23.4190
  }'
```

Example response:

```json
{
  "predicted_rul": 120.07,
  "risk_category": "Healthy"
}
```

## Run with Docker

Build the image:

```bash
docker build -t aero-rul-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 aero-rul-api
```

Then visit:

```text
http://localhost:8000/docs
```

Note: the Docker image copies the `models/` directory. Train the model first if
`models/xgboost_rul_model.joblib` is missing.

## Model Inputs

The trained model expects these 17 features:

```text
operational_setting_1
operational_setting_2
sensor_2
sensor_3
sensor_4
sensor_6
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

Constant sensors and non-predictive identifier/time columns are excluded during
feature selection.

## Reports and Notebooks

The notebooks document the modeling workflow:

- `notebooks/01_experiments.ipynb`: exploration, validation strategy, model
  comparison, and tuning.
- `notebooks/02_final_evaluation.ipynb`: final evaluation on the official test
  set.
- `notebooks/03_results_interpretation.ipynb`: risk category analysis and
  threshold comparison.

Generated report files are stored in `reports/`.

Risk threshold comparison from the current reports:

| Strategy | Warning threshold | Overall accuracy | Critical alert recall | Critical missed as healthy |
| --- | ---: | ---: | ---: | ---: |
| Normal Threshold | 60 | 87.0% | 92.0% | 2 |
| Conservative Threshold | 70 | 86.0% | 100.0% | 0 |

The API currently uses the conservative threshold: predictions up to 70 cycles
are treated as `Warning`, and predictions up to 30 cycles are treated as
`Critical`.

## Notes

- This project currently targets FD001 only.
- The API predicts one engine sensor reading per request.
- Predictions depend on the saved model artifact matching the expected feature
  list.
- The risk categories are rule-based thresholds applied after the regression
  prediction.
