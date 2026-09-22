# Diabetes Prediction Service

Production-oriented machine learning service for diabetes risk prediction using
the Pima Indians Diabetes dataset. The project contains a leakage-safe,
calibrated SVM pipeline, a FastAPI backend, a Streamlit frontend, automated
tests, and Docker Compose deployment.

## Architecture

```text
Streamlit UI (8501) --> FastAPI API (8000) --> sklearn model pipeline
                                             |
                                             +--> checksum metadata
```

The API loads one serialized pipeline containing preprocessing, imputation,
scaling, classification, and probability calibration. The Streamlit app does
not load the model directly; it calls the FastAPI endpoint over HTTP.

## Project structure

```text
main.py                         FastAPI application
app.py                          Streamlit frontend
docker-compose.yml              API and UI orchestration
Dockerfile                      FastAPI image
Dockerfile.streamlit             Streamlit image
pyproject.toml                  Project dependencies and optional extras
uv.lock                         Reproducible dependency lockfile
src/
  data/diabetes.csv             Source dataset
  models/diabetes_pipeline.pkl  Trained model pipeline
  models/training_metrics.json  Metrics and artifact checksum
  ml_pipeline.py                Shared training and prediction logic
  train.py                      Training command-line entry point
  notebooks/                    EDA, preprocessing, training, evaluation, deployment
tests/                          API and ML pipeline tests
```

## Requirements

- Python 3.12 or newer
- `uv`
- Docker Engine and Docker Compose, for containerized execution

Install `uv` using the official instructions at
<https://docs.astral.sh/uv/getting-started/installation/>.

## Local setup

From the project root:

```bash
uv sync
```

Install the optional Streamlit dependencies when running the frontend locally:

```bash
uv sync --extra ui
```

Install notebook visualization dependencies when working with EDA notebooks:

```bash
uv sync --extra notebooks
```

`uv.lock` is committed and should be used for reproducible environments. Use
`uv sync --frozen` in automated or deployment environments.

## Train the model

Training validates the input schema, numeric values, target classes, and feature
ranges. It performs a stratified train/test split and cross-validated
hyperparameter search without fitting preprocessing on the test data.

```bash
uv run python src/train.py
```

Generated artifacts:

- `src/models/diabetes_pipeline.pkl`
- `src/models/training_metrics.json`

The metadata file records accuracy, ROC-AUC, selected parameters, dataset hash,
model hash, pipeline version, scikit-learn version, and training timestamp.

## Run locally

Start the FastAPI backend:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Start the Streamlit frontend in a second terminal:

```bash
uv run --extra ui streamlit run app.py --server.port 8501
```

Open the frontend at <http://localhost:8501>.

FastAPI endpoints:

- API root: <http://localhost:8000/>
- Health check: <http://localhost:8000/health>
- Interactive API documentation: <http://localhost:8000/docs>
- Prediction endpoint: `POST http://localhost:8000/diabetes_prediction`

The frontend uses `API_URL` to locate the backend. It defaults to
`http://localhost:8000`:

```bash
API_URL=http://localhost:8000 uv run --extra ui streamlit run app.py
```

## Run with Docker Compose

Build and start both services:

```bash
docker compose up --build
```

Open:

- Streamlit: <http://localhost:8501>
- FastAPI: <http://localhost:8000>
- FastAPI docs: <http://localhost:8000/docs>

Run in the background:

```bash
docker compose up --build -d
```

View service status and logs:

```bash
docker compose ps
docker compose logs -f api
docker compose logs -f ui
```

Stop the services:

```bash
docker compose down
```

If ports are already in use, choose alternative host ports. The values before
the colon are host ports; container ports remain unchanged:

```bash
API_PORT=8001 UI_PORT=8502 docker compose up --build
```

Then open <http://localhost:8502> for Streamlit and
<http://localhost:8001/docs> for FastAPI.

## Configuration

The FastAPI service supports these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `src/models/diabetes_pipeline.pkl` | Model artifact location |
| `METRICS_PATH` | `src/models/training_metrics.json` | Model metadata location |
| `PORT` | `8000` | API container port |

The Streamlit service supports:

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_URL` | `http://localhost:8000` | FastAPI base URL |
| `UI_PORT` | `8501` | Compose host port for Streamlit |
| `API_PORT` | `8000` | Compose host port for FastAPI |

At startup, FastAPI verifies the model SHA-256 checksum recorded in the
metadata file. If the artifact is missing or altered, the service fails closed
instead of serving predictions from an unverified model.

## Testing and quality checks

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

Compile the Python source:

```bash
uv run python -m compileall -q src tests main.py app.py
```

Validate the dependency lockfile:

```bash
uv lock --check
```

## Notebook workflow

The notebooks are deliberately separated by responsibility:

1. `00_EDA.ipynb` - exploratory data analysis
2. `01_Data_Preprocessing.ipynb` - schema and preprocessing validation
3. `02_Model_Training.ipynb` - model training and artifact creation
4. `03_Model_Evaluation.ipynb` - held-out evaluation
5. `04_Deployment.ipynb` - API smoke test and deployment commands

The reusable implementation in `src/ml_pipeline.py` remains the source of
truth. Notebooks call that implementation instead of maintaining a separate
production algorithm.

## Production notes

- Do not commit secrets or local `.env` files.
- Only load trusted model artifacts because joblib uses Python serialization.
- Keep the model artifact and its metadata file from the same training run.
- Monitor request latency, error rate, prediction distribution, and model drift
  in a production environment.
- This model is an educational risk prediction service, not a medical
  diagnosis or a substitute for professional clinical judgment.

## Dataset

The original dataset is available from Kaggle:
<https://www.kaggle.com/datasets/mathchi/diabetes-data-set>
