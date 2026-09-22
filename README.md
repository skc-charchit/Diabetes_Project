# Diabetes Prediction: Machine Learning Pipeline and Deployment

## Executive Summary

This project develops and deploys a binary classification model for estimating
diabetes risk from the Pima Indians Diabetes dataset. It combines exploratory
data analysis, validated preprocessing, calibrated SVM training, model
evaluation, and a production-style serving layer.

The final system exposes a FastAPI prediction service and a Streamlit interface.
The Streamlit application sends requests to FastAPI; the model is loaded and
validated only by the API service. Docker Compose runs both services together.

> This is an educational risk-estimation project. It is not a medical diagnosis
> system and must not replace professional clinical judgment.

## Project Objectives

- Build a reproducible diabetes classification workflow.
- Treat physiologically impossible zero values consistently.
- Prevent preprocessing leakage between training and test data.
- Compare model performance using clinically meaningful classification metrics.
- Expose predictions through a validated HTTP API.
- Provide a usable frontend for local and containerized demonstrations.

## Dataset

The project uses the Pima Indians Diabetes dataset. It contains 768 patient
records, eight input features, and a binary `Outcome` target:

| Feature | Description |
| --- | --- |
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | Two-hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes hereditary risk score |
| `Age` | Age in years |
| `Outcome` | Target: `0` or `1` |

Source dataset: <https://www.kaggle.com/datasets/mathchi/diabetes-data-set>

## Methodology

### Data quality and preprocessing

Exploratory analysis identified zero values in measurement columns where zero is
not physiologically meaningful. The workflow treats zeros as missing for:

- Glucose
- Blood pressure
- Skin thickness
- Insulin
- BMI

`Pregnancies` and `Outcome` retain zero as a valid value. Missing measurements
are imputed with training-set medians.

All preprocessing is implemented inside the serialized scikit-learn pipeline.
This ensures that imputation and scaling are fitted only on training data and
then applied consistently during evaluation and inference.

### Model

The selected estimator is a calibrated Support Vector Machine:

- Standardized numeric features
- SVM classifier
- Five-fold probability calibration
- Five-fold cross-validated hyperparameter search
- ROC-AUC used as the model-selection metric
- Stratified 80/20 train/test split
- Fixed random seed: `42`

Selected configuration:

```text
C      = 0.1
kernel = linear
gamma  = scale
```

## Results

Evaluation was performed on a held-out test set of 154 records.

| Metric | Result |
| --- | ---: |
| Accuracy | 0.695 |
| ROC-AUC | 0.813 |
| Class 0 precision | 0.743 |
| Class 0 recall | 0.810 |
| Class 1 precision | 0.578 |
| Class 1 recall | 0.481 |
| Class 1 F1-score | 0.525 |

### Interpretation

- The ROC-AUC of approximately `0.813` indicates useful ranking ability across
  positive and negative cases.
- The model recognizes negative cases more reliably than positive cases.
- The positive-class recall of approximately `48%` means many diabetic cases are
  not detected at the current operating point.
- Accuracy alone is therefore insufficient for judging this model.
- A clinical deployment would require threshold tuning, calibration analysis,
  external validation, subgroup analysis, and stronger sensitivity requirements.

The metrics are stored in
[src/models/training_metrics.json](src/models/training_metrics.json).

## System Architecture

```text
User
  |
  v
Streamlit frontend :8501
  |
  | HTTP POST /diabetes_prediction
  v
FastAPI service :8000
  |
  v
Validated sklearn pipeline
  |
  +-- imputation
  +-- standardization
  +-- calibrated SVM
```

The API verifies the model SHA-256 checksum against the training metadata before
serving traffic. If the artifact is missing or modified, startup fails rather
than serving an unverified model.

## Repository Structure

```text
main.py                         FastAPI service
app.py                          Streamlit frontend
docker-compose.yml              API and frontend orchestration
Dockerfile                      FastAPI container
Dockerfile.streamlit             Streamlit container
pyproject.toml                  Dependencies and optional extras
uv.lock                         Locked dependency graph
src/
  data/diabetes.csv             Input dataset
  models/diabetes_pipeline.pkl  Serialized preprocessing and model pipeline
  models/training_metrics.json  Evaluation metrics and artifact metadata
  ml_pipeline.py                Shared training, validation, and inference logic
  train.py                      Reproducible training command
  notebooks/                    EDA, preprocessing, training, evaluation, deployment
tests/
  test_api.py                   API contract tests
  test_ml_pipeline.py           Data validation tests
```

## Running the Project

### Local execution

Install the base environment:

```bash
uv sync
```

Train or regenerate the model:

```bash
uv run python src/train.py
```

Start FastAPI:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

In another terminal, install the UI extra and start Streamlit:

```bash
uv sync --extra ui
uv run --extra ui streamlit run app.py --server.port 8501
```

Open the frontend at <http://localhost:8501>.

### Docker Compose

```bash
docker compose up --build
```

Services:

- Streamlit: <http://localhost:8501>
- FastAPI: <http://localhost:8000>
- API documentation: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

If the default ports are occupied:

```bash
API_PORT=8001 UI_PORT=8502 docker compose up --build
```

### API example

```bash
curl -X POST http://localhost:8000/diabetes_prediction \
  -H 'Content-Type: application/json' \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 125,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

The response contains the predicted class, estimated probability, and a human-
readable result.

## Validation and Quality Controls

The service and training pipeline enforce:

- Required feature and target columns
- Numeric input values
- Finite values
- Valid feature ranges
- Binary target values
- Extra-field rejection at the API boundary
- Atomic model writes during training
- Model checksum verification at API startup
- Reproducible train/test splitting

Run the automated tests:

```bash
uv run python -m unittest discover -s tests -v
```

Validate Python syntax and the dependency lock:

```bash
uv run python -m py_compile main.py app.py src/ml_pipeline.py src/train.py
uv lock --check
```

## Notebook Workflow

The analysis is separated into five notebooks:

1. `00_EDA.ipynb` - data exploration and quality analysis
2. `01_Data_Preprocessing.ipynb` - schema and transformation validation
3. `02_Model_Training.ipynb` - training and artifact generation
4. `03_Model_Evaluation.ipynb` - held-out performance evaluation
5. `04_Deployment.ipynb` - API smoke testing and deployment commands

The notebooks call the shared implementation in `src/ml_pipeline.py`; the
production logic is not duplicated across notebook cells.

## Limitations and Future Improvements

- The dataset is relatively small and may not represent current populations.
- There is no independent external validation dataset.
- Positive-class recall is not yet strong enough for clinical screening.
- Threshold selection is not optimized for a specific clinical cost function.
- Fairness and subgroup performance analysis should be added.
- Production monitoring should track latency, errors, drift, and prediction
  distribution.
- Model versioning should eventually move from local files to a trusted model
  registry.

## Conclusion

The project demonstrates a complete, reproducible ML lifecycle from data
exploration through deployment. The calibrated SVM provides meaningful ranking
performance, with an ROC-AUC of approximately `0.813`, and is packaged behind a
validated FastAPI contract with a Streamlit frontend.

The main technical conclusion is that the system is suitable as a working
engineering demonstration and research baseline. It is not yet suitable for
clinical decision-making because positive-case recall remains limited and the
model has not undergone external, fairness, or prospective validation.
