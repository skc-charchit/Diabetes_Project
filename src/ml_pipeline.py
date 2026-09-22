from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PIPELINE_VERSION = "1.0.0"
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_NAME = "Outcome"
MEASUREMENT_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]
FEATURE_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 1000),
    "BMI": (0, 100),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (1, 120),
}


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data = pd.read_csv(data_path)
    required_columns = set(FEATURE_NAMES + [TARGET_NAME])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    return data[FEATURE_NAMES + [TARGET_NAME]].copy()


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if data.empty:
        raise ValueError("Dataset must contain at least one row")
    data = data.drop_duplicates().reset_index(drop=True)
    if data.empty:
        raise ValueError("Dataset must contain at least one unique row")
    for column in FEATURE_NAMES + [TARGET_NAME]:
        original_values = data[column]
        numeric_values = pd.to_numeric(original_values, errors="coerce")
        invalid_values = numeric_values.isna() & original_values.notna()
        if invalid_values.any():
            raise ValueError(f"{column} contains non-numeric values")
        if np.isinf(numeric_values.dropna()).any():
            raise ValueError(f"{column} contains non-finite values")
        data[column] = numeric_values
    if data[TARGET_NAME].isna().any():
        raise ValueError("Outcome must contain numeric values")
    if not data[TARGET_NAME].isin([0, 1]).all():
        raise ValueError("Outcome must contain only binary values 0 and 1")
    for column, (minimum, maximum) in FEATURE_RANGES.items():
        observed_values = data[column].dropna()
        if ((observed_values < minimum) | (observed_values > maximum)).any():
            raise ValueError(f"{column} contains values outside [{minimum}, {maximum}]")
    data[MEASUREMENT_COLUMNS] = data[MEASUREMENT_COLUMNS].replace(0, np.nan)
    features = data[FEATURE_NAMES]
    target = data[TARGET_NAME].astype(int)
    return features, target


def build_pipeline() -> Pipeline:
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessor, FEATURE_NAMES)]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=SVC(random_state=42), cv=5, ensemble=False
                ),
            ),
        ]
    )


def train_model(
    data_path: Path,
    model_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    features, target = prepare_features(load_dataset(data_path))
    if target.nunique() < 2:
        raise ValueError("Dataset must contain both outcome classes")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "classifier__estimator__C": [0.1, 1, 10],
            "classifier__estimator__kernel": ["linear", "rbf"],
            "classifier__estimator__gamma": ["scale", "auto"],
        },
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    predictions = search.predict(X_test)
    probabilities = search.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
        "best_params": search.best_params_,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "pipeline_version": PIPELINE_VERSION,
        "feature_names": FEATURE_NAMES,
        "data_sha256": sha256_file(data_path),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
    joblib.dump(search.best_estimator_, temporary_model_path)
    temporary_model_path.replace(model_path)
    metrics["model_sha256"] = sha256_file(model_path)
    return {"model": search.best_estimator_, "metrics": metrics}


def predict(model: Pipeline, values: dict[str, float | int]) -> tuple[int, float]:
    missing_features = set(FEATURE_NAMES).difference(values)
    if missing_features:
        raise ValueError(f"Missing features: {sorted(missing_features)}")
    row = pd.DataFrame([{name: values[name] for name in FEATURE_NAMES}])
    row[MEASUREMENT_COLUMNS] = row[MEASUREMENT_COLUMNS].replace(0, np.nan)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return prediction, probability
