import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.preprocessing import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    prepare_placement_dataset,
    summarize_placement_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "placement_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "placement_model_metadata.json"


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[str, object, list[dict[str, float]]]:
    candidate_models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42,
            class_weight="balanced",
        ),
    }

    leaderboard = []
    fitted_models = {}

    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        leaderboard.append(
            {
                "model": model_name,
                "accuracy": round(accuracy_score(y_test, predictions), 4),
                "precision": round(precision_score(y_test, predictions), 4),
                "recall": round(recall_score(y_test, predictions), 4),
                "f1": round(f1_score(y_test, predictions), 4),
                "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
            }
        )

    leaderboard.sort(
        key=lambda item: (item["roc_auc"], item["accuracy"], item["f1"]),
        reverse=True,
    )
    best_model_name = leaderboard[0]["model"]
    return best_model_name, fitted_models[best_model_name], leaderboard


def train_and_save_model() -> dict[str, object]:
    cleaned_dataframe = prepare_placement_dataset()
    X = cleaned_dataframe[FEATURE_COLUMNS]
    y = cleaned_dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    best_model_name, best_model, leaderboard = evaluate_models(X_train, X_test, y_train, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        MODEL_PATH,
    )

    best_metrics = next(item for item in leaderboard if item["model"] == best_model_name)
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_path": "data/raw/placement.csv",
        "processed_data_path": str(PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT)),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "dataset_summary": summarize_placement_dataset(cleaned_dataframe),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_model_name,
        "selection_metric": "roc_auc",
        "test_metrics": best_metrics,
        "leaderboard": leaderboard,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return metadata


if __name__ == "__main__":
    training_metadata = train_and_save_model()
    print(json.dumps(training_metadata, indent=2))
