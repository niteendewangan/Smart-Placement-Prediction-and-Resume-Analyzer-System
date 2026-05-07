import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "placement_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "placement_model_metadata.json"


@lru_cache(maxsize=1)
def load_model_artifact() -> dict[str, object]:
    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {
        "model": artifact,
        "model_name": "Legacy Model",
        "feature_columns": ["cgpa", "placement_exam_marks"],
        "target_column": "placed",
    }


def get_model_metadata() -> dict[str, object]:
    if not METADATA_PATH.exists():
        return {}

    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def predict_placement(features) -> float:
    artifact = load_model_artifact()
    feature_columns = artifact["feature_columns"]

    if isinstance(features, dict):
        ordered_features = [features[column] for column in feature_columns]
    else:
        ordered_features = list(features)

    if len(ordered_features) != len(feature_columns):
        raise ValueError(
            f"Expected {len(feature_columns)} features: {', '.join(feature_columns)}"
        )

    feature_array = pd.DataFrame(
        [np.array(ordered_features, dtype=float)],
        columns=feature_columns,
    )
    prediction = artifact["model"].predict_proba(feature_array)[0][1]
    return round(float(prediction) * 100, 2)
