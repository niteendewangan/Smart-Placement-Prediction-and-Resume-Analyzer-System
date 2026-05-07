from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "placement.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "placement.csv"

FEATURE_COLUMNS = ["cgpa", "placement_exam_marks"]
TARGET_COLUMN = "placed"
REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


def load_raw_placement_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    dataframe.columns = [column.strip().lower() for column in dataframe.columns]
    return dataframe


def clean_placement_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    if missing_columns:
        raise ValueError(
            f"Placement dataset is missing required columns: {', '.join(missing_columns)}"
        )

    cleaned = cleaned[REQUIRED_COLUMNS].drop_duplicates()

    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned[cleaned["cgpa"].between(0, 10)]
    cleaned = cleaned[cleaned["placement_exam_marks"].between(0, 100)]
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin([0, 1])]
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    return cleaned.reset_index(drop=True)


def save_processed_placement_data(
    dataframe: pd.DataFrame,
    path: Path | str = PROCESSED_DATA_PATH,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def prepare_placement_dataset(
    raw_path: Path | str = RAW_DATA_PATH,
    processed_path: Path | str = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    raw_dataframe = load_raw_placement_data(raw_path)
    cleaned_dataframe = clean_placement_data(raw_dataframe)
    save_processed_placement_data(cleaned_dataframe, processed_path)
    return cleaned_dataframe


def summarize_placement_dataset(dataframe: pd.DataFrame) -> dict[str, float]:
    if dataframe.empty:
        return {
            "sample_size": 0,
            "placement_rate": 0.0,
            "average_cgpa": 0.0,
            "average_exam_marks": 0.0,
        }

    return {
        "sample_size": int(len(dataframe)),
        "placement_rate": round(dataframe[TARGET_COLUMN].mean() * 100, 2),
        "average_cgpa": round(dataframe["cgpa"].mean(), 2),
        "average_exam_marks": round(dataframe["placement_exam_marks"].mean(), 2),
    }
