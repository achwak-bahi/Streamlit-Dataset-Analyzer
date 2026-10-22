import io
import json
import tempfile
from pathlib import Path

import pandas as pd

import tp


DATA_DIR = Path(__file__).resolve().parent / "data"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".tsv"}


class DatasetError(ValueError):
    pass


DATASET_GROUPS = {
    "German Solar Farm": {
        "files": [f"german-solar-pv-{i:02d}.csv" for i in range(1, 22)],
        "joinable": True,
        "id_column": "site_id",
        "overall_file": "german-solar-pv-overall.csv",
        "description": "21 photovoltaic facilities in Germany with historical NWP weather and normalized power output.",
        "labels": {f"german-solar-pv-{i:02d}.csv": f"Site PV-{i:02d}" for i in range(1, 22)},
    },
    "Electrical Load": {
        "files": [
            "electrical-load-continuous.csv",
            "electrical-weekly-forecast.csv",
            "electrical-load-test.csv",
            "electrical-load-train.csv",
        ],
        "joinable": False,
        "id_column": None,
        "description": "Hourly power grid demand, weekly dispatch forecasts, test period, and 14-week training datasets.",
        "labels": {
            "electrical-load-continuous.csv": "Continuous Load (Hourly)",
            "electrical-weekly-forecast.csv": "Weekly Forecast (Pre-dispatch)",
            "electrical-load-test.csv": "Test Period (Week X)",
            "electrical-load-train.csv": "Training Data (All 14 Weeks)",
        },
    },
    "Parkinson's Disease": {
        "files": [
            "parkinson-data1.csv",
            "parkinson-data2.csv",
            "parkinson-scopa-aut.csv",
            "parkinson-montreal-cognitive-assessment-moca-.csv",
            "parkinson-epworth-sleepiness-scale.csv",
            "parkinson-rem-sleep-disorder-questionnaire.csv",
            "parkinson-geriatric-depression-scale-short-.csv",
            "parkinson-state-trait-anxiety-inventory.csv",
            "parkinson-cognitive-categorization.csv",
            "parkinson-cognitive-assessments.csv",
            "parkinson-benton-judgment-of-line-orientation.csv",
            "parkinson-hopkins-verbal-learning-test.csv",
            "parkinson-letter--number-sequencing-pd-.csv",
            "parkinson-semantic-fluency.csv",
            "parkinson-symbol-digit-modalities.csv",
            "parkinson-olfactory-upsit.csv",
            "parkinson-university-of-pennsylvania-smell-id-test.csv",
            "parkinson-features-of-rem-behavior-disorder.csv",
            "parkinson-quip-current-short.csv",
        ],
        "joinable": False,
        "id_column": None,
        "description": "Multimodal clinical motor and non-motor cognitive assessment studies.",
        "labels": {
            "parkinson-data1.csv": "Serum Biomarkers (Data 1)",
            "parkinson-data2.csv": "Serum Biomarkers (Data 2)",
            "parkinson-scopa-aut.csv": "SCOPA-AUT (Autonomic Symptoms)",
            "parkinson-montreal-cognitive-assessment-moca-.csv": "MoCA (Cognitive Assessment)",
            "parkinson-epworth-sleepiness-scale.csv": "Epworth Sleepiness Scale",
            "parkinson-rem-sleep-disorder-questionnaire.csv": "REM Sleep Disorder Questionnaire",
            "parkinson-geriatric-depression-scale-short-.csv": "Geriatric Depression Scale",
            "parkinson-state-trait-anxiety-inventory.csv": "State-Trait Anxiety Inventory",
            "parkinson-cognitive-categorization.csv": "Cognitive Categorization",
            "parkinson-cognitive-assessments.csv": "Cognitive Assessments",
            "parkinson-benton-judgment-of-line-orientation.csv": "Benton Line Orientation",
            "parkinson-hopkins-verbal-learning-test.csv": "Hopkins Verbal Learning",
            "parkinson-letter--number-sequencing-pd-.csv": "Letter-Number Sequencing",
            "parkinson-semantic-fluency.csv": "Semantic Fluency",
            "parkinson-symbol-digit-modalities.csv": "Symbol Digit Modalities",
            "parkinson-olfactory-upsit.csv": "UPSIT Olfactory Smell Test",
            "parkinson-university-of-pennsylvania-smell-id-test.csv": "Penn Smell ID Test",
            "parkinson-features-of-rem-behavior-disorder.csv": "REM Behavior Disorder Features",
            "parkinson-quip-current-short.csv": "QUIP (Impulsive-Compulsive)",
        },
    },
    "Heart Attack Analysis": {
        "files": ["heart-attack.csv", "heart-o2-saturation.csv"],
        "joinable": False,
        "id_column": None,
        "description": "Clinical heart attack risk factors and blood oxygen saturation measurements.",
        "labels": {
            "heart-attack.csv": "Heart Attack Risk Factors",
            "heart-o2-saturation.csv": "O2 Saturation Series",
        },
    },
    "Heart Disease Comprehensive": {
        "files": ["heart-disease.csv"],
        "joinable": False,
        "id_column": None,
        "description": "Comprehensive 5-hospital statlog clinical cardiovascular dataset (1,190 records).",
        "labels": {
            "heart-disease.csv": "Comprehensive Multi-Hospital Study",
        },
    },
    "Covid-19": {
        "files": ["covid19-dataset2.csv", "covid19-new-data.csv"],
        "joinable": False,
        "id_column": None,
        "description": "Patient laboratory diagnostic cohort and normalized biomarker screening.",
        "labels": {
            "covid19-dataset2.csv": "Clinical Patient Cohort (Dataset 2)",
            "covid19-new-data.csv": "Biomarker Screening (New Data)",
        },
    },
    "Photovoltaic (PV) Power": {
        "files": ["pv-power.csv", "pv-power-dataset2.csv"],
        "joinable": False,
        "id_column": None,
        "description": "PV solar plant power generation and inverter DC capacity time series.",
        "labels": {
            "pv-power.csv": "Plant Power Output (kW)",
            "pv-power-dataset2.csv": "Inverter DC Power Series",
        },
    },
    "Solar Irradiance": {
        "files": ["solar-irradiance-algeria.csv", "solar-irradiance-canopy.csv"],
        "joinable": False,
        "id_column": None,
        "description": "Direct and global solar irradiance measurements from Algeria and NIST Canopy.",
        "labels": {
            "solar-irradiance-algeria.csv": "Algeria PV Irradiance (Tlemcen)",
            "solar-irradiance-canopy.csv": "Canopy PV NIST (POA Irradiance)",
        },
    },
    "Traffic Flow": {
        "files": ["traffic-data-r1.csv", "traffic-data-r2.csv"],
        "joinable": True,
        "id_column": "route",
        "overall_file": "traffic-data-overall.csv",
        "description": "Traffic flow rate series across monitored arterial routes.",
        "labels": {
            "traffic-data-r1.csv": "Route 1 (Data R1)",
            "traffic-data-r2.csv": "Route 2 (Data R2)",
        },
    },
    "Alcohol": {
        "files": ["alcohol.csv"],
        "joinable": False,
        "id_column": None,
        "description": "Facial thermography and blood alcohol concentration classification.",
        "labels": {
            "alcohol.csv": "Alcohol & Facial Thermography",
        },
    },
    "Alzheimer MRI": {
        "files": ["alzheimer-mri-dataset.csv"],
        "joinable": False,
        "id_column": None,
        "description": "MRI brain scan metadata across 4 diagnostic dementia stages and splits.",
        "labels": {
            "alzheimer-mri-dataset.csv": "MRI Brain Scans Metadata",
        },
    },
    "Weather History": {
        "files": ["weather-history.csv"],
        "joinable": False,
        "id_column": None,
        "description": "Detailed hourly weather observations for Szeged, Hungary (2006-2016).",
        "labels": {
            "weather-history.csv": "Szeged Hourly Weather History",
        },
    },
}


def list_datasets():
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def get_dataset_groups():
    return DATASET_GROUPS


def get_group_info(group_name):
    return DATASET_GROUPS.get(group_name, {})


def get_group_files(group_name):
    group = DATASET_GROUPS.get(group_name)
    if not group:
        return {}
    files = group.get("files", [])
    labels = group.get("labels", {})
    return {labels.get(f, f): f for f in files}


def get_subsets_summary(group_name):
    """Returns a DataFrame summarizing all individual subsets in group_name."""
    group = DATASET_GROUPS.get(group_name)
    if not group:
        return pd.DataFrame()

    files = group.get("files", [])
    labels = group.get("labels", {})
    records = []

    for f in files:
        path = DATA_DIR / f
        if not path.is_file():
            continue

        size_kb = path.stat().st_size / 1024
        size_str = f"{size_kb / 1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"

        try:
            with open(path, "rb") as fp:
                row_count = sum(1 for _ in fp) - 1
        except Exception:
            row_count = 0

        try:
            sample = pd.read_csv(path, nrows=5)
            n_cols = sample.shape[1]
            n_num = len(sample.select_dtypes(include="number").columns)
        except Exception:
            n_cols = 0
            n_num = 0

        records.append({
            "Subset": labels.get(f, Path(f).stem),
            "Filename": f,
            "Rows": f"{row_count:,}",
            "Columns": n_cols,
            "Numeric Features": n_num,
            "Size": size_str,
        })

    return pd.DataFrame(records)


def get_group_dataframe(group_name, subset_file=None):
    """Returns the DataFrame for the requested group view.
    If subset_file is None or '__overall__':
      - For joinable groups: returns the concatenated DataFrame with id_column.
      - For non-joinable groups: returns the first subset's DataFrame.
    If subset_file is given, returns that subset's DataFrame.
    """
    group = DATASET_GROUPS.get(group_name)
    if not group:
        return None

    files = group.get("files", [])
    if not files:
        return None

    if subset_file and subset_file != "__overall__":
        if subset_file in files or (DATA_DIR / subset_file).is_file():
            return get_dataframe(dataset_name=subset_file)
        return None

    # Overall view
    if group.get("joinable", False) and len(files) > 1:
        overall_file = group.get("overall_file")
        if overall_file and (DATA_DIR / overall_file).is_file():
            return get_dataframe(dataset_name=overall_file)

        # Dynamic concatenation
        id_col = group.get("id_column") or "site"
        labels = group.get("labels", {})
        dfs = []
        for f in files:
            df = get_dataframe(dataset_name=f)
            if df is not None:
                df = df.copy()
                label = labels.get(f, Path(f).stem)
                df.insert(0, id_col, label)
                dfs.append(df)
        if dfs:
            return pd.concat(dfs, ignore_index=True)

    return get_dataframe(dataset_name=files[0])


def analyze_group(group_name, subset_file=None):
    """Analyzes the group's Overall joined dataset or a specific subset."""
    group = DATASET_GROUPS.get(group_name)
    if not group:
        raise DatasetError(f"Dataset group '{group_name}' not found.")

    files = group.get("files", [])
    if not files:
        raise DatasetError(f"No files for group '{group_name}'.")

    if subset_file and subset_file != "__overall__":
        if subset_file in files or (DATA_DIR / subset_file).is_file():
            return analyze_dataset(subset_file)
        raise DatasetError(f"Subset file '{subset_file}' not found.")

    # Overall view
    if group.get("joinable", False) and len(files) > 1:
        overall_file = group.get("overall_file")
        if overall_file and (DATA_DIR / overall_file).is_file():
            return analyze_dataset(overall_file)

        joined_df = get_group_dataframe(group_name, subset_file="__overall__")
        if joined_df is not None:
            tag = f"{group_name.lower().replace(' ', '-')}-overall.csv"
            return _run_dataframe(tag, joined_df)

    return analyze_dataset(files[0])


def _is_numeric_like(val):
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        try:
            float(val.strip())
            return True
        except ValueError:
            return False
    return False


def _clean_dataframe_headers(df):
    # If all columns look like floats or numbers, file was headerless
    if all(_is_numeric_like(c) for c in df.columns):
        if df.shape[1] == 1:
            df.columns = ["value"]
        else:
            df.columns = [f"feature_{i+1:02d}" for i in range(df.shape[1])]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Drop trailing unnamed all-NaN columns if any
    cols_to_drop = [c for c in df.columns if str(c).lower().startswith("unnamed") and df[c].isna().all()]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def _read_dataframe(filename, content):
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            "Unsupported file type. Upload a .csv, .xlsx, .xls, or .json file."
        )
    if not content:
        raise DatasetError("The file is empty.")

    try:
        if extension in {".csv", ".tsv"}:
            df = None
            encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
            delimiters = [",", ";", "\t", None]

            for enc in encodings:
                for sep in delimiters:
                    try:
                        temp_df = pd.read_csv(
                            io.BytesIO(content),
                            sep=sep,
                            encoding=enc,
                            engine="python" if sep is None else "c",
                        )
                        if temp_df.shape[1] > 1:
                            df = temp_df
                            break
                        if df is None:
                            df = temp_df
                    except Exception:
                        continue
                if df is not None and df.shape[1] > 1:
                    break

            if df is None:
                df = pd.read_csv(io.BytesIO(content), encoding="latin-1")

            dataframe = _clean_dataframe_headers(df)

        elif extension in {".xlsx", ".xls"}:
            engine = "openpyxl" if extension == ".xlsx" else "xlrd"
            df = pd.read_excel(io.BytesIO(content), engine=engine)
            dataframe = _clean_dataframe_headers(df)
        else:
            dataframe = pd.read_json(io.BytesIO(content))
    except UnicodeDecodeError as exc:
        raise DatasetError("The file encoding could not be read.") from exc
    except Exception as exc:
        raise DatasetError(
            f"The file could not be read: {exc}"
        ) from exc

    if dataframe.empty or len(dataframe.columns) == 0:
        raise DatasetError("The dataset contains no rows or columns.")
    return dataframe


def get_dataframe(dataset_name=None, filename=None, content=None):
    if content is not None and filename is not None:
        return _read_dataframe(filename, content)
    if dataset_name is not None:
        path = DATA_DIR / dataset_name
        if not path.is_file():
            return None
        return _read_dataframe(path.name, path.read_bytes())
    return None


def _plain_python_result(result):
    try:
        if isinstance(result, str):
            parsed = json.loads(result)
        else:
            parsed = json.loads(
                json.dumps(result, ensure_ascii=False, allow_nan=False)
            )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetError("The analysis returned missing or malformed JSON.") from exc

    if not isinstance(parsed, (dict, list)):
        raise DatasetError("The analysis returned missing or malformed JSON.")
    return parsed


def _run_tp(path):
    try:
        result = tp.analyze_dataset(str(path))
    except Exception as exc:
        raise DatasetError("The dataset could not be analysed.") from exc
    return _plain_python_result(result)


def _run_dataframe(filename, dataframe):
    csv_name = f"{Path(filename).stem}.csv"
    try:
        with tempfile.TemporaryDirectory(prefix="dataset-analysis-") as temp_dir:
            temporary_path = Path(temp_dir) / csv_name
            dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
            return _run_tp(temporary_path)
    except OSError as exc:
        raise DatasetError("The temporary dataset could not be prepared.") from exc


def analyze_dataset(dataset_name):
    if dataset_name not in list_datasets():
        raise DatasetError("The selected dataset is not available.")

    path = DATA_DIR / dataset_name
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DatasetError("The selected dataset could not be read.") from exc

    dataframe = _read_dataframe(path.name, content)
    return _run_dataframe(path.name, dataframe)


def analyze_uploaded_file(filename, content):
    safe_filename = Path(filename).name
    dataframe = _read_dataframe(safe_filename, content)
    return _run_dataframe(safe_filename, dataframe)
