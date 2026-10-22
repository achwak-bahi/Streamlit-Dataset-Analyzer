"""EDA Engine (fichier unique) - Partie 1 : dataset -> JSON.

Usage :
  python eda_analysis.py --input data/weatherHistory.csv --target "Precip Type"
  python eda_analysis.py --input data --targets "weatherHistory=Precip Type;iris=species"
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

SCHEMA_VERSION = "1.0"
SEED = 42
SUPPORTED = (".csv", ".tsv", ".xlsx", ".xls")


# ============================ utils ============================
def clean(obj):
    """Convertit recursivement en types JSON valides (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    return obj


def r(x, nd=4):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


def numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def non_constant(df, cols):
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


def is_classification(series, max_classes=20):
    if not pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return series.nunique(dropna=True) <= max_classes


# ============================ loader ============================
def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Format non supporte : {ext}")


def list_datasets(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    return [os.path.join(input_path, n) for n in sorted(os.listdir(input_path))
            if n.lower().endswith(SUPPORTED)]


# ============================ 1. description ============================
def describe(df):
    num_cols = numeric_columns(df)
    cat_cols = [c for c in df.columns if c not in num_cols]
    columns = {}
    for c in df.columns:
        s = df[c]
        info = {"dtype": str(s.dtype),
                "type": "numeric" if c in num_cols else "categorical",
                "count": int(s.count()),
                "n_unique": int(s.nunique(dropna=True))}
        if c in num_cols:
            info.update({
                "min": r(s.min()), "max": r(s.max()), "range": r(s.max() - s.min()),
                "avg": r(s.mean()), "median": r(s.median()),
                "std": r(s.std()), "var": r(s.var()),
                "q1": r(s.quantile(0.25)), "q3": r(s.quantile(0.75)),
                "skewness": r(s.skew()), "kurtosis": r(s.kurt()),
            })
        else:
            top = s.value_counts().head(5)
            info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
        columns[c] = info

    cov = df[num_cols].cov() if num_cols else pd.DataFrame()
    return {
        "overview": {"n_rows": int(len(df)), "n_columns": int(df.shape[1]),
                     "numeric_columns": num_cols, "categorical_columns": cat_cols,
                     "memory_mb": r(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)},
        "statistics": columns,
        "covariance": {"labels": list(cov.columns),
                       "matrix": [[r(v) for v in row] for row in cov.values]},
    }


# ============================ 2/3/4. qualite ============================
def missing_data(df):
    n = len(df)
    per_col = {}
    for c in df.columns:
        m = int(df[c].isna().sum())
        per_col[c] = {"missing": m, "pct": r(m / n * 100, 2) if n else 0.0}
    return {"total_missing_cells": int(df.isna().sum().sum()),
            "rows_with_missing": int(df.isna().any(axis=1).sum()),
            "columns_with_missing": [c for c, v in per_col.items() if v["missing"] > 0],
            "per_column": per_col}


def noisy_data(df, rolling_window=24):
    num_cols = numeric_columns(df)
    constant = []
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 1:
            d = df[c].dropna()
            constant.append({"column": c, "value": None if d.empty else str(d.iloc[0])})

    suspicious_zeros = []
    for c in non_constant(df, num_cols):
        s = df[c]
        zeros = int((s == 0).sum())
        std, med = s.std(), s.median()
        if zeros > 0 and std > 0 and abs(med) > 3 * std:
            suspicious_zeros.append({"column": c, "zeros": zeros,
                                     "pct": r(zeros / len(df) * 100, 2),
                                     "median": r(med), "std": r(std)})

    rolling = []
    if rolling_window and rolling_window > 1:
        for c in non_constant(df, num_cols):
            s = df[c]
            noise = s - s.rolling(rolling_window, center=True).mean()
            ns = noise.std()
            rolling.append({"column": c, "noise_std": r(ns),
                            "ratio_to_std": r(ns / s.std()) if s.std() else None})
    return {"duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": constant,
            "suspicious_zeros": suspicious_zeros,
            "rolling_noise": {"window": rolling_window, "columns": rolling,
                              "note": "Pertinent seulement si les lignes sont ordonnees dans le temps."}}


def outliers(df, z_threshold=3.0, iqr_factor=1.5):
    n = len(df)
    out = {}
    for c in numeric_columns(df):
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        std = s.std()
        n_z = int((((s - s.mean()) / std).abs() > z_threshold).sum()) if std and std > 0 else 0
        out[c] = {"iqr": {"lower": r(lo), "upper": r(hi), "count": n_iqr, "pct": r(n_iqr / n * 100, 2)},
                  "zscore": {"threshold": z_threshold, "count": n_z, "pct": r(n_z / n * 100, 2)}}
    return {"method": {"iqr_factor": iqr_factor, "z_threshold": z_threshold}, "per_column": out}


# ============================ 5. balanced ============================
def _verdict(ratio):
    if ratio <= 1.5:
        return "balanced", "Aucune action necessaire."
    if ratio <= 3:
        return "slightly_imbalanced", "Stratified split ; class_weight optionnel."
    if ratio <= 5:
        return "imbalanced", "Stratified split + class_weight."
    if ratio <= 20:
        return "highly_imbalanced", "SMOTE (sur TRAIN uniquement) ou cost-sensitive learning."
    return "extremely_imbalanced", "ADASYN / ensembles / detection d'anomalies ; metriques F1, recall."


def balance(df, target):
    if not target:
        return {"applicable": False, "reason": "Aucune colonne cible fournie."}
    if target not in df.columns:
        return {"applicable": False, "reason": f"Colonne cible introuvable : {target}"}
    s = df[target]
    if not is_classification(s):
        return {"applicable": False, "target": target, "reason": "La cible semble continue (regression)."}
    vc = s.value_counts(dropna=True)
    total = int(vc.sum())
    ratio = float(vc.max() / vc.min()) if len(vc) > 1 else 1.0
    verdict, advice = _verdict(ratio)
    return {"applicable": True, "target": target, "n_classes": int(len(vc)),
            "missing_target": int(s.isna().sum()),
            "classes": [{"label": str(k), "count": int(v), "pct": r(v / total * 100, 2)} for k, v in vc.items()],
            "majority_class": str(vc.idxmax()), "minority_class": str(vc.idxmin()),
            "imbalance_ratio": r(ratio, 2), "is_balanced": verdict == "balanced",
            "verdict": verdict, "suggestion": advice,
            "note": "Ratio calcule sans les valeurs manquantes de la cible."}


# ============================ Viz ============================
def correlation(df, top_n=10):
    num_cols = numeric_columns(df)
    corr = df[num_cols].corr()
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append({"var1": a, "var2": b, "r": r(v), "abs_r": r(abs(v))})
    pairs.sort(key=lambda p: p["abs_r"], reverse=True)
    return {"method": "pearson", "labels": list(corr.columns),
            "matrix": [[r(v) for v in row] for row in corr.values],
            "top_pairs": pairs[:top_n]}


def feature_importance(df, target, max_rows=20000):
    if not target or target not in df.columns:
        return {"applicable": False, "reason": "Aucune colonne cible valide."}
    feats = [c for c in non_constant(df, numeric_columns(df)) if c != target]
    if not feats:
        return {"applicable": False, "reason": "Aucune feature numerique exploitable."}
    data = df[feats + [target]].dropna(subset=[target])
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=SEED)
    X = data[feats].fillna(data[feats].median())
    y = data[target]
    clf = is_classification(y)
    if clf:
        y_enc = pd.factorize(y)[0]
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight="balanced")
        mi = mutual_info_classif(X, y_enc, random_state=SEED)
    else:
        y_enc = y.values
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        mi = mutual_info_regression(X, y_enc, random_state=SEED)
    rf.fit(X, y_enc)
    mi_max = float(mi.max()) if len(mi) and mi.max() > 0 else 1.0
    rf_list = sorted(({"feature": f, "importance": r(v)} for f, v in zip(feats, rf.feature_importances_)),
                     key=lambda d: d["importance"], reverse=True)
    mi_list = sorted(({"feature": f, "score": r(v), "normalized": r(v / mi_max)} for f, v in zip(feats, mi)),
                     key=lambda d: d["score"], reverse=True)
    return {"applicable": True, "task": "classification" if clf else "regression",
            "target": target, "n_rows_used": int(len(data)),
            "random_forest": rf_list, "mutual_info": mi_list,
            "note": "Calcul exploratoire sur tout le dataset (pas une evaluation de modele)."}


def _kde_curve(values, grid):
    try:
        return gaussian_kde(values)(grid)
    except Exception:
        return None


def kde(df, target=None, n_points=100, max_rows=5000, max_classes=6):
    use_classes = (target in df.columns) if target else False
    clf = use_classes and is_classification(df[target])
    out = {}
    for c in non_constant(df, numeric_columns(df)):
        if c == target:
            continue
        s = df[c].dropna()
        grid = np.linspace(s.min(), s.max(), n_points)
        sample = s.sample(min(len(s), max_rows), random_state=SEED)
        dens = _kde_curve(sample.values, grid)
        if dens is None:
            continue
        entry = {"x": [r(v) for v in grid], "density": [r(v, 6) for v in dens]}
        if clf and df[target].nunique(dropna=True) <= max_classes:
            by_class = {}
            for cls, grp in df[[c, target]].dropna().groupby(target):
                v = grp[c]
                if len(v) >= 10 and v.nunique() > 1:
                    v = v.sample(min(len(v), 3000), random_state=SEED)
                    d = _kde_curve(v.values, grid)
                    if d is not None:
                        by_class[str(cls)] = [r(x, 6) for x in d]
            if by_class:
                entry["by_class"] = by_class
        out[c] = entry
    return {"n_points": n_points, "bandwidth": "scott", "columns": out}


# ============================ TP Visualizations ============================
def plot_feature_importance(df, target=None, num_cols=None, show=False, save_path=None):
    """Calculates and visualizes feature importance using RandomForestRegressor/Classifier.
    Automates target selection if not specified (defaults to 'Temperature (C)' if present).
    """
    if num_cols is None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if target is None:
        if "Temperature (C)" in df.columns:
            target = "Temperature (C)"
        elif len(num_cols) > 1:
            target = num_cols[-1]
        elif len(df.columns) > 1:
            target = df.columns[-1]
        else:
            return None, None

    if target not in df.columns:
        return None, None

    is_reg = pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20

    X_cols = [c for c in num_cols if c != target]
    if not X_cols:
        return None, None

    data = df[X_cols + [target]].dropna(subset=[target])
    if len(data) > 10000:
        data = data.sample(10000, random_state=SEED)

    X = data[X_cols].fillna(data[X_cols].median())
    y = data[target]

    if is_reg:
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(X, y)
    else:
        y_enc = pd.factorize(y)[0]
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(X, y_enc)

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", ax=ax, color="#2563eb")
    ax.set_title(f"Feature importance (RandomForest) — Target: {target}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig, imp


def plot_missing_data(df, show=False, save_path=None):
    """Visualizes missing values per column with seaborn barplot."""
    missing = df.isna().sum()
    if len(missing) > 15:
        cols_with_miss = missing[missing > 0]
        plot_missing = cols_with_miss.head(15) if len(cols_with_miss) > 0 else missing.head(15)
    else:
        plot_missing = missing
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(x=plot_missing.index, y=plot_missing.values, ax=ax, color="#2563eb")
    ax.set_xticks(range(len(plot_missing)))
    ax.set_xticklabels(plot_missing.index, rotation=45, ha="right", fontsize=8)
    ax.set_title("Missing values per column", fontsize=11, fontweight="bold")
    ax.set_ylabel("Missing Count", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig, missing


def plot_outliers(df, col=None, show=False, save_path=None):
    """Visualizes outliers using seaborn boxplot for a single column (dropdown) or 3x3 grid."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return None
    if col and col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(y=df[col], ax=ax, color="#38bdf8")
        ax.set_title(col, fontsize=11, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        return fig
    else:
        plot_cols = num_cols[:9]
        fig, axes = plt.subplots(3, 3, figsize=(8, 5))
        axes_flat = axes.flatten()
        for ax, c in zip(axes_flat, plot_cols):
            sns.boxplot(y=df[c], ax=ax, color="#38bdf8")
            ax.set_title(c, fontsize=8, fontweight="bold")
            ax.tick_params(labelsize=7)
        for ax in axes_flat[len(plot_cols):]:
            ax.set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        return fig


def plot_balance(df, columns=None, show=False, save_path=None):
    """Visualizes class balance for specified or detected categorical columns."""
    if columns is None:
        candidates = [c for c in ["Precip Type", "Summary"] if c in df.columns]
        if not candidates:
            candidates = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique() <= 20][:2]
        columns = candidates
    elif isinstance(columns, str):
        columns = [columns]

    figs = []
    for col in columns:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            df[col].value_counts().head(10).plot(kind="bar", ax=ax, color="#2563eb")
            ax.set_title(f"Balance — {col}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_{col}.png")
            if show:
                plt.show()
            figs.append((col, fig))
    return figs


# ============================ pipeline ============================
def analyze_dataset(path, target=None, rolling_window=24):
    df = load_dataset(path)
    name = os.path.splitext(os.path.basename(path))[0]
    result = {
        "schema_version": SCHEMA_VERSION,
        "meta": {"dataset_name": name, "file": os.path.basename(path),
                 "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
                 "target": target if target in df.columns else None},
        "data_description": describe(df),
        "missing_data": missing_data(df),
        "noisy_data": noisy_data(df, rolling_window),
        "outliers": outliers(df),
        "balanced": balance(df, target),
        "visualizations": {"correlation": correlation(df),
                           "feature_importance": feature_importance(df, target),
                           "kde": kde(df, target)},
    }
    return clean(result)


# ============================ CLI ============================
def parse_targets(text):
    mapping = {}
    if not text:
        return mapping
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            mapping[k.strip()] = v.strip()
    return mapping


def main():
    p = argparse.ArgumentParser(description="EDA -> JSON (partie 1 du projet)")
    p.add_argument("--input", required=True, help="Fichier dataset ou dossier de datasets")
    p.add_argument("--out-dir", default="results", help="Dossier de sortie des JSON")
    p.add_argument("--target", default=None, help="Colonne cible (cas d'un seul fichier)")
    p.add_argument("--targets", default=None, help='Cibles par dataset : "nom1=col1;nom2=col2"')
    p.add_argument("--rolling-window", type=int, default=24,
                   help="Fenetre de la moyenne mobile pour le bruit (0 = desactive)")
    args = p.parse_args()

    files = list_datasets(args.input)
    if not files:
        sys.exit("Aucun dataset trouve.")
    targets = parse_targets(args.targets)
    os.makedirs(args.out_dir, exist_ok=True)

    index = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        target = targets.get(name, args.target if len(files) == 1 else None)
        print(f"[...] {name} (target={target})")
        res = analyze_dataset(path, target=target, rolling_window=args.rolling_window)
        out_path = os.path.join(args.out_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        index.append({"dataset": name, "json": os.path.basename(out_path),
                      "n_rows": res["meta"]["n_rows"], "n_columns": res["meta"]["n_columns"],
                      "target": res["meta"]["target"]})
        print(f"[OK]  -> {out_path}")

    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": index}, f, ensure_ascii=False, indent=2)
    print(f"[OK]  index.json ({len(index)} dataset(s))")


if __name__ == "__main__":
    main()
