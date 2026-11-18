"""Model training and inference for the recipe traffic predictor.

Reproduces the selected model from notebooks/recipe_traffic_prediction.ipynb:
the same cleaning steps, a tuned Logistic Regression pipeline and a business
decision threshold chosen on out-of-fold training predictions so that
precision on the "High" class reaches at least 80%.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import sklearn
from django.conf import settings
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_PRECISION = 0.80
NUTR_COLS = ["calories", "carbohydrate", "sugar", "protein"]
FEATURES = NUTR_COLS + ["servings", "category"]
CATEGORIES = ["Lunch/Snacks", "Beverages", "Potato", "Vegetable", "Meat",
              "Chicken", "Pork", "Dessert", "Breakfast", "One Dish Meal"]
CATEGORY_ALIASES = {"Chicken Breast": "Chicken"}


class InputError(ValueError):
    """Raised when uploaded recipe data cannot be scored."""


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def normalize_category(series):
    return series.astype(str).str.strip().replace(CATEGORY_ALIASES)


def parse_servings(series):
    """'4 as a snack' -> 4. Values without leading digits become NaN."""
    return pd.to_numeric(series.astype(str).str.extract(r"^\s*(\d+)")[0], errors="coerce")


def clean_training_data(df):
    """Cleaning exactly as in the notebook (section: Data Validation)."""
    df = df.copy()
    df["servings"] = parse_servings(df["servings"]).astype("int64")
    df["category"] = normalize_category(df["category"])
    for col in NUTR_COLS:
        df[col] = df.groupby("category")[col].transform(lambda s: s.fillna(s.median()))
    df["high_traffic"] = df["high_traffic"].fillna("Low")
    return df


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_pipeline():
    prep = ColumnTransformer([
        ("nutr", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("scale", StandardScaler())]), NUTR_COLS),
        ("serv", StandardScaler(), ["servings"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])


def threshold_for_precision(y_true, proba, target=TARGET_PRECISION):
    """Lowest threshold whose precision meets the target (maximises recall)."""
    prec, _, thr = precision_recall_curve(y_true, proba)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5


def _scores(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "flagged": int(pred.sum()),
        "false_positives": int(((pred == 1) & (y_true == 0)).sum()),
    }


def train(data_path=None):
    """Fit the model and return a serialisable bundle with model and metrics."""
    raw = pd.read_csv(data_path or settings.RECIPE_DATA_PATH, dtype={"recipe": str})
    df = clean_training_data(raw)

    X = df[FEATURES].copy()
    y = df["high_traffic"].eq("High").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(build_pipeline(), {"model__C": [0.01, 0.1, 0.3, 1, 3, 10]},
                      cv=cv, scoring="roc_auc").fit(X_train, y_train)
    model = gs.best_estimator_

    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = threshold_for_precision(y_train, oof)

    proba_test = model.predict_proba(X_test)[:, 1]
    top30 = np.argsort(-proba_test)[:30]

    # Coefficients, for the "what drives traffic" chart
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    coefficients = sorted(
        ({"feature": n.split("__", 1)[1].replace("category_", ""), "coef": float(c)}
         for n, c in zip(names, coefs)),
        key=lambda d: d["coef"], reverse=True)

    cat_stats = (df.assign(is_high=y)
                 .groupby("category")["is_high"].agg(recipes="size", high_rate="mean")
                 .reindex(CATEGORIES).reset_index()
                 .sort_values("high_rate", ascending=False))

    metrics = {
        "n_recipes": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_C": gs.best_params_["model__C"],
        "cv_roc_auc": float(gs.best_score_),
        "test_roc_auc": float(roc_auc_score(y_test, proba_test)),
        "baseline_precision": float(y_test.mean()),
        "default": _scores(y_test, proba_test, 0.5),
        "business": _scores(y_test, proba_test, threshold),
        "top30_hit_rate": float(y_test.to_numpy()[top30].mean()),
    }

    return {
        "model": model,
        "threshold": threshold,
        "metrics": metrics,
        "coefficients": coefficients,
        "category_stats": cat_stats.to_dict("records"),
        "category_medians": df.groupby("category")[NUTR_COLS].median().to_dict("index"),
        "sklearn_version": sklearn.__version__,
    }


def save(bundle, path=None):
    path = path or settings.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


@lru_cache(maxsize=1)
def get_bundle():
    """Load the trained bundle, training it first if missing or stale."""
    path = settings.MODEL_PATH
    if path.exists():
        bundle = joblib.load(path)
        if bundle.get("sklearn_version") == sklearn.__version__:
            return bundle
    bundle = train()
    save(bundle)
    return bundle


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def prepare_input(df, category_medians):
    """Validate and clean user-supplied recipes. Raises InputError on bad data."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in ("category", "servings") if c not in df.columns]
    if missing:
        raise InputError(f"Missing required column(s): {', '.join(missing)}.")
    if df.empty:
        raise InputError("No recipes to score.")

    df["category"] = normalize_category(df["category"])
    unknown = sorted(set(df["category"]) - set(CATEGORIES))
    if unknown:
        raise InputError(f"Unknown category: {', '.join(unknown)}. "
                         f"Allowed: {', '.join(CATEGORIES)}.")

    df["servings"] = parse_servings(df["servings"])
    if df["servings"].isna().any():
        rows = (df.index[df["servings"].isna()] + 1).tolist()[:5]
        raise InputError(f"Servings must start with a number (row {', '.join(map(str, rows))}).")

    for col in NUTR_COLS:
        if col not in df.columns:
            df[col] = np.nan
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.isna() & df[col].notna() & df[col].astype(str).str.strip().ne("")
        if bad.any():
            raise InputError(f"Column '{col}' must be numeric.")
        if (values < 0).any():
            raise InputError(f"Column '{col}' cannot be negative.")
        # Same rule as training: impute with the category median
        df[col] = values.fillna(df["category"].map(lambda c: category_medians[c][col]))
    return df


def predict(df):
    """Score recipes. Returns the input with probability and prediction columns."""
    bundle = get_bundle()
    clean = prepare_input(df, bundle["category_medians"])
    proba = bundle["model"].predict_proba(clean[FEATURES])[:, 1]
    clean["probability_high"] = proba.round(4)
    clean["prediction"] = np.where(proba >= bundle["threshold"], "High", "Low")
    return clean


def predict_one(**recipe):
    row = predict(pd.DataFrame([recipe])).iloc[0]
    return {"probability_high": float(row["probability_high"]),
            "prediction": row["prediction"],
            "threshold": get_bundle()["threshold"],
            "imputed": [c for c in NUTR_COLS if recipe.get(c) is None]}
