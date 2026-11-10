"""Training pipeline: compare models, tune the best one, persist artefacts.

Usage::

    python -m src.train            # full pipeline
    python -m src.train --fast     # fewer search iterations (CI / smoke test)

Steps
-----
1. Load + clean the data, stratified 80/20 split.
2. 5-fold cross-validation of five candidate models (training set only).
3. Automatic selection of the best model on cross-validated ROC AUC.
4. Hyper-parameter optimisation of the winner with ``RandomizedSearchCV``.
5. Final evaluation of every model on the untouched test set.
6. Persistence: model, preprocessor, metrics, ROC curves, SHAP values and
   the monitoring baseline.
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    N_SEARCH_ITER,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    ROC_PATH,
    SHAP_PATH,
    ensure_directories,
    get_logger,
)
from src.data_loader import load_raw_data
from src.evaluate import (
    build_comparison_table,
    compute_confusion,
    compute_metrics,
    compute_roc_points,
    find_best_threshold,
    save_json,
    select_best_model,
)
from src.explain import build_explainer, compute_shap_values
from src.monitoring import build_baseline, save_baseline
from src.preprocessing import build_preprocessor, numeric_display_frame, split_data

logger = get_logger(__name__)

SHAP_SAMPLE_SIZE: int = 500
SHAP_BACKGROUND_SIZE: int = 200


# --------------------------------------------------------------------------- #
# Candidate models and search spaces
# --------------------------------------------------------------------------- #
def get_models() -> dict[str, ClassifierMixin]:
    """Return the five candidate classifiers with sensible defaults."""
    # Imported here so that a missing optional wheel gives a clear error
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, n_jobs=1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=1,
            verbose=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def get_param_distributions() -> dict[str, dict[str, Any]]:
    """Search space of each model for ``RandomizedSearchCV``.

    Keys are prefixed with ``model__`` because the search runs on the full
    ``Pipeline(preprocessor, model)``: preprocessing is re-fitted inside every
    fold, so no statistic of the validation fold leaks into training.
    """
    return {
        "Logistic Regression": {
            "model__C": loguniform(1e-3, 1e2),
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": randint(200, 600),
            "model__max_depth": [4, 6, 8, 10, None],
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "XGBoost": {
            "model__n_estimators": randint(150, 600),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 7),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 8),
            "model__reg_lambda": loguniform(0.1, 10),
        },
        "LightGBM": {
            "model__n_estimators": randint(150, 600),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__num_leaves": randint(4, 32),
            "model__min_child_samples": randint(10, 60),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(0.1, 10),
        },
        "Gradient Boosting": {
            "model__n_estimators": randint(100, 500),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 5),
            "model__subsample": uniform(0.6, 0.4),
            "model__min_samples_leaf": randint(1, 30),
        },
    }


def make_pipeline(model: ClassifierMixin) -> Pipeline:
    """Chain a fresh preprocessor with ``model``."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def compare_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[dict[str, dict[str, Any]], dict[str, Pipeline]]:
    """Cross-validate, fit and evaluate every candidate model.

    Returns:
        ``(results, fitted_pipelines)`` keyed by model name.
    """
    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}

    for name, model in get_models().items():
        logger.info("Training %s ...", name)
        pipeline = make_pipeline(model)
        try:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1
            )
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
        except Exception:  # one broken model must not sink the whole comparison
            logger.exception("Model %s failed and is skipped", name)
            continue

        proba = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "test": compute_metrics(y_test, proba),
            "confusion_matrix": compute_confusion(y_test, proba),
            "roc": compute_roc_points(y_test, proba),
            "cv_roc_auc": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "fit_seconds": round(fit_seconds, 3),
            "tuned": False,
        }
        fitted[name] = pipeline
        logger.info(
            "%-20s CV AUC %.4f (+/- %.4f) | test AUC %.4f | F1 %.4f",
            name,
            cv_scores.mean(),
            cv_scores.std(),
            results[name]["test"]["roc_auc"],
            results[name]["test"]["f1"],
        )

    if not results:
        raise RuntimeError("Every candidate model failed to train")
    return results, fitted


def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: StratifiedKFold,
    n_iter: int,
) -> RandomizedSearchCV:
    """Optimise the hyper-parameters of ``name`` with a randomised search."""
    logger.info("Tuning %s with RandomizedSearchCV (%d iterations)", name, n_iter)
    search = RandomizedSearchCV(
        estimator=make_pipeline(clone(get_models()[name])),
        param_distributions=get_param_distributions()[name],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info("Best CV AUC after tuning: %.4f | params: %s", search.best_score_, search.best_params_)
    return search


def save_shap_artifacts(
    pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    """Pre-compute SHAP values on a test sample for the Explainability page.

    Computing them once at training time keeps the web app instantaneous.
    """
    preprocessor: Pipeline = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    background = preprocessor.transform(
        X_train.sample(min(SHAP_BACKGROUND_SIZE, len(X_train)), random_state=RANDOM_STATE)
    )
    raw_sample = X_test.sample(min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=RANDOM_STATE)
    transformed = preprocessor.transform(raw_sample)

    explainer = build_explainer(model, background)
    explanation = compute_shap_values(
        explainer, transformed, numeric_display_frame(preprocessor, transformed)
    )
    joblib.dump(
        {
            "values": explanation.values,
            "base_values": explanation.base_values,
            "data": explanation.data,
            "display_data": explanation.display_data,
            "feature_names": explanation.feature_names,
            "columns": list(transformed.columns),
            "raw_sample": raw_sample.reset_index(drop=True),
            "background": background,
            "probabilities": model.predict_proba(transformed)[:, 1],
        },
        SHAP_PATH,
    )
    logger.info("SHAP artefacts saved to %s", SHAP_PATH)


def run_training(n_iter: int = N_SEARCH_ITER) -> dict[str, Any]:
    """Run the end-to-end training pipeline.

    Args:
        n_iter: Number of ``RandomizedSearchCV`` iterations.

    Returns:
        The metadata dict that is also saved to ``reports/metadata.json``.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_directories()
    started = time.perf_counter()

    X_train, X_test, y_train, y_test = split_data(load_raw_data())
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 1-2. Compare the candidates ------------------------------------------
    results, fitted = compare_models(X_train, y_train, X_test, y_test, cv)
    best_name = select_best_model(build_comparison_table(results))

    # 3. Tune the winner ----------------------------------------------------
    baseline_result = dict(results[best_name])
    best_pipeline = fitted[best_name]
    best_params: dict[str, Any] = {}
    try:
        search = tune_model(best_name, X_train, y_train, cv, n_iter)
        # Keep the tuned model only if cross-validation says it is better
        if search.best_score_ >= baseline_result["cv_roc_auc"]:
            best_pipeline = search.best_estimator_
            best_params = {
                k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                for k, v in search.best_params_.items()
            }
            proba = best_pipeline.predict_proba(X_test)[:, 1]
            std = float(search.cv_results_["std_test_score"][search.best_index_])
            results[best_name] = {
                "test": compute_metrics(y_test, proba),
                "confusion_matrix": compute_confusion(y_test, proba),
                "roc": compute_roc_points(y_test, proba),
                "cv_roc_auc": float(search.best_score_),
                "cv_roc_auc_std": std,
                "fit_seconds": baseline_result["fit_seconds"],
                "tuned": True,
            }
        else:
            logger.info("Tuning did not improve on the defaults; keeping the baseline")
    except Exception:
        logger.exception("Hyper-parameter search failed; keeping the baseline model")

    # 3b. Business decision threshold (from out-of-fold predictions) ---------
    oof_proba = cross_val_predict(
        clone(best_pipeline), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = find_best_threshold(y_train, oof_proba)
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    threshold_metrics = compute_metrics(y_test, test_proba, threshold)
    logger.info(
        "Decision threshold %.2f -> test recall %.3f, precision %.3f, F1 %.3f",
        threshold,
        threshold_metrics["recall"],
        threshold_metrics["precision"],
        threshold_metrics["f1"],
    )

    # 4. Persist model + preprocessor ----------------------------------------
    joblib.dump(best_pipeline.named_steps["model"], MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["preprocessor"], PREPROCESSOR_PATH)
    logger.info("Saved %s and %s", MODEL_PATH.name, PREPROCESSOR_PATH.name)

    # 5. Reports -------------------------------------------------------------
    save_json({n: {k: v for k, v in r.items() if k != "roc"} for n, r in results.items()}, METRICS_PATH)
    save_json({n: r["roc"] for n, r in results.items()}, ROC_PATH)

    train_scores = best_pipeline.predict_proba(X_train)[:, 1]
    save_baseline(build_baseline(X_train, train_scores))

    try:
        save_shap_artifacts(best_pipeline, X_train, X_test)
    except Exception:  # explainability is valuable but not blocking
        logger.exception("SHAP computation failed; the app will run without it")

    metadata: dict[str, Any] = {
        "best_model": best_name,
        "selection_metric": "cv_roc_auc",
        "tuned": results[best_name]["tuned"],
        "best_params": best_params,
        "baseline_metrics": baseline_result["test"],
        "baseline_cv_roc_auc": baseline_result["cv_roc_auc"],
        "final_metrics": results[best_name]["test"],
        "final_cv_roc_auc": results[best_name]["cv_roc_auc"],
        "decision_threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "threshold_confusion_matrix": compute_confusion(y_test, test_proba, threshold),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": float(np.mean(np.concatenate([y_train, y_test]))),
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "training_seconds": round(time.perf_counter() - started, 1),
    }
    save_json(metadata, METADATA_PATH)

    table = build_comparison_table(results)
    logger.info("Model comparison (test set):\n%s", table.round(4).to_string())
    logger.info("Training pipeline finished in %.1fs", metadata["training_seconds"])
    return metadata


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the churn prediction models")
    parser.add_argument("--fast", action="store_true", help="only 5 search iterations")
    parser.add_argument("--n-iter", type=int, default=N_SEARCH_ITER, help="search iterations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(n_iter=5 if args.fast else args.n_iter)
