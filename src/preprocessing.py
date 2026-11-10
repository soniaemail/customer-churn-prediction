"""Data cleaning, feature engineering and preprocessing pipeline.

The preprocessing is a single scikit-learn ``Pipeline``::

    FeatureEngineer  ->  ColumnTransformer
                           |- numeric : median imputation + StandardScaler
                           |- binary  : label (ordinal) encoding
                           |- multi   : one-hot encoding

Because everything lives in one fitted object (``models/preprocessor.pkl``),
the exact same transformations are applied at training and inference time,
which rules out training/serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    ID_COLUMN,
    MULTI_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_FEATURES,
    TARGET,
    TEST_SIZE,
    get_logger,
)

logger = get_logger(__name__)

# Add-on services counted by the ``NumServices`` engineered feature
SERVICE_COLUMNS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Features created by ``FeatureEngineer``
ENGINEERED_NUMERIC: list[str] = ["AvgMonthlySpend", "ChargeDelta", "NumServices"]
ENGINEERED_CATEGORICAL: list[str] = ["TenureGroup"]

TENURE_BINS: list[float] = [-np.inf, 12, 24, 48, np.inf]
TENURE_LABELS: list[str] = ["0-1 year", "1-2 years", "2-4 years", "4+ years"]


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quality issues of the raw Telco dataset.

    * ``TotalCharges`` is read as text because 11 brand-new customers
      (tenure = 0) have a blank value -> converted to float (NaN for blanks,
      handled later by the pipeline).
    * ``SeniorCitizen`` is 0/1 while every other binary column is Yes/No ->
      harmonised to Yes/No.
    * The identifier column carries no signal -> dropped.

    Args:
        df: Raw dataframe.

    Returns:
        A cleaned copy; the input is left untouched.
    """
    df = df.copy()
    # NB: rows are NOT de-duplicated once the id is gone - two distinct
    # customers can legitimately share the same profile.
    df = df.drop(columns=[ID_COLUMN], errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    for column in BINARY_FEATURES + MULTI_FEATURES:
        # plain object dtype keeps scikit-learn imputers/encoders happy
        df[column] = (
            df[column].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
        )

    n_missing = int(df["TotalCharges"].isna().sum())
    if n_missing:
        logger.info("TotalCharges: %d missing values (new customers)", n_missing)
    return df


def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Label-encode the target (``No`` -> 0, ``Yes`` -> 1).

    Returns:
        The encoded array and the fitted ``LabelEncoder``.
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str))
    return encoded, encoder


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create business-driven features from the raw columns.

    New features
    ------------
    AvgMonthlySpend  Historical average bill (TotalCharges / tenure).
    ChargeDelta      Current bill minus historical average: a positive value
                     means the customer now pays more than they used to.
    NumServices      Number of subscribed services (0-8): a proxy for how
                     deeply the customer is embedded in the product.
    TenureGroup      Tenure bucketed into life-cycle stages.

    The transformer is stateless (``fit`` learns nothing) so it can never leak
    information from the validation folds.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> FeatureEngineer:
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(RAW_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")

        X = X[RAW_FEATURES].copy()

        # Missing TotalCharges only happens for tenure == 0 -> tenure * monthly
        X["TotalCharges"] = X["TotalCharges"].fillna(X["tenure"] * X["MonthlyCharges"])

        X["AvgMonthlySpend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
        X["ChargeDelta"] = X["MonthlyCharges"] - X["AvgMonthlySpend"]

        n_addons = (X[SERVICE_COLUMNS] == "Yes").sum(axis=1)
        has_phone = (X["PhoneService"] == "Yes").astype(int)
        has_internet = (X["InternetService"] != "No").astype(int)
        X["NumServices"] = (n_addons + has_phone + has_internet).astype(float)

        X["TenureGroup"] = pd.cut(X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(
            object
        )
        return X


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_preprocessor() -> Pipeline:
    """Build the (unfitted) preprocessing pipeline.

    Returns:
        ``Pipeline([FeatureEngineer, ColumnTransformer])`` returning a pandas
        DataFrame so that downstream models and SHAP keep feature names.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Label encoding of two-level columns (No -> 0, Yes -> 1, Female -> 0 ...)
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "label",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    # One-hot encoding of nominal columns; unseen categories become all-zeros
    multi_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    columns = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES + ENGINEERED_NUMERIC),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("cat", multi_pipeline, MULTI_FEATURES + ENGINEERED_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline(steps=[("features", FeatureEngineer()), ("columns", columns)])


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Clean the data and create a stratified train/test split.

    Args:
        df: Raw dataframe including the target column.

    Returns:
        ``X_train, X_test, y_train, y_test`` (targets are 0/1 arrays).
    """
    cleaned = clean_data(df)
    y, _ = encode_target(cleaned[TARGET])
    X = cleaned[RAW_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Split: %d train / %d test rows (churn rate %.1f%% / %.1f%%)",
        len(X_train),
        len(X_test),
        100 * y_train.mean(),
        100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def numeric_display_frame(preprocessor: Pipeline, transformed: pd.DataFrame) -> pd.DataFrame:
    """Undo the scaling of numeric columns for human-readable displays.

    SHAP plots show the feature value next to each bar; a scaled value such as
    ``tenure = -1.28`` means nothing to a business user, so we map it back to
    months / dollars.
    """
    display = transformed.copy()
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_NUMERIC
    scaler: StandardScaler = (
        preprocessor.named_steps["columns"].named_transformers_["num"].named_steps["scaler"]
    )
    display[numeric_cols] = np.round(scaler.inverse_transform(transformed[numeric_cols]), 2)
    return display
