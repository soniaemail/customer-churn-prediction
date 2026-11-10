"""Core package of the Customer Churn Prediction project.

Modules
-------
config         Paths, constants and logging configuration.
data_loader    Automatic dataset download + loading.
preprocessing  Cleaning, feature engineering and the ColumnTransformer pipeline.
train          Model training, comparison, tuning and persistence.
evaluate       Metrics, ROC curves and comparison tables.
explain        SHAP-based explainability helpers.
monitoring     Prediction logging and data-drift (PSI) monitoring.
"""

__version__ = "1.0.0"
