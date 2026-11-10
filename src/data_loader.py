"""Dataset download and loading.

Run as a script to fetch the Telco Customer Churn dataset::

    python -m src.data_loader
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.config import (
    DATASET_URLS,
    ID_COLUMN,
    RAW_DATA_PATH,
    RAW_FEATURES,
    TARGET,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)

try:
    # Verify TLS certificates against the operating-system trust store. This
    # is what makes downloads work behind corporate proxies / antivirus HTTPS
    # inspection, without ever disabling certificate verification.
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # optional dependency: fall back to certifi's bundle
    pass


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be downloaded or is malformed."""


def _validate(df: pd.DataFrame) -> None:
    """Check that the dataframe has the expected Telco schema."""
    missing = set(RAW_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise DatasetError(f"Dataset is missing expected columns: {sorted(missing)}")
    if df.empty:
        raise DatasetError("Dataset is empty")


def download_dataset(destination: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    """Download the Telco Customer Churn CSV if it is not already on disk.

    Each mirror in ``DATASET_URLS`` is tried in turn; the file is only written
    once its schema has been validated.

    Args:
        destination: Where to store the CSV.
        force: Re-download even when the file already exists.

    Returns:
        Path of the local CSV file.

    Raises:
        DatasetError: If no mirror returned a valid dataset.
    """
    ensure_directories()
    if destination.exists() and not force:
        logger.info("Dataset already present at %s", destination)
        return destination

    errors: list[str] = []
    for url in DATASET_URLS:
        try:
            logger.info("Downloading dataset from %s", url)
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            tmp_path = destination.with_suffix(".tmp")
            tmp_path.write_bytes(response.content)
            _validate(pd.read_csv(tmp_path))
            tmp_path.replace(destination)
            logger.info("Dataset saved to %s", destination)
            return destination
        except (requests.RequestException, DatasetError, pd.errors.ParserError) as exc:
            logger.warning("Mirror failed (%s): %s", url, exc)
            errors.append(f"{url}: {exc}")

    raise DatasetError("All dataset mirrors failed:\n" + "\n".join(errors))


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset, downloading it first when necessary.

    Returns:
        The untouched dataframe (7,043 rows x 21 columns).
    """
    if not path.exists():
        download_dataset(path)
    df = pd.read_csv(path)
    _validate(df)
    logger.info("Loaded dataset: %d rows x %d columns", *df.shape)
    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(data.drop(columns=[ID_COLUMN]).head())
