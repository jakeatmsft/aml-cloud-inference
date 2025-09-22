# %% [markdown]
# # Load the MLFlow model locally and try predictions
#
# (Original explanatory markdown retained)

# ================================
# Debug / Logging Setup
# ================================
import os
import sys
import time
import json
import logging
from pathlib import Path

DEBUG_ENABLED = os.getenv("DEBUG", "1").lower() in ("1", "true", "yes", "y")

LOG_LEVEL = logging.DEBUG if DEBUG_ENABLED else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("score")

logger.debug("Debug logging enabled.")
logger.info("Starting score script.")

# ================================
# Configuration / Paths
# ================================
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./model/mlflow-model"))
TEST_DATA_PATH = Path(
    os.getenv("TEST_DATA_PATH", "./data/test-mltable-folder/bank_marketing_test_data.csv")
)

logger.debug(f"Resolved MODEL_DIR={MODEL_DIR.resolve()}")
logger.debug(f"Resolved TEST_DATA_PATH={TEST_DATA_PATH.resolve()}")

if not MODEL_DIR.exists():
    logger.error(f"Model directory not found: {MODEL_DIR.resolve()}")
    # Provide directory listing hint
    try:
        cwd_listing = [p.name for p in Path(".").iterdir()]
        logger.debug(f"Current working directory: {Path('.').resolve()}")
        logger.debug(f"Top-level entries: {cwd_listing}")
    except Exception as e:
        logger.debug(f"Failed to list CWD: {e}")
    raise FileNotFoundError(f"Expected MLflow model directory at {MODEL_DIR}")

mlmodel_file = MODEL_DIR / "MLmodel"
if not mlmodel_file.exists():
    logger.warning(f"MLmodel file not found at {mlmodel_file} (model load may fail).")
else:
    logger.debug("Found MLmodel file; dumping first 25 lines for inspection.")
    try:
        with mlmodel_file.open("r", encoding="utf-8") as f:
            snippet = "".join(f.readlines()[:25])
        for line in snippet.splitlines():
            logger.debug(f"MLmodel> {line}")
    except Exception as e:
        logger.debug(f"Could not read MLmodel file: {e}")

# ================================
# Load Test Data
# ================================
import pandas as pd

start_time = time.perf_counter()
if not TEST_DATA_PATH.exists():
    logger.error(f"Test data file not found: {TEST_DATA_PATH}")
    raise FileNotFoundError(f"Expected test CSV at {TEST_DATA_PATH}")

logger.info(f"Loading test data from: {TEST_DATA_PATH}")
try:
    test_df = pd.read_csv(TEST_DATA_PATH)
except Exception as e:
    logger.exception(f"Failed to read CSV: {e}")
    raise

logger.debug(f"Loaded test_df shape: {test_df.shape}")
logger.debug(f"Columns: {list(test_df.columns)}")

TARGET_COL = "y"
if TARGET_COL in test_df.columns:
    y_actual = test_df.pop(TARGET_COL)
    logger.debug(f"Extracted target column '{TARGET_COL}' with shape {y_actual.shape}")
else:
    y_actual = None
    logger.warning(f"Target column '{TARGET_COL}' not found in test data; continuing without ground truth.")

if DEBUG_ENABLED:
    # Show a few rows & basic stats
    logger.debug("Head(3) of feature data:\n" + test_df.head(3).to_string())
    logger.debug("Feature dtypes:\n" + test_df.dtypes.to_string())
    if y_actual is not None:
        logger.debug(f"Target value counts (top 5):\n{y_actual.value_counts().head()}")

# ================================
# Load Model
# ================================
import mlflow.pyfunc
import mlflow.sklearn

model_load_start = time.perf_counter()
MODEL_LOAD_PATH = str(MODEL_DIR)  # mlflow expects a string path
logger.info(f"Loading MLflow model from: {MODEL_LOAD_PATH}")

try:
    model = mlflow.sklearn.load_model(MODEL_LOAD_PATH)
    flavor_used = "sklearn"
except Exception as primary_exc:
    logger.warning(f"Sklearn flavor load failed ({primary_exc}); attempting generic pyfunc.")
    try:
        model = mlflow.pyfunc.load_model(MODEL_LOAD_PATH)
        flavor_used = "pyfunc"
    except Exception as secondary_exc:
        logger.exception(
            f"Both sklearn and pyfunc model load attempts failed. "
            f"Primary: {primary_exc} | Secondary: {secondary_exc}"
        )
        raise

logger.info(f"Model loaded successfully using flavor: {flavor_used}")
logger.debug(f"Model object type: {type(model)}")
logger.debug(f"Model load time: {time.perf_counter() - model_load_start:.3f}s")

# ================================
# Run Predictions
# ================================
predict_start = time.perf_counter()
logger.info("Running predictions...")
try:
    y_preds = model.predict(test_df)
except Exception as e:
    logger.exception(f"Prediction failed: {e}")
    raise

predict_elapsed = time.perf_counter() - predict_start
logger.info(f"Predictions complete in {predict_elapsed:.3f}s")
logger.debug(f"Predictions type: {type(y_preds)}")

# Convert predictions to a consistent structure
import numpy as np

if isinstance(y_preds, (list, tuple)):
    y_preds_array = np.asarray(y_preds)
else:
    y_preds_array = y_preds if isinstance(y_preds, np.ndarray) else np.asarray(y_preds)

logger.debug(f"Predictions array shape: {y_preds_array.shape}, dtype: {y_preds_array.dtype}")

# Show sample predictions
SAMPLE_SHOW = min(10, y_preds_array.shape[0])
logger.info(f"Sample predictions (first {SAMPLE_SHOW}): {y_preds_array[:SAMPLE_SHOW]}")

# Classification diagnostics
if y_preds_array.ndim == 1 and y_preds_array.dtype != object:
    unique_count = len(np.unique(y_preds_array))
    logger.debug(f"Unique prediction values (count={unique_count}): {np.unique(y_preds_array)[:15]}")
elif y_preds_array.ndim > 1:
    logger.debug(f"Predictions appear multi-dimensional (shape={y_preds_array.shape}).")

# If ground truth available, compute a quick accuracy (only if discrete & shapes align)
if y_actual is not None and y_preds_array.shape[0] == len(y_actual):
    try:
        # Basic heuristic: treat as classification if <= 20 unique preds
        if len(np.unique(y_preds_array)) <= 20:
            accuracy = (y_preds_array == y_actual.to_numpy()).mean()
            logger.info(f"Quick accuracy estimate (not official metric): {accuracy:.4f}")
        else:
            logger.debug("Skipping quick accuracy (predictions look continuous / regression).")
    except Exception as e:
        logger.debug(f"Could not compute quick accuracy: {e}")

# ================================
# Structured Debug Artifact (optional)
# ================================
if DEBUG_ENABLED:
    artifact_payload = {
        "model_dir": str(MODEL_DIR.resolve()),
        "data_rows": int(test_df.shape[0]),
        "data_cols": int(test_df.shape[1]),
        "prediction_shape": list(y_preds_array.shape),
        "flavor_used": flavor_used,
        "elapsed": {
            "load_model_sec": round(time.perf_counter() - model_load_start, 4),
            "predict_sec": round(predict_elapsed, 4),
            "total_runtime_sec": round(time.perf_counter() - start_time, 4),
        },
    }
    try:
        with open("debug_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(artifact_payload, f, indent=2)
        logger.debug("Wrote debug_run_summary.json")
    except Exception as e:
        logger.debug(f"Failed to write debug summary: {e}")

logger.info("Score script completed successfully.")