"""
Global configuration for IndoBERT SHAP Sentiment Classification project.

All hyperparameters, paths, and constants are defined here.
No hardcoded values should appear in other modules.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

for _d in [OUTPUT_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_FILENAME: str = "Rating_labeled.csv"
DATASET_PATH: Path = DATA_DIR / DATASET_FILENAME
DATASET_DOI: str = "https://doi.org/10.17632/XXXXXXXX"  # Mendeley reference

# Column names
TEXT_COL: str = "content"
LABEL_COL: str = "label"           # numeric label (created during preprocessing)
APP_COL: str = "app"
RATING_COL: str = "score"
RAW_LABEL_COL: str = "labelScoreBase"  # original string label in the CSV

# Labels
LABEL_MAP: dict = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_NAMES: list = ["Negative", "Neutral", "Positive"]
NUM_LABELS: int = 3
# Reverse mapping: string label → integer
LABEL_STR_TO_INT: dict = {"Negative": 0, "Neutral": 1, "Positive": 2}

# ---------------------------------------------------------------------------
# Data Splitting
# ---------------------------------------------------------------------------
TRAIN_SPLIT: float = 0.8
VAL_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MIN_TEXT_LEN: int = 3            # minimum token count after cleaning

# ---------------------------------------------------------------------------
# TF-IDF / Baseline
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: tuple = (1, 2)
TFIDF_SUBLINEAR_TF: bool = True
RF_N_ESTIMATORS: int = 200
SVC_MAX_ITER: int = 5000
CV_N_SPLITS: int = 5

# ---------------------------------------------------------------------------
# IndoBERT Fine-tuning
# ---------------------------------------------------------------------------
MODEL_NAME: str = "indobenchmark/indobert-base-p1"
MODEL_NAME_ALT: str = "indolem/indobert-base-uncased"
MAX_LEN: int = 128
BATCH_SIZE: int = 32
EVAL_BATCH_SIZE: int = 64
NUM_EPOCHS: int = 5
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01
WARMUP_RATIO: float = 0.1
EARLY_STOPPING_PATIENCE: int = 2
BEST_MODEL_PATH: Path = MODELS_DIR / "indobert_best.pt"
METRICS_LOG_PATH: Path = LOGS_DIR / "training_metrics.csv"

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
SHAP_BATCH_SIZE: int = 16
SHAP_MAX_EVALS: int = 500        # max evaluations for shap.Explainer (partition)
SHAP_TOP_N_TOKENS: int = 20

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE: torch.device = get_device()

# ---------------------------------------------------------------------------
# Progress Bars
# ---------------------------------------------------------------------------
def _is_notebook() -> bool:
    """Detect if running inside a Jupyter/Colab notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False

IS_NOTEBOOK: bool = _is_notebook()

def get_tqdm():
    """Return the appropriate tqdm class for the current environment."""
    if IS_NOTEBOOK:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm

def get_tqdm_pandas():
    """Return tqdm.pandas for DataFrame.progress_apply."""
    if IS_NOTEBOOK:
        from tqdm.notebook import tqdm
        tqdm.pandas
        return tqdm.pandas
    else:
        from tqdm import tqdm
        return tqdm.pandas

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # HuggingFace transformers
    os.environ["TRANSFORMERS_SEED"] = str(seed)
