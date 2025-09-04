"""
Configuration file for file paths used throughout the application.
"""

from pathlib import Path
import os
import tempfile

# Get the project root directory (3 levels up from this config file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directory paths
DATA_DIR = (PROJECT_ROOT / "data").resolve()
PREPROCESSING_DATA_DIR = (DATA_DIR / "preprocessing").resolve()
VISUALIZATION_DATA_DIR = (DATA_DIR / "visualizations").resolve()
JOBS_VISUALIZATION_DATA_DIR = (VISUALIZATION_DATA_DIR / "jobs").resolve()
USERS_VISUALIZATION_DATA_DIR = (VISUALIZATION_DATA_DIR / "users").resolve()
PI_GROUPS_VISUALIZATION_DATA_DIR = (VISUALIZATION_DATA_DIR / "pi_groups").resolve()
REPORTS_DATA_DIR = (DATA_DIR / "reports").resolve()

# Specific file paths
_default_preprocess_log = PREPROCESSING_DATA_DIR / "preprocessing_errors.log"

# If running in test environment, redirect preprocessing errors log to a temporary file.
if os.getenv("RUN_ENV") == "TEST":
    # NamedTemporaryFile is not used as a context manager to keep file accessible during tests.
    _tmp_dir = tempfile.mkdtemp(prefix="preprocess_logs_")
    PREPROCESSING_ERRORS_LOG_FILE = Path(_tmp_dir) / "preprocessing_errors.log"
else:
    PREPROCESSING_ERRORS_LOG_FILE = _default_preprocess_log

# Ensure data directories exist when imported
DATA_DIR.mkdir(exist_ok=True)
PREPROCESSING_DATA_DIR.mkdir(exist_ok=True)
VISUALIZATION_DATA_DIR.mkdir(exist_ok=True)
JOBS_VISUALIZATION_DATA_DIR.mkdir(exist_ok=True)
USERS_VISUALIZATION_DATA_DIR.mkdir(exist_ok=True)
PI_GROUPS_VISUALIZATION_DATA_DIR.mkdir(exist_ok=True)
REPORTS_DATA_DIR.mkdir(exist_ok=True)
