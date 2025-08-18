"""
Configuration file for file paths used throughout the application.
"""

from pathlib import Path

# Get the project root directory (3 levels up from this config file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directory paths
DATA_DIR = (PROJECT_ROOT / "data").resolve()
PREPROCESSING_DATA_DIR = (DATA_DIR / "preprocessing").resolve()
VISUALIZATION_DATA_DIR = (DATA_DIR / "visualizations").resolve()
PI_GROUPS_VISUALIZATION_DATA_DIR = (VISUALIZATION_DATA_DIR / "pi_groups").resolve()
REPORTS_DATA_DIR = (DATA_DIR / "reports").resolve()

# Specific file paths
PREPROCESSING_ERRORS_LOG_FILE = PREPROCESSING_DATA_DIR / "preprocessing_errors.log"

# Ensure data directories exist when imported
DATA_DIR.mkdir(exist_ok=True)
PREPROCESSING_DATA_DIR.mkdir(exist_ok=True)
VISUALIZATION_DATA_DIR.mkdir(exist_ok=True)
PI_GROUPS_VISUALIZATION_DATA_DIR.mkdir(exist_ok=True)
REPORTS_DATA_DIR.mkdir(exist_ok=True)
