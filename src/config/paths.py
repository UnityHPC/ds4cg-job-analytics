"""
Configuration file for file paths used throughout the application.
"""

from pathlib import Path

# Get the project root directory (3 levels up from this config file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directory paths
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSING_DATA_DIR = DATA_DIR / "preprocessing"
VISUALIZATION_DATA_DIR = DATA_DIR / "visualizations"
REPORTS_DATA_DIR = DATA_DIR / "reports"

# Specific file paths
ERROR_SUMMARY_FILE = PREPROCESSING_DATA_DIR / "error_summary.txt"

# Ensure data directories exist when imported
DATA_DIR.mkdir(exist_ok=True)
PREPROCESSING_DATA_DIR.mkdir(exist_ok=True)
VISUALIZATION_DATA_DIR.mkdir(exist_ok=True)
REPORTS_DATA_DIR.mkdir(exist_ok=True)
