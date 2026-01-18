"""Constants used throughout the EcoPredict system"""

# File paths
CONFIG_DIR = "config"
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

# Data subdirectories
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
EXTERNAL_DATA_DIR = f"{DATA_DIR}/external"

# Model subdirectories
TRAINED_MODELS_DIR = f"{MODELS_DIR}/trained"
MODEL_METRICS_DIR = f"{MODELS_DIR}/metrics"

# File extensions
MODEL_EXTENSION = ".pkl"
CONFIG_EXTENSION = ".yaml"
DATA_EXTENSION = ".csv"

# Model types
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
LINEAR_REGRESSION = "linear_regression"

# Risk levels
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

# Default parameters
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Dashboard settings
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501

# Coordinate system
DEFAULT_CRS = "EPSG:4326"  # WGS84

# Feature columns
CLIMATE_FEATURES = [
    'temperature', 'precipitation', 'humidity', 'wind_speed'
]

LANDUSE_FEATURES = [
    'forest_cover', 'agricultural_area', 'urban_area', 'water_bodies'
]

SPECIES_FEATURES = [
    'species_count', 'endemic_species', 'threatened_species'
]

# All features combined
ALL_FEATURES = CLIMATE_FEATURES + LANDUSE_FEATURES + SPECIES_FEATURES