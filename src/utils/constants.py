"""Constants for EcoPredict"""

# Data paths
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
EXTERNAL_DATA_PATH = "data/external/"

# Model paths
MODELS_PATH = "models/trained/"
METRICS_PATH = "models/metrics/"

# Output paths
OUTPUTS_PATH = "outputs/"
MAPS_PATH = "outputs/maps/"
REPORTS_PATH = "outputs/reports/"

# File extensions
SUPPORTED_DATA_FORMATS = ['.csv', '.xlsx', '.json', '.geojson', '.shp']
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.tiff', '.geotiff']

# Model parameters
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 1.0
}

# Geographic constants
DEFAULT_CRS = 'EPSG:4326'  # WGS84
INDIA_BOUNDS = {
    'min_lat': 6.0,
    'max_lat': 37.0,
    'min_lon': 68.0,
    'max_lon': 98.0
}

# API constants
API_VERSION = "v1"
MAX_PREDICTION_POINTS = 1000
DEFAULT_TIMEOUT = 30

# Dashboard constants
DEFAULT_MAP_CENTER = [20.5937, 78.9629]  # India center
DEFAULT_MAP_ZOOM = 5