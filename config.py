"""
Main configuration file for Crime Hotspot Simulation project.

This file contains project-wide settings, constants, and configuration
parameters for the Roysambu ward crime analysis system.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# GEOGRAPHIC SETTINGS
# ============================================================================

# Roysambu Ward boundaries (approximate)
ROYSAMBU_BOUNDS = {
    'min_lat': -1.2200,
    'max_lat': -1.2000, 
    'min_lon': 36.8900,
    'max_lon': 36.9100
}

# Coordinate reference systems
CRS_WGS84 = "EPSG:4326"  # WGS 84 (lat/lon)
CRS_UTM_37S = "EPSG:32737"  # UTM Zone 37S (meters)

# Grid settings for spatial analysis
DEFAULT_GRID_SIZE = 0.005  # ~500m grid cells in decimal degrees
FINE_GRID_SIZE = 0.002     # ~200m grid cells
COARSE_GRID_SIZE = 0.01    # ~1km grid cells

# ============================================================================
# DATA PATHS
# ============================================================================

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RISK_LAYERS_DIR = DATA_DIR / "risk_layers"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Output directories
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RISK_LAYERS_DIR,
                 MODELS_DIR, SAVED_MODELS_DIR, VISUALIZATIONS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================

# Agent-based model parameters
SIMULATION_CONFIG = {
    'max_steps': 8760,  # 1 year in hours
    'time_step': 1,     # 1 hour per step
    'agents': {
        'criminals': 50,
        'guardians': 20,
        'victims': 200
    },
    'environment': {
        'grid_size': DEFAULT_GRID_SIZE,
        'update_frequency': 24  # Update environment every 24 steps (1 day)
    }
}

# Crime types and their relative frequencies
CRIME_TYPES = {
    'theft': 0.35,
    'robbery': 0.20,
    'burglary': 0.15,
    'assault': 0.15,
    'vehicle_crime': 0.10,
    'fraud': 0.05
}

# Temporal patterns (hour of day multipliers)
HOURLY_CRIME_MULTIPLIERS = {
    0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.5,
    6: 0.7, 7: 1.0, 8: 1.2, 9: 1.1, 10: 1.0, 11: 1.1,
    12: 1.3, 13: 1.2, 14: 1.1, 15: 1.2, 16: 1.3, 17: 1.5,
    18: 1.7, 19: 1.8, 20: 1.6, 21: 1.4, 22: 1.2, 23: 0.8
}

# ============================================================================
# MACHINE LEARNING SETTINGS
# ============================================================================

# Model parameters
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'models': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'class_weight': 'balanced'
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'logistic': {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    }
}

# Feature engineering settings
FEATURE_CONFIG = {
    'spatial_features': True,
    'temporal_features': True,
    'contextual_features': True,
    'distance_features': True,
    'density_features': True
}

# Hotspot classification threshold
HOTSPOT_THRESHOLD = 3  # Minimum crimes per grid cell to classify as hotspot

# ============================================================================
# RISK TERRAIN MODELING SETTINGS
# ============================================================================

# Risk factors and their weights
RISK_FACTORS = {
    'bars_entertainment': {
        'weight': 0.25,
        'max_distance': 500,  # meters
        'decay_function': 'exponential'
    },
    'atms': {
        'weight': 0.20,
        'max_distance': 200,
        'decay_function': 'linear'
    },
    'schools': {
        'weight': -0.15,  # Negative weight (protective factor)
        'max_distance': 300,
        'decay_function': 'exponential'
    },
    'police_stations': {
        'weight': -0.20,
        'max_distance': 1000,
        'decay_function': 'exponential'
    },
    'public_transport': {
        'weight': 0.15,
        'max_distance': 100,
        'decay_function': 'linear'
    },
    'commercial_areas': {
        'weight': 0.10,
        'max_distance': 200,
        'decay_function': 'linear'
    }
}

# ============================================================================
# CLUSTERING SETTINGS
# ============================================================================

# DBSCAN parameters
DBSCAN_CONFIG = {
    'eps': 0.01,           # ~1km in decimal degrees
    'min_samples': 5,
    'metric': 'euclidean'
}

# K-means parameters
KMEANS_CONFIG = {
    'n_clusters': 8,
    'random_state': 42,
    'n_init': 10
}

# Hierarchical clustering parameters
HIERARCHICAL_CONFIG = {
    'n_clusters': 8,
    'linkage': 'ward'
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Map settings
MAP_CONFIG = {
    'default_zoom': 13,
    'tiles': 'OpenStreetMap',
    'center_lat': (ROYSAMBU_BOUNDS['min_lat'] + ROYSAMBU_BOUNDS['max_lat']) / 2,
    'center_lon': (ROYSAMBU_BOUNDS['min_lon'] + ROYSAMBU_BOUNDS['max_lon']) / 2
}

# Color schemes
COLOR_SCHEMES = {
    'crime_types': {
        'theft': '#FF6B6B',
        'robbery': '#4ECDC4',
        'burglary': '#45B7D1',
        'assault': '#FFA07A',
        'vehicle_crime': '#98D8C8',
        'fraud': '#F7DC6F'
    },
    'risk_levels': {
        'low': '#2ECC71',
        'medium': '#F39C12',
        'high': '#E74C3C',
        'very_high': '#8B0000'
    }
}

# ============================================================================
# API AND EXTERNAL SERVICE SETTINGS
# ============================================================================

# External data sources
EXTERNAL_APIS = {
    'openstreetmap': {
        'base_url': 'https://overpass-api.de/api/interpreter',
        'timeout': 60
    },
    'kenya_open_data': {
        'base_url': 'https://www.opendata.go.ke/api/views/',
        'timeout': 30
    }
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'crime_analysis.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================

# Streamlit dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': 'Roysambu Crime Hotspot Analysis',
    'page_icon': '🚔',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'max_file_size': 200,  # MB
    'supported_formats': ['csv', 'json', 'geojson', 'xlsx']
}

# ============================================================================
# CONSTANTS
# ============================================================================

# Time constants
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365

# Distance constants (in meters)
METERS_PER_DEGREE_LAT = 111000  # Approximate
METERS_PER_DEGREE_LON = 111000 * 0.7071  # At equator, adjusted for latitude

# Statistical constants
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_data_path(filename: str, data_type: str = 'raw') -> Path:
    """Get full path for data file."""
    if data_type == 'raw':
        return RAW_DATA_DIR / filename
    elif data_type == 'processed':
        return PROCESSED_DATA_DIR / filename
    elif data_type == 'risk':
        return RISK_LAYERS_DIR / filename
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def get_model_path(filename: str) -> Path:
    """Get full path for model file."""
    return SAVED_MODELS_DIR / filename

def get_output_path(filename: str, output_type: str = 'results') -> Path:
    """Get full path for output file."""
    if output_type == 'results':
        return RESULTS_DIR / filename
    elif output_type == 'visualizations':
        return VISUALIZATIONS_DIR / filename
    else:
        raise ValueError(f"Unknown output type: {output_type}")

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate if coordinates are within Roysambu bounds."""
    return (ROYSAMBU_BOUNDS['min_lat'] <= lat <= ROYSAMBU_BOUNDS['max_lat'] and
            ROYSAMBU_BOUNDS['min_lon'] <= lon <= ROYSAMBU_BOUNDS['max_lon'])

# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "0.1.0"
__author__ = "Crime Hotspot Analysis Team"
__description__ = "Crime Hotspot Simulation and Mapping for Roysambu Ward"
