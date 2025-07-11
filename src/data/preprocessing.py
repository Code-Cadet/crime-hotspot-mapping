"""
Data preprocessing and ETL pipeline for Roysambu crime data.

This module handles data cleaning, validation, transformation, and
feature engineering for crime hotspot analysis.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
import logging
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CrimeDataProcessor:
    """
    Comprehensive data processing pipeline for crime data.
    
    Handles data cleaning, validation, geocoding, temporal processing,
    and feature engineering for Roysambu ward crime analysis.
    """
    
    def __init__(self, roysambu_bounds: Tuple[float, float, float, float] = None):
        """
        Initialize data processor.
        
        Args:
            roysambu_bounds: (min_lat, min_lon, max_lat, max_lon) for Roysambu ward
        """
        self.roysambu_bounds = roysambu_bounds or (-1.2200, 36.8900, -1.2000, 36.9100)
        self.processed_data = None
        self.validation_report = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: str, file_type: str = 'auto') -> pd.DataFrame:
        """
        Load crime data from various file formats.
        
        Args:
            file_path: Path to the data file
            file_type: File type ('csv', 'json', 'geojson', 'excel', 'auto')
            
        Returns:
            Raw DataFrame
        """
        if file_type == 'auto':
            file_type = file_path.split('.')[-1].lower()
            
        try:
            if file_type == 'csv':
                data = pd.read_csv(file_path)
            elif file_type == 'json':
                data = pd.read_json(file_path)
            elif file_type == 'geojson':
                data = gpd.read_file(file_path)
                # Convert to regular DataFrame if needed
                if isinstance(data, gpd.GeoDataFrame):
                    # Extract coordinates from geometry
                    data['longitude'] = data.geometry.x
                    data['latitude'] = data.geometry.y
                    data = pd.DataFrame(data.drop(columns='geometry'))
            elif file_type in ['xlsx', 'xls', 'excel']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            self.logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate crime data quality and completeness.
        
        Args:
            data: Raw crime data
            
        Returns:
            Validation report dictionary
        """
        report = {
            'total_records': len(data),
            'missing_values': {},
            'invalid_coordinates': 0,
            'outside_roysambu': 0,
            'invalid_dates': 0,
            'duplicate_records': 0,
            'data_quality_score': 0.0
        }
        
        # Check for missing values
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                report['missing_values'][column] = missing_count
                
        # Validate coordinates
        if 'latitude' in data.columns and 'longitude' in data.columns:
            # Check for invalid coordinates
            invalid_coords = (
                (data['latitude'] < -90) | (data['latitude'] > 90) |
                (data['longitude'] < -180) | (data['longitude'] > 180) |
                data['latitude'].isnull() | data['longitude'].isnull()
            )
            report['invalid_coordinates'] = invalid_coords.sum()
            
            # Check if coordinates are within Roysambu bounds
            if not invalid_coords.all():
                valid_coords = ~invalid_coords
                outside_bounds = (
                    (data.loc[valid_coords, 'latitude'] < self.roysambu_bounds[0]) |
                    (data.loc[valid_coords, 'latitude'] > self.roysambu_bounds[2]) |
                    (data.loc[valid_coords, 'longitude'] < self.roysambu_bounds[1]) |
                    (data.loc[valid_coords, 'longitude'] > self.roysambu_bounds[3])
                )
                report['outside_roysambu'] = outside_bounds.sum()
                
        # Validate dates
        date_columns = [col for col in data.columns 
                       if 'date' in col.lower() or 'time' in col.lower()]
        for date_col in date_columns:
            try:
                pd.to_datetime(data[date_col], errors='coerce')
                invalid_dates = pd.to_datetime(data[date_col], errors='coerce').isnull()
                report['invalid_dates'] += invalid_dates.sum()
            except:
                continue
                
        # Check for duplicates
        report['duplicate_records'] = data.duplicated().sum()
        
        # Calculate data quality score
        total_issues = (
            sum(report['missing_values'].values()) +
            report['invalid_coordinates'] +
            report['outside_roysambu'] +
            report['invalid_dates'] +
            report['duplicate_records']
        )
        report['data_quality_score'] = max(0, 1 - (total_issues / len(data)))
        
        self.validation_report = report
        return report
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess crime data.
        
        Args:
            data: Raw crime data
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Remove duplicates
        cleaned_data = cleaned_data.drop_duplicates()
        
        # Clean coordinate data
        if 'latitude' in cleaned_data.columns and 'longitude' in cleaned_data.columns:
            # Remove invalid coordinates
            valid_coords = (
                (cleaned_data['latitude'] >= -90) & (cleaned_data['latitude'] <= 90) &
                (cleaned_data['longitude'] >= -180) & (cleaned_data['longitude'] <= 180) &
                cleaned_data['latitude'].notnull() & cleaned_data['longitude'].notnull()
            )
            cleaned_data = cleaned_data[valid_coords]
            
            # Filter to Roysambu bounds (with small buffer)
            buffer = 0.01  # ~1km buffer
            in_bounds = (
                (cleaned_data['latitude'] >= self.roysambu_bounds[0] - buffer) &
                (cleaned_data['latitude'] <= self.roysambu_bounds[2] + buffer) &
                (cleaned_data['longitude'] >= self.roysambu_bounds[1] - buffer) &
                (cleaned_data['longitude'] <= self.roysambu_bounds[3] + buffer)
            )
            cleaned_data = cleaned_data[in_bounds]
            
        # Clean temporal data
        cleaned_data = self._clean_temporal_data(cleaned_data)
        
        # Clean categorical data
        cleaned_data = self._clean_categorical_data(cleaned_data)
        
        # Clean text fields
        cleaned_data = self._clean_text_fields(cleaned_data)
        
        self.logger.info(f"Cleaned data: {len(data)} -> {len(cleaned_data)} records")
        return cleaned_data
        
    def _clean_temporal_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean temporal/datetime columns."""
        cleaned_data = data.copy()
        
        # Find datetime columns
        datetime_columns = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                datetime_columns.append(col)
                
        for col in datetime_columns:
            try:
                # Convert to datetime
                cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce')
                
                # Remove invalid dates
                cleaned_data = cleaned_data[cleaned_data[col].notnull()]
                
                # Filter reasonable date ranges (e.g., not future dates)
                max_date = datetime.now() + timedelta(days=1)
                min_date = datetime(2000, 1, 1)  # Reasonable historical limit
                
                valid_dates = (
                    (cleaned_data[col] >= min_date) & 
                    (cleaned_data[col] <= max_date)
                )
                cleaned_data = cleaned_data[valid_dates]
                
            except Exception as e:
                self.logger.warning(f"Could not clean datetime column {col}: {str(e)}")
                
        return cleaned_data
        
    def _clean_categorical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical columns like crime types."""
        cleaned_data = data.copy()
        
        # Crime type standardization
        if 'crime_type' in cleaned_data.columns:
            # Convert to lowercase and strip whitespace
            cleaned_data['crime_type'] = (
                cleaned_data['crime_type']
                .astype(str)
                .str.lower()
                .str.strip()
            )
            
            # Standardize common crime types
            crime_mapping = {
                'theft': ['theft', 'stealing', 'larceny', 'shoplifting'],
                'robbery': ['robbery', 'mugging', 'armed robbery'],
                'burglary': ['burglary', 'breaking and entering', 'housebreaking'],
                'assault': ['assault', 'battery', 'violence', 'attack'],
                'vehicle_crime': ['vehicle theft', 'car theft', 'carjacking', 'vandalism'],
                'fraud': ['fraud', 'scam', 'embezzlement', 'forgery']
            }
            
            for standard_type, variants in crime_mapping.items():
                mask = cleaned_data['crime_type'].str.contains('|'.join(variants), na=False)
                cleaned_data.loc[mask, 'crime_type'] = standard_type
                
        return cleaned_data
        
    def _clean_text_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean text fields and descriptions."""
        cleaned_data = data.copy()
        
        text_columns = cleaned_data.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            if col not in ['crime_type']:  # Skip already processed columns
                # Basic text cleaning
                cleaned_data[col] = (
                    cleaned_data[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r'\s+', ' ', regex=True)  # Multiple spaces to single
                    .replace('nan', np.nan)
                )
                
        return cleaned_data
        
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime columns.
        
        Args:
            data: Cleaned crime data
            
        Returns:
            DataFrame with additional temporal features
        """
        enhanced_data = data.copy()
        
        # Find the main datetime column
        datetime_col = None
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
                datetime_col = col
                break
                
        if datetime_col and datetime_col in enhanced_data.columns:
            dt = pd.to_datetime(enhanced_data[datetime_col])
            
            # Basic temporal features
            enhanced_data['year'] = dt.dt.year
            enhanced_data['month'] = dt.dt.month
            enhanced_data['day'] = dt.dt.day
            enhanced_data['hour'] = dt.dt.hour
            enhanced_data['day_of_week'] = dt.dt.dayofweek
            enhanced_data['week_of_year'] = dt.dt.isocalendar().week
            
            # Boolean features
            enhanced_data['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            enhanced_data['is_night'] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 5)).astype(int)
            enhanced_data['is_rush_hour'] = dt.dt.hour.isin([7, 8, 17, 18]).astype(int)
            
            # Seasonal features
            enhanced_data['season'] = (dt.dt.month % 12 // 3 + 1)
            
            # Cyclical encoding
            enhanced_data['hour_sin'] = np.sin(2 * np.pi * enhanced_data['hour'] / 24)
            enhanced_data['hour_cos'] = np.cos(2 * np.pi * enhanced_data['hour'] / 24)
            enhanced_data['day_sin'] = np.sin(2 * np.pi * enhanced_data['day_of_week'] / 7)
            enhanced_data['day_cos'] = np.cos(2 * np.pi * enhanced_data['day_of_week'] / 7)
            enhanced_data['month_sin'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
            enhanced_data['month_cos'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)
            
        return enhanced_data
        
    def create_spatial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create spatial features from coordinate data.
        
        Args:
            data: Crime data with coordinates
            
        Returns:
            DataFrame with spatial features
        """
        enhanced_data = data.copy()
        
        if 'latitude' in data.columns and 'longitude' in data.columns:
            # Grid-based features
            lat_bins = np.linspace(self.roysambu_bounds[0], self.roysambu_bounds[2], 20)
            lon_bins = np.linspace(self.roysambu_bounds[1], self.roysambu_bounds[3], 20)
            
            enhanced_data['lat_grid'] = pd.cut(enhanced_data['latitude'], lat_bins, labels=False)
            enhanced_data['lon_grid'] = pd.cut(enhanced_data['longitude'], lon_bins, labels=False)
            enhanced_data['grid_id'] = (
                enhanced_data['lat_grid'].astype(str) + '_' + 
                enhanced_data['lon_grid'].astype(str)
            )
            
            # Distance to center of Roysambu
            center_lat = (self.roysambu_bounds[0] + self.roysambu_bounds[2]) / 2
            center_lon = (self.roysambu_bounds[1] + self.roysambu_bounds[3]) / 2
            
            enhanced_data['distance_to_center'] = np.sqrt(
                (enhanced_data['latitude'] - center_lat) ** 2 +
                (enhanced_data['longitude'] - center_lon) ** 2
            )
            
        return enhanced_data
        
    def aggregate_to_grid(self, data: pd.DataFrame, grid_size: float = 0.005) -> pd.DataFrame:
        """
        Aggregate crime data to spatial grid cells.
        
        Args:
            data: Processed crime data
            grid_size: Size of grid cells in decimal degrees
            
        Returns:
            Aggregated DataFrame with grid cell statistics
        """
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            raise ValueError("Data must contain latitude and longitude columns")
            
        # Create grid
        lat_bins = np.arange(self.roysambu_bounds[0], self.roysambu_bounds[2] + grid_size, grid_size)
        lon_bins = np.arange(self.roysambu_bounds[1], self.roysambu_bounds[3] + grid_size, grid_size)
        
        # Assign grid coordinates
        data_copy = data.copy()
        data_copy['grid_lat'] = pd.cut(data_copy['latitude'], lat_bins, labels=False)
        data_copy['grid_lon'] = pd.cut(data_copy['longitude'], lon_bins, labels=False)
        
        # Aggregate by grid cell
        grid_agg = data_copy.groupby(['grid_lat', 'grid_lon']).agg({
            'latitude': ['count', 'mean'],
            'longitude': 'mean',
        }).reset_index()
        
        # Flatten column names
        grid_agg.columns = ['grid_lat', 'grid_lon', 'crime_count', 'center_lat', 'center_lon']
        
        # Add grid cell identifiers
        grid_agg['grid_id'] = (
            grid_agg['grid_lat'].astype(str) + '_' + 
            grid_agg['grid_lon'].astype(str)
        )
        
        return grid_agg
        
    def export_processed_data(self, data: pd.DataFrame, output_path: str, 
                            file_format: str = 'csv') -> None:
        """
        Export processed data to file.
        
        Args:
            data: Processed data to export
            output_path: Output file path
            file_format: Output format ('csv', 'json', 'geojson', 'parquet')
        """
        try:
            if file_format == 'csv':
                data.to_csv(output_path, index=False)
            elif file_format == 'json':
                data.to_json(output_path, orient='records', indent=2)
            elif file_format == 'parquet':
                data.to_parquet(output_path, index=False)
            elif file_format == 'geojson':
                # Convert to GeoDataFrame for GeoJSON export
                if 'latitude' in data.columns and 'longitude' in data.columns:
                    from shapely.geometry import Point
                    geometry = [Point(lon, lat) for lat, lon in 
                              zip(data['latitude'], data['longitude'])]
                    gdf = gpd.GeoDataFrame(data, geometry=geometry)
                    gdf.to_file(output_path, driver='GeoJSON')
                else:
                    raise ValueError("Latitude and longitude required for GeoJSON export")
            else:
                raise ValueError(f"Unsupported output format: {file_format}")
                
            self.logger.info(f"Data exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            raise
            
    def process_pipeline(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Run complete data processing pipeline.
        
        Args:
            input_path: Path to raw data file
            output_path: Path for processed data output
            
        Returns:
            Fully processed DataFrame
        """
        self.logger.info("Starting data processing pipeline...")
        
        # Load data
        raw_data = self.load_data(input_path)
        
        # Validate data
        validation_report = self.validate_data(raw_data)
        self.logger.info(f"Data quality score: {validation_report['data_quality_score']:.2f}")
        
        # Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Create features
        enhanced_data = self.create_temporal_features(cleaned_data)
        enhanced_data = self.create_spatial_features(enhanced_data)
        
        # Export processed data
        self.export_processed_data(enhanced_data, output_path)
        
        self.processed_data = enhanced_data
        self.logger.info("Data processing pipeline completed!")
        
        return enhanced_data


def main():
    """Example usage of data preprocessing pipeline."""
    print("Crime data preprocessing module initialized")
    print("Features:")
    print("- Data validation and quality assessment")
    print("- Coordinate and temporal data cleaning")
    print("- Feature engineering (spatial and temporal)")
    print("- Grid aggregation for hotspot analysis")
    

if __name__ == "__main__":
    main()
