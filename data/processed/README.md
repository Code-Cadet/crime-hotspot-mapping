# Processed Data Directory

This directory contains cleaned and processed crime data files.

## Contents

- `cleaned_crime_data.csv`: Main cleaned crime dataset
- `spatial_features.csv`: Crime data with spatial features
- `temporal_features.csv`: Crime data with temporal features  
- `grid_aggregated.csv`: Crime counts aggregated to spatial grid
- `simulation_results/`: Results from crime simulations
- `model_features/`: Feature matrices for ML models

## Data Processing Pipeline

1. **Data Validation**: Check data quality and completeness
2. **Cleaning**: Remove duplicates, invalid coordinates, bad dates
3. **Feature Engineering**: Create spatial and temporal features
4. **Aggregation**: Aggregate to spatial grids for analysis
5. **Export**: Save in multiple formats (CSV, Parquet, GeoJSON)
