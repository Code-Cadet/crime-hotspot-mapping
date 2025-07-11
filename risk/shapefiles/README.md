# Shapefiles for Roysambu Ward

This directory contains GIS shapefiles for spatial analysis.

## Contents

- `roysambu_boundary.shp`: Ward boundary shapefile
- `streets.shp`: Street network data
- `buildings.shp`: Building footprints
- `facilities.shp`: Points of interest (schools, hospitals, etc.)
- `administrative.shp`: Administrative boundaries
- `land_use.shp`: Land use classifications

## Data Sources

- Kenya National Bureau of Statistics
- Nairobi County GIS data
- OpenStreetMap extracts
- Survey of Kenya mapping data

## Coordinate System

All shapefiles should use:
- **Projection**: WGS 84 / UTM Zone 37S
- **EPSG Code**: 32737
- **Units**: Meters

## Usage Notes

1. Ensure all shapefiles have complete component files (.shp, .shx, .dbf, .prj)
2. Use consistent coordinate reference systems
3. Validate geometry before analysis
4. Document data sources and collection dates
