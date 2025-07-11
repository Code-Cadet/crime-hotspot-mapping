# Placeholder for raw crime data files

This directory should contain:
- Crime incident data files (CSV, JSON, GeoJSON)
- Police report data
- Incident location data
- Temporal crime data

## Sample Data Format

Crime data should include the following columns:
- `incident_id`: Unique identifier for each incident
- `latitude`: Latitude coordinate (decimal degrees)
- `longitude`: Longitude coordinate (decimal degrees) 
- `datetime`: Date and time of incident (ISO format)
- `crime_type`: Type of crime (theft, robbery, burglary, assault, etc.)
- `description`: Brief description of the incident
- `location_description`: Description of the location
- `district`: Police district or ward
- `status`: Case status (open, closed, etc.)

## Data Sources

Potential data sources for Roysambu ward:
- Kenya Police Service crime reports
- National Crime Research Centre data
- Open data portals
- Simulation-generated synthetic data
