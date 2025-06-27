# Crime Hotspot Mapping

A data-driven system for predicting and mapping crime hotspots using machine learning and geospatial analysis tools. This project combines statistical modeling, geospatial data processing, and interactive visualization to identify areas with high crime probability.

## 🚀 Features

- **Predictive Modeling**: Machine learning algorithms to predict crime hotspots
- **Geospatial Analysis**: Advanced spatial analysis using GeoPandas and spatial statistics
- **Interactive Visualizations**: Dynamic maps and charts using Folium and Plotly
- **Data Processing Pipeline**: Automated ETL processes for crime data
- **Risk Assessment**: Quantitative risk scoring for geographic areas
- **Temporal Analysis**: Time-series analysis for crime pattern identification

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Models](#models)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Code-Cadet/crime-hotspot-mapping.git
   cd crime-hotspot-mapping
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Prepare your data**
   - Place your crime dataset in the `data/raw/` directory
   - Ensure the data includes coordinates (latitude/longitude) and timestamps
   - Supported formats: CSV, JSON, GeoJSON

2. **Run the analysis**
   ```bash
   python src/main.py --data data/raw/crime_data.csv
   ```

3. **View results**
   - Generated maps will be saved in `visualizations/`
   - Model outputs will be in `models/`
   - Processed data will be in `data/processed/`

## 📁 Project Structure

```
crime-hotspot-mapping/
├── app/                    # Web application (Flask/Streamlit)
│   ├── templates/         # HTML templates
│   ├── static/           # CSS, JS, images
│   └── app.py            # Main application file
├── data/                  # Data storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned and processed data
│   └── external/         # External datasets (demographics, etc.)
├── models/               # Trained models and model artifacts
│   ├── saved_models/     # Serialized model files
│   ├── metrics/          # Model performance metrics
│   └── configs/          # Model configuration files
├── notebooks/            # Jupyter notebooks for analysis
│   ├── exploratory/      # Exploratory data analysis
│   ├── modeling/         # Model development
│   └── visualization/    # Visualization experiments
├── src/                  # Source code
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering
│   ├── models/           # Model classes and training
│   ├── visualization/    # Plotting and mapping functions
│   └── utils/            # Utility functions
├── visualizations/       # Generated plots and maps
├── requirements.txt      # Python dependencies
├── LICENSE              # MIT License
└── README.md           # This file
```

## 📊 Usage

### Data Processing

```python
from src.data.preprocessing import CrimeDataProcessor

# Initialize processor
processor = CrimeDataProcessor()

# Load and clean data
df = processor.load_data('data/raw/crime_data.csv')
df_clean = processor.clean_data(df)
df_features = processor.create_features(df_clean)
```

### Model Training

```python
from src.models.hotspot_predictor import HotspotPredictor

# Initialize and train model
model = HotspotPredictor(model_type='xgboost')
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
hotspot_areas = model.identify_hotspots(predictions, threshold=0.7)
```

### Visualization

```python
from src.visualization.mapping import HotspotMapper

# Create interactive map
mapper = HotspotMapper()
crime_map = mapper.create_hotspot_map(
    predictions=predictions,
    coordinates=coordinates,
    save_path='visualizations/crime_hotspots.html'
)
```
## 🤖 Models

The project supports multiple machine learning algorithms:

- **XGBoost**: Gradient boosting for high-accuracy predictions
- **Random Forest**: Ensemble method for robust predictions
- **Logistic Regression**: Baseline linear model
- **Support Vector Machine**: For complex decision boundaries
- **Neural Networks**: Deep learning approaches for complex patterns

### Model Performance Metrics

- **Precision**: Accuracy of hotspot predictions
- **Recall**: Coverage of actual crime incidents
- **F1-Score**: Balanced performance measure
- **AUC-ROC**: Overall model discrimination ability
- **Spatial Accuracy**: Geographic precision of predictions

## 🗺️ Visualization

### Interactive Maps
- **Heatmaps**: Density-based crime visualization
- **Choropleth Maps**: Administrative boundary analysis
- **Point Maps**: Individual incident mapping
- **Time-series Maps**: Temporal crime evolution

### Statistical Plots
- **Crime Trends**: Time-series analysis
- **Spatial Autocorrelation**: Moran's I and LISA statistics
- **Feature Importance**: Model interpretation plots
- **Performance Metrics**: ROC curves and confusion matrices

## 🔧 Configuration

Create a `config.yaml` file to customize settings:

```yaml
data:
  input_path: "data/raw/"
  output_path: "data/processed/"
  
model:
  algorithm: "xgboost"
  test_size: 0.2
  random_state: 42
  
visualization:
  map_style: "OpenStreetMap"
  color_scheme: "viridis"
  output_format: "html"
```

## 🧪 Testing

Run the test suite:

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 Documentation

Generate documentation:

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Open crime data provided by local law enforcement agencies
- Geospatial analysis tools from the Python ecosystem
- Machine learning frameworks: scikit-learn, XGBoost
- Visualization libraries: Folium, Plotly, Matplotlib

## 📞 Contact

- **Author**: Benja Kivaa
- **Project**: [https://github.com/Code-Cadet/crime-hotspot-mapping](https://github.com/Code-Cadet/crime-hotspot-mapping)
- **Issues**: [GitHub Issues](https://github.com/Code-Cadet/crime-hotspot-mapping/issues)

---

**Disclaimer**: This tool is for research and analysis purposes only. Crime prediction models should be used responsibly and in conjunction with domain expertise from law enforcement professionals.
