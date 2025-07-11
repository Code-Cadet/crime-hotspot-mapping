"""
Streamlit dashboard for Crime Hotspot Simulation and Mapping in Roysambu Ward.

This interactive dashboard provides visualization and analysis tools for
crime hotspot data, simulation results, and predictive models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import json
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import custom modules
try:
    from simulation.simulator import CrimeSimulator, SimulationConfig
    from clustering.cluster_analysis import CrimeClusterAnalyzer
    from prediction.model_train import CrimePredictor
    from risk.risk_model import RiskTerrainModel, RiskLayerGenerator
except ImportError as e:
    st.error(f"Error importing modules: {e}")


class CrimeDashboard:
    """
    Main dashboard class for crime hotspot analysis.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Roysambu Crime Hotspot Analysis",
            page_icon="🚔",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .sidebar-section {
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'crime_data' not in st.session_state:
            st.session_state.crime_data = None
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
            
    def render_sidebar(self):
        """Render sidebar navigation and controls."""
        st.sidebar.title("🚔 Crime Analysis Dashboard")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Home", "Data Overview", "Simulation", "Clustering Analysis", 
             "Risk Modeling", "Prediction Models", "Visualization"]
        )
        
        # Data upload section
        st.sidebar.markdown("### 📊 Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Crime Data",
            type=['csv', 'json', 'geojson'],
            help="Upload your crime dataset for analysis"
        )
        
        if uploaded_file:
            self.load_data(uploaded_file)
            
        # Quick stats if data is loaded
        if st.session_state.crime_data is not None:
            st.sidebar.markdown("### 📈 Quick Stats")
            data = st.session_state.crime_data
            st.sidebar.metric("Total Incidents", len(data))
            if 'crime_type' in data.columns:
                st.sidebar.metric("Crime Types", data['crime_type'].nunique())
            
        return page
        
    def load_data(self, uploaded_file):
        """Load and validate uploaded data."""
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.geojson'):
                data = gpd.read_file(uploaded_file)
            else:
                st.error("Unsupported file format")
                return
                
            # Basic validation
            required_columns = ['latitude', 'longitude']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Data must contain columns: {required_columns}")
                return
                
            st.session_state.crime_data = data
            st.sidebar.success("Data loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")
            
    def render_home_page(self):
        """Render home page."""
        st.markdown('<h1 class="main-header">Roysambu Ward Crime Hotspot Analysis</h1>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Simulation
            - Agent-based crime modeling
            - Environmental factor analysis
            - Temporal pattern simulation
            """)
            
        with col2:
            st.markdown("""
            ### 🔍 Analysis
            - Spatial clustering detection
            - Risk terrain modeling
            - Statistical pattern analysis
            """)
            
        with col3:
            st.markdown("""
            ### 🤖 Prediction
            - Machine learning models
            - Hotspot forecasting
            - Risk assessment
            """)
            
        # Display sample data if available
        if st.session_state.crime_data is not None:
            st.subheader("📊 Loaded Dataset Preview")
            st.dataframe(st.session_state.crime_data.head(10))
            
    def render_data_overview(self):
        """Render data overview page."""
        st.header("📊 Data Overview")
        
        if st.session_state.crime_data is None:
            st.warning("Please upload crime data to view overview.")
            return
            
        data = st.session_state.crime_data
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Incidents", len(data))
        with col2:
            if 'crime_type' in data.columns:
                st.metric("Crime Types", data['crime_type'].nunique())
        with col3:
            if 'datetime' in data.columns:
                date_range = pd.to_datetime(data['datetime'])
                st.metric("Date Range (Days)", 
                         (date_range.max() - date_range.min()).days)
        with col4:
            st.metric("Geographic Points", len(data))
            
        # Crime type distribution
        if 'crime_type' in data.columns:
            st.subheader("Crime Type Distribution")
            crime_counts = data['crime_type'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=crime_counts.index, y=crime_counts.values,
                           title="Crime Type Frequency")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.pie(values=crime_counts.values, names=crime_counts.index,
                           title="Crime Type Proportion")
                st.plotly_chart(fig, use_container_width=True)
                
        # Temporal analysis
        if 'datetime' in data.columns:
            st.subheader("Temporal Patterns")
            
            # Convert to datetime
            data['datetime'] = pd.to_datetime(data['datetime'])
            data['hour'] = data['datetime'].dt.hour
            data['day_of_week'] = data['datetime'].dt.day_name()
            
            col1, col2 = st.columns(2)
            with col1:
                hourly_crimes = data['hour'].value_counts().sort_index()
                fig = px.line(x=hourly_crimes.index, y=hourly_crimes.values,
                            title="Crimes by Hour of Day")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                daily_crimes = data['day_of_week'].value_counts()
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                            'Friday', 'Saturday', 'Sunday']
                daily_crimes = daily_crimes.reindex(days_order, fill_value=0)
                fig = px.bar(x=daily_crimes.index, y=daily_crimes.values,
                           title="Crimes by Day of Week")
                st.plotly_chart(fig, use_container_width=True)
                
        # Geographic distribution
        st.subheader("Geographic Distribution")
        self.render_crime_map(data)
        
    def render_simulation_page(self):
        """Render simulation page."""
        st.header("🎯 Crime Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Simulation Parameters")
            
            # Simulation configuration
            max_steps = st.number_input("Simulation Steps (Hours)", 
                                      min_value=24, max_value=8760, value=168)
            n_criminals = st.number_input("Number of Criminal Agents", 
                                        min_value=10, max_value=200, value=50)
            n_guardians = st.number_input("Number of Guardian Agents", 
                                        min_value=5, max_value=100, value=20)
            n_victims = st.number_input("Number of Potential Victims", 
                                      min_value=50, max_value=1000, value=200)
            
            if st.button("Run Simulation"):
                self.run_simulation(max_steps, n_criminals, n_guardians, n_victims)
                
        with col2:
            st.subheader("Simulation Results")
            
            if st.session_state.simulation_results:
                results = st.session_state.simulation_results
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Crimes", results.get('total_crimes', 0))
                with col2:
                    st.metric("Simulation Steps", results.get('total_steps', 0))
                with col3:
                    st.metric("Crime Rate/Hour", 
                             round(results.get('total_crimes', 0) / 
                                  max(results.get('total_steps', 1), 1), 3))
                    
                # Plot crime events over time
                if 'crime_events' in results and results['crime_events']:
                    crime_df = pd.DataFrame(results['crime_events'])
                    if 'time_step' in crime_df.columns:
                        crimes_by_time = crime_df['time_step'].value_counts().sort_index()
                        fig = px.line(x=crimes_by_time.index, y=crimes_by_time.values,
                                    title="Simulated Crimes Over Time")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run simulation to see results here.")
                
    def run_simulation(self, max_steps, n_criminals, n_guardians, n_victims):
        """Run crime simulation with given parameters."""
        with st.spinner("Running simulation..."):
            try:
                # Create simulation configuration
                config = {
                    'bounds': [-1.2200, 36.8900, -1.2000, 36.9100],  # Roysambu bounds
                    'max_steps': max_steps,
                    'agents': {
                        'criminals': n_criminals,
                        'guardians': n_guardians,
                        'victims': n_victims
                    }
                }
                
                # Initialize and run simulator
                simulator = CrimeSimulator(config)
                results = simulator.run_simulation(save_results=False)
                
                st.session_state.simulation_results = results
                st.success("Simulation completed successfully!")
                
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
                
    def render_clustering_page(self):
        """Render clustering analysis page."""
        st.header("🔍 Spatial Clustering Analysis")
        
        if st.session_state.crime_data is None:
            st.warning("Please upload crime data for clustering analysis.")
            return
            
        data = st.session_state.crime_data
        
        # Clustering parameters
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Clustering Parameters")
            
            method = st.selectbox("Clustering Method", 
                                ["DBSCAN", "K-Means", "Hierarchical"])
            
            if method == "DBSCAN":
                eps = st.slider("Epsilon (neighborhood size)", 0.001, 0.1, 0.01)
                min_samples = st.slider("Minimum samples", 3, 20, 5)
                
            elif method == "K-Means":
                n_clusters = st.slider("Number of clusters", 2, 20, 8)
                
            if st.button("Run Clustering"):
                self.run_clustering_analysis(data, method, eps if method == "DBSCAN" else None, 
                                           min_samples if method == "DBSCAN" else None,
                                           n_clusters if method == "K-Means" else None)
                
        with col2:
            st.subheader("Clustering Results")
            # TODO: Display clustering results and visualizations
            st.info("Clustering results will be displayed here.")
            
    def run_clustering_analysis(self, data, method, eps=None, min_samples=None, n_clusters=None):
        """Run clustering analysis on crime data."""
        try:
            # Create GeoDataFrame if needed
            if not isinstance(data, gpd.GeoDataFrame):
                from shapely.geometry import Point
                geometry = [Point(lon, lat) for lon, lat in 
                          zip(data['longitude'], data['latitude'])]
                gdf = gpd.GeoDataFrame(data, geometry=geometry)
            else:
                gdf = data
                
            # Run clustering
            analyzer = CrimeClusterAnalyzer(gdf)
            
            if method == "DBSCAN":
                labels = analyzer.dbscan_clustering(eps=eps, min_samples=min_samples)
            elif method == "K-Means":
                labels = analyzer.kmeans_clustering(n_clusters=n_clusters)
            elif method == "Hierarchical":
                labels = analyzer.hierarchical_clustering(n_clusters=n_clusters)
                
            st.success(f"{method} clustering completed!")
            
            # Display cluster summary
            summary = analyzer.get_cluster_summary()
            st.dataframe(summary)
            
        except Exception as e:
            st.error(f"Clustering error: {str(e)}")
            
    def render_crime_map(self, data):
        """Render interactive crime map."""
        if data is None or len(data) == 0:
            st.warning("No data available for mapping.")
            return
            
        # Calculate map center
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
        
        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Add crime points
        for idx, row in data.iterrows():
            if idx > 1000:  # Limit points for performance
                break
                
            popup_text = f"Crime Type: {row.get('crime_type', 'Unknown')}"
            if 'datetime' in row:
                popup_text += f"<br>Date: {row['datetime']}"
                
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=popup_text,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.6
            ).add_to(m)
            
        # Display map
        st_folium(m, width=700, height=500)
        
    def run(self):
        """Run the dashboard application."""
        page = self.render_sidebar()
        
        if page == "Home":
            self.render_home_page()
        elif page == "Data Overview":
            self.render_data_overview()
        elif page == "Simulation":
            self.render_simulation_page()
        elif page == "Clustering Analysis":
            self.render_clustering_page()
        elif page == "Risk Modeling":
            st.header("🗺️ Risk Terrain Modeling")
            st.info("Risk modeling interface coming soon...")
        elif page == "Prediction Models":
            st.header("🤖 Machine Learning Prediction")
            st.info("Prediction modeling interface coming soon...")
        elif page == "Visualization":
            st.header("📊 Advanced Visualizations")
            st.info("Advanced visualization tools coming soon...")


def main():
    """Main function to run the Streamlit dashboard."""
    dashboard = CrimeDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
