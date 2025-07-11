"""
Spatial clustering analysis for crime hotspot identification in Roysambu ward.

This module implements various clustering algorithms to identify spatial
patterns and hotspots in crime data, including DBSCAN, K-means, and
hierarchical clustering approaches.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, Polygon
import warnings
warnings.filterwarnings('ignore')


class CrimeClusterAnalyzer:
    """
    Performs spatial clustering analysis on crime data.
    
    Implements multiple clustering algorithms to identify crime hotspots
    and analyze spatial patterns in the Roysambu ward.
    """
    
    def __init__(self, crime_data: gpd.GeoDataFrame):
        """
        Initialize cluster analyzer with crime data.
        
        Args:
            crime_data: GeoDataFrame containing crime incidents with geometry
        """
        self.crime_data = crime_data.copy()
        self.coordinates = None
        self.clusters = {}
        self.cluster_metrics = {}
        self.scaler = StandardScaler()
        
        self._prepare_coordinates()
        
    def _prepare_coordinates(self) -> None:
        """Extract and prepare coordinate arrays from crime data."""
        if 'geometry' in self.crime_data.columns:
            # Extract coordinates from geometry
            self.coordinates = np.array([
                [point.x, point.y] for point in self.crime_data.geometry
            ])
        elif 'longitude' in self.crime_data.columns and 'latitude' in self.crime_data.columns:
            # Use explicit coordinate columns
            self.coordinates = self.crime_data[['longitude', 'latitude']].values
        else:
            raise ValueError("Crime data must contain geometry or longitude/latitude columns")
            
    def dbscan_clustering(self, eps: float = 0.01, min_samples: int = 5,
                         metric: str = 'euclidean') -> np.ndarray:
        """
        Perform DBSCAN clustering on crime locations.
        
        Args:
            eps: Maximum distance between samples in a neighborhood
            min_samples: Minimum number of samples in a neighborhood
            metric: Distance metric to use
            
        Returns:
            Array of cluster labels (-1 for noise points)
        """
        # Scale coordinates for better clustering
        coords_scaled = self.scaler.fit_transform(self.coordinates)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        cluster_labels = dbscan.fit_predict(coords_scaled)
        
        # Store results
        self.clusters['dbscan'] = {
            'labels': cluster_labels,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': list(cluster_labels).count(-1),
            'params': {'eps': eps, 'min_samples': min_samples, 'metric': metric}
        }
        
        # Calculate metrics
        if len(set(cluster_labels)) > 1:
            self.cluster_metrics['dbscan'] = self._calculate_cluster_metrics(
                coords_scaled, cluster_labels
            )
        
        return cluster_labels
        
    def kmeans_clustering(self, n_clusters: int = 8, random_state: int = 42) -> np.ndarray:
        """
        Perform K-means clustering on crime locations.
        
        Args:
            n_clusters: Number of clusters to form
            random_state: Random state for reproducibility
            
        Returns:
            Array of cluster labels
        """
        # Scale coordinates
        coords_scaled = self.scaler.fit_transform(self.coordinates)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_scaled)
        
        # Store results
        self.clusters['kmeans'] = {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'centroids': self.scaler.inverse_transform(kmeans.cluster_centers_),
            'params': {'n_clusters': n_clusters, 'random_state': random_state}
        }
        
        # Calculate metrics
        self.cluster_metrics['kmeans'] = self._calculate_cluster_metrics(
            coords_scaled, cluster_labels
        )
        
        return cluster_labels
        
    def hierarchical_clustering(self, n_clusters: int = 8, 
                               linkage: str = 'ward') -> np.ndarray:
        """
        Perform hierarchical clustering on crime locations.
        
        Args:
            n_clusters: Number of clusters to form
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            
        Returns:
            Array of cluster labels
        """
        # Scale coordinates
        coords_scaled = self.scaler.fit_transform(self.coordinates)
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage
        )
        cluster_labels = hierarchical.fit_predict(coords_scaled)
        
        # Store results
        self.clusters['hierarchical'] = {
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'params': {'n_clusters': n_clusters, 'linkage': linkage}
        }
        
        # Calculate metrics
        self.cluster_metrics['hierarchical'] = self._calculate_cluster_metrics(
            coords_scaled, cluster_labels
        )
        
        return cluster_labels
        
    def _calculate_cluster_metrics(self, coordinates: np.ndarray, 
                                  labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        
        Args:
            coordinates: Scaled coordinate array
            labels: Cluster labels
            
        Returns:
            Dictionary of clustering metrics
        """
        metrics = {}
        
        # Remove noise points for metric calculation
        valid_mask = labels != -1
        if valid_mask.sum() > 0:
            valid_coords = coordinates[valid_mask]
            valid_labels = labels[valid_mask]
            
            # Silhouette score
            if len(set(valid_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(valid_coords, valid_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                    valid_coords, valid_labels
                )
            
            # Intra-cluster distances
            metrics['avg_intra_cluster_distance'] = self._calculate_avg_intra_cluster_distance(
                valid_coords, valid_labels
            )
            
        return metrics
        
    def _calculate_avg_intra_cluster_distance(self, coordinates: np.ndarray, 
                                            labels: np.ndarray) -> float:
        """Calculate average intra-cluster distance."""
        total_distance = 0
        total_pairs = 0
        
        for cluster_id in set(labels):
            cluster_points = coordinates[labels == cluster_id]
            if len(cluster_points) > 1:
                distances = pdist(cluster_points)
                total_distance += distances.sum()
                total_pairs += len(distances)
                
        return total_distance / total_pairs if total_pairs > 0 else 0
        
    def optimize_dbscan_parameters(self, eps_range: Tuple[float, float] = (0.005, 0.05),
                                  min_samples_range: Tuple[int, int] = (3, 10),
                                  n_trials: int = 20) -> Dict:
        """
        Optimize DBSCAN parameters using grid search.
        
        Args:
            eps_range: Range of eps values to test
            min_samples_range: Range of min_samples values to test
            n_trials: Number of parameter combinations to test
            
        Returns:
            Best parameters and their metrics
        """
        best_params = {}
        best_score = -1
        
        # Generate parameter combinations
        eps_values = np.linspace(eps_range[0], eps_range[1], int(np.sqrt(n_trials)))
        min_samples_values = np.linspace(
            min_samples_range[0], min_samples_range[1], 
            int(np.sqrt(n_trials)), dtype=int
        )
        
        coords_scaled = self.scaler.fit_transform(self.coordinates)
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(coords_scaled)
                
                # Skip if only one cluster or all noise
                unique_labels = set(labels)
                if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
                    continue
                    
                # Calculate silhouette score for valid points
                valid_mask = labels != -1
                if valid_mask.sum() > min_samples:
                    score = silhouette_score(coords_scaled[valid_mask], labels[valid_mask])
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'eps': eps,
                            'min_samples': min_samples,
                            'silhouette_score': score,
                            'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
                            'n_noise': list(labels).count(-1)
                        }
        
        return best_params
        
    def analyze_cluster_characteristics(self, cluster_method: str = 'dbscan') -> Dict:
        """
        Analyze characteristics of identified clusters.
        
        Args:
            cluster_method: Clustering method to analyze
            
        Returns:
            Dictionary with cluster analysis results
        """
        if cluster_method not in self.clusters:
            raise ValueError(f"Clustering method '{cluster_method}' not found. "
                           f"Available methods: {list(self.clusters.keys())}")
            
        labels = self.clusters[cluster_method]['labels']
        analysis = {}
        
        # Add cluster labels to crime data
        crime_with_clusters = self.crime_data.copy()
        crime_with_clusters['cluster'] = labels
        
        # Analyze each cluster
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_crimes = crime_with_clusters[crime_with_clusters['cluster'] == cluster_id]
            
            cluster_info = {
                'size': len(cluster_crimes),
                'center': self._calculate_cluster_center(cluster_crimes),
                'area': self._calculate_cluster_area(cluster_crimes),
                'density': len(cluster_crimes) / self._calculate_cluster_area(cluster_crimes)
            }
            
            # Add crime type analysis if available
            if 'crime_type' in cluster_crimes.columns:
                cluster_info['crime_types'] = cluster_crimes['crime_type'].value_counts().to_dict()
                
            # Add temporal analysis if available
            if 'datetime' in cluster_crimes.columns:
                cluster_info['temporal_patterns'] = self._analyze_temporal_patterns(cluster_crimes)
                
            analysis[f'cluster_{cluster_id}'] = cluster_info
            
        return analysis
        
    def _calculate_cluster_center(self, cluster_data: gpd.GeoDataFrame) -> Tuple[float, float]:
        """Calculate the geographic center of a cluster."""
        if 'geometry' in cluster_data.columns:
            centroid = cluster_data.geometry.centroid.iloc[0]
            return (centroid.x, centroid.y)
        else:
            return (
                cluster_data['longitude'].mean(),
                cluster_data['latitude'].mean()
            )
            
    def _calculate_cluster_area(self, cluster_data: gpd.GeoDataFrame) -> float:
        """Calculate the area covered by a cluster (rough approximation)."""
        if len(cluster_data) < 3:
            return 0.001  # Small default area
            
        # Use convex hull for area calculation
        if 'geometry' in cluster_data.columns:
            points = cluster_data.geometry.tolist()
        else:
            points = [Point(row['longitude'], row['latitude']) 
                     for _, row in cluster_data.iterrows()]
            
        # Create convex hull and calculate area
        from shapely.ops import unary_union
        hull = unary_union(points).convex_hull
        return hull.area
        
    def _analyze_temporal_patterns(self, cluster_data: gpd.GeoDataFrame) -> Dict:
        """Analyze temporal patterns within a cluster."""
        # TODO: Implement temporal pattern analysis
        return {'hour_distribution': {}, 'day_distribution': {}}
        
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary of all clustering results."""
        summary_data = []
        
        for method, results in self.clusters.items():
            row = {
                'method': method,
                'n_clusters': results['n_clusters'],
                'n_noise': results.get('n_noise', 0),
                'silhouette_score': self.cluster_metrics.get(method, {}).get('silhouette_score', None),
                'calinski_harabasz_score': self.cluster_metrics.get(method, {}).get('calinski_harabasz_score', None)
            }
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)
        
    def export_clusters_to_shapefile(self, cluster_method: str, output_path: str) -> None:
        """Export clustering results to shapefile."""
        if cluster_method not in self.clusters:
            raise ValueError(f"Clustering method '{cluster_method}' not found")
            
        # Create GeoDataFrame with cluster labels
        result_gdf = self.crime_data.copy()
        result_gdf['cluster'] = self.clusters[cluster_method]['labels']
        
        # Export to shapefile
        result_gdf.to_file(output_path)


def main():
    """Example usage of clustering analysis."""
    print("Crime clustering analysis module initialized for Roysambu ward")
    print("Available clustering methods: DBSCAN, K-means, Hierarchical")
    

if __name__ == "__main__":
    main()
