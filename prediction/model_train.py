"""
Machine learning model training for crime hotspot prediction in Roysambu ward.

This module implements various machine learning algorithms for predicting
crime hotspots using spatial, temporal, and contextual features.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Handles feature engineering for crime prediction models.
    
    Creates spatial, temporal, and contextual features from raw crime
    and environmental data.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        self.scalers = {}
        self.encoders = {}
        
    def create_spatial_features(self, data: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Create spatial features from geographic data.
        
        Args:
            data: GeoDataFrame containing crime incidents
            
        Returns:
            DataFrame with spatial features
        """
        features = pd.DataFrame()
        
        # Extract coordinates
        if 'geometry' in data.columns:
            features['longitude'] = data.geometry.x
            features['latitude'] = data.geometry.y
        else:
            features['longitude'] = data['longitude']
            features['latitude'] = data['latitude']
            
        # Distance to various facilities
        # TODO: Implement distance calculations to:
        # - Police stations
        # - Schools
        # - Bars/entertainment venues
        # - ATMs
        # - Public transport stops
        # - Markets/shopping centers
        
        # Grid-based features
        features['grid_x'] = self._create_grid_features(features['longitude'], 'x')
        features['grid_y'] = self._create_grid_features(features['latitude'], 'y')
        
        # Density features
        # TODO: Implement local crime density calculations
        
        self.feature_names.extend(['longitude', 'latitude', 'grid_x', 'grid_y'])
        
        return features
        
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime information.
        
        Args:
            data: DataFrame containing datetime column
            
        Returns:
            DataFrame with temporal features
        """
        features = pd.DataFrame()
        
        if 'datetime' in data.columns:
            dt = pd.to_datetime(data['datetime'])
            
            # Basic temporal features
            features['hour'] = dt.dt.hour
            features['day_of_week'] = dt.dt.dayofweek
            features['month'] = dt.dt.month
            features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            
            # Time periods
            features['is_night'] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 5)).astype(int)
            features['is_rush_hour'] = ((dt.dt.hour.isin([7, 8, 17, 18]))).astype(int)
            
            # Cyclical encoding for temporal features
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
            self.feature_names.extend([
                'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 
                'is_rush_hour', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ])
            
        return features
        
    def create_contextual_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create contextual features from environmental and social data.
        
        Args:
            data: DataFrame containing contextual information
            
        Returns:
            DataFrame with contextual features
        """
        features = pd.DataFrame()
        
        # Weather features (if available)
        if 'temperature' in data.columns:
            features['temperature'] = data['temperature']
            features['is_hot'] = (data['temperature'] > 25).astype(int)
            
        if 'rainfall' in data.columns:
            features['rainfall'] = data['rainfall']
            features['is_rainy'] = (data['rainfall'] > 0).astype(int)
            
        # Economic indicators (if available)
        # TODO: Add unemployment rate, poverty index, etc.
        
        # Event-based features
        # TODO: Add features for:
        # - Public holidays
        # - School holidays
        # - Pay days
        # - Special events
        
        return features
        
    def _create_grid_features(self, coordinates: pd.Series, axis: str) -> pd.Series:
        """Create grid-based features."""
        # Simple grid encoding
        if axis == 'x':
            return pd.cut(coordinates, bins=10, labels=False)
        else:
            return pd.cut(coordinates, bins=10, labels=False)
            
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        encoded_data = data.copy()
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                encoded_data[col] = self.encoders[col].fit_transform(data[col])
            else:
                encoded_data[col] = self.encoders[col].transform(data[col])
                
        return encoded_data
        
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        scaled_data = data.copy()
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            scaled_data[numerical_columns] = self.scalers['standard'].fit_transform(
                data[numerical_columns]
            )
        else:
            scaled_data[numerical_columns] = self.scalers['standard'].transform(
                data[numerical_columns]
            )
            
        return scaled_data


class CrimePredictor:
    """
    Main class for training and deploying crime prediction models.
    
    Supports multiple machine learning algorithms and provides
    comprehensive model evaluation capabilities.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize crime predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'logistic', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.feature_importance = {}
        
    def _get_model(self) -> Any:
        """Get model instance based on model type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def prepare_features(self, crime_data: gpd.GeoDataFrame, 
                        environmental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.
        
        Args:
            crime_data: GeoDataFrame containing crime incidents
            environmental_data: Optional environmental/contextual data
            
        Returns:
            DataFrame with engineered features
        """
        # Create spatial features
        spatial_features = self.feature_engineer.create_spatial_features(crime_data)
        
        # Create temporal features
        temporal_features = self.feature_engineer.create_temporal_features(crime_data)
        
        # Create contextual features
        if environmental_data is not None:
            contextual_features = self.feature_engineer.create_contextual_features(
                environmental_data
            )
        else:
            contextual_features = pd.DataFrame()
            
        # Combine all features
        all_features = pd.concat([
            spatial_features, 
            temporal_features, 
            contextual_features
        ], axis=1)
        
        # Handle categorical variables
        all_features = self.feature_engineer.encode_categorical_features(all_features)
        
        return all_features
        
    def create_target_variable(self, grid_data: pd.DataFrame, 
                             crime_threshold: int = 1) -> np.ndarray:
        """
        Create binary target variable for hotspot classification.
        
        Args:
            grid_data: DataFrame with grid cells and crime counts
            crime_threshold: Minimum crimes to classify as hotspot
            
        Returns:
            Binary array indicating hotspot status
        """
        if 'crime_count' in grid_data.columns:
            return (grid_data['crime_count'] >= crime_threshold).astype(int)
        else:
            # TODO: Calculate crime counts per grid cell
            return np.zeros(len(grid_data))
            
    def train(self, X: pd.DataFrame, y: np.ndarray, 
              validation_split: float = 0.2) -> Dict:
        """
        Train the crime prediction model.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction of data for validation
            
        Returns:
            Training results dictionary
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)
        X_val_scaled = self.feature_engineer.scale_features(X_val, fit=False)
        
        # Initialize and train model
        self.model = self._get_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        results = {
            'accuracy': self.model.score(X_val_scaled, y_val),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'classification_report': classification_report(y_val, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                X.columns, 
                self.model.feature_importances_
            ))
            results['feature_importance'] = self.feature_importance
            
        self.is_trained = True
        return results
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_scaled = self.feature_engineer.scale_features(X, fit=False)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        X_scaled = self.feature_engineer.scale_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
        
    def hyperparameter_tuning(self, X: pd.DataFrame, y: np.ndarray, 
                            cv_folds: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best parameters and CV results
        """
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'logistic': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        }
        
        if self.model_type not in param_grids:
            raise ValueError(f"Hyperparameter tuning not implemented for {self.model_type}")
            
        # Scale features
        X_scaled = self.feature_engineer.scale_features(X, fit=True)
        
        # Perform grid search
        grid_search = GridSearchCV(
            self._get_model(),
            param_grids[self.model_type],
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_engineer': self.feature_engineer,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = True


def main():
    """Example usage of crime prediction model."""
    print("Crime prediction model training module initialized")
    print(f"Available models: random_forest, xgboost, logistic, svm, gradient_boosting")
    

if __name__ == "__main__":
    main()
