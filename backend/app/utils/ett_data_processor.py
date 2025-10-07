# backend/app/utils/ett_data_processor.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ETTDataType(Enum):
    """ETT Dataset types"""
    HOURLY_1 = "ETTh1"  # Hourly data, station 1
    HOURLY_2 = "ETTh2"  # Hourly data, station 2
    MINUTE_1 = "ETTm1"  # 15-minute data, station 1
    MINUTE_2 = "ETTm2"  # 15-minute data, station 2

@dataclass
class ETTConfig:
    """Configuration for ETT dataset processing"""
    base_path: str
    target_column: str = "OT"  # Oil Temperature
    feature_columns: List[str] = None
    date_column: str = "date"
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

@dataclass
class ETTSample:
    """Single ETT data sample with metadata"""
    timestamp: datetime
    station_id: str
    data_type: ETTDataType
    features: Dict[str, float]
    oil_temperature: float
    load_pattern: str  # "high", "medium", "low"
    anomaly_score: float = 0.0

class ETTDataProcessor:
    """
    ETT Dataset processor for operational transformer data.
    Integrates with existing GridX pipeline for predictive maintenance.
    """
    
    def __init__(self, config: ETTConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.raw_data = {}  # type: Dict[str, pd.DataFrame]
        self.processed_data = {}  # type: Dict[str, List[ETTSample]]
        
        # Define load level thresholds (will be calculated from data)
        self.load_thresholds = {
            'high_load_min': 0.0,
            'medium_load_min': 0.0,
            'low_load_max': 0.0
        }
        
    def load_ett_datasets(self) -> Dict[str, Any]:
        """
        Load all ETT dataset files and return summary statistics.
        
        Returns:
            Dictionary with loading statistics
        """
        logger.info("Loading ETT datasets...")
        
        stats = {
            'files_loaded': 0,
            'total_samples': 0,
            'date_range': {},
            'stations': [],
            'data_types': []
        }
        
        # Load each dataset file
        for data_type in ETTDataType:
            file_path = self.base_path / f"{data_type.value}.csv"
            
            if file_path.exists():
                logger.info(f"Loading {data_type.value}...")
                df = pd.read_csv(file_path)
                
                # Parse date column
                df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
                
                # Store raw data
                self.raw_data[data_type.value] = df
                
                # Update statistics
                stats['files_loaded'] += 1
                stats['total_samples'] += len(df)
                stats['date_range'][data_type.value] = {
                    'start': df[self.config.date_column].min(),
                    'end': df[self.config.date_column].max(),
                    'samples': len(df)
                }
                stats['stations'].append(data_type.value[-1])  # Extract station number
                stats['data_types'].append('hourly' if 'h' in data_type.value else '15-minute')
                
                logger.info(f"  Loaded {len(df):,} samples from {df[self.config.date_column].min()} to {df[self.config.date_column].max()}")
                
            else:
                logger.warning(f"File not found: {file_path}")
                
        # Calculate load thresholds
        self._calculate_load_thresholds()
        
        logger.info(f"ETT dataset loading complete: {stats['files_loaded']} files, {stats['total_samples']:,} total samples")
        return stats
        
    def _calculate_load_thresholds(self) -> None:
        """Calculate load level thresholds from all datasets"""
        
        all_loads = []
        for df in self.raw_data.values():
            # Combine all useful loads for threshold calculation
            useful_loads = df[['HUFL', 'MUFL', 'LUFL']].values.flatten()
            all_loads.extend(useful_loads)
            
        all_loads = np.array(all_loads)
        
        # Define thresholds based on percentiles
        self.load_thresholds = {
            'high_load_min': np.percentile(all_loads, 66.7),    # Top 1/3
            'medium_load_min': np.percentile(all_loads, 33.3),  # Middle 1/3  
            'low_load_max': np.percentile(all_loads, 33.3)      # Bottom 1/3
        }
        
        logger.info(f"Load thresholds calculated: High≥{self.load_thresholds['high_load_min']:.2f}, "
                   f"Medium≥{self.load_thresholds['medium_load_min']:.2f}, "
                   f"Low<{self.load_thresholds['low_load_max']:.2f}")
        
    def create_ett_samples(self, anomaly_detection: bool = True) -> Dict[str, List[ETTSample]]:
        """
        Convert raw ETT data into structured samples for analysis.
        
        Args:
            anomaly_detection: Whether to calculate anomaly scores
            
        Returns:
            Dictionary mapping dataset names to sample lists
        """
        logger.info("Creating ETT samples...")
        
        for dataset_name, df in self.raw_data.items():
            logger.info(f"Processing {dataset_name}...")
            
            data_type = ETTDataType(dataset_name)
            station_id = dataset_name[-1]  # Extract station number
            samples = []
            
            # Calculate anomaly scores if requested
            anomaly_scores = np.zeros(len(df))
            if anomaly_detection:
                anomaly_scores = self._detect_anomalies(df)
                
            # Create samples
            for idx, row in df.iterrows():
                # Determine load pattern
                total_useful_load = row['HUFL'] + row['MUFL'] + row['LUFL']
                
                if total_useful_load >= self.load_thresholds['high_load_min']:
                    load_pattern = "high"
                elif total_useful_load >= self.load_thresholds['medium_load_min']:
                    load_pattern = "medium"
                else:
                    load_pattern = "low"
                    
                # Create sample
                sample = ETTSample(
                    timestamp=row[self.config.date_column],
                    station_id=station_id,
                    data_type=data_type,
                    features={
                        'HUFL': row['HUFL'],
                        'HULL': row['HULL'], 
                        'MUFL': row['MUFL'],
                        'MULL': row['MULL'],
                        'LUFL': row['LUFL'],
                        'LULL': row['LULL'],
                        'OT': row['OT']
                    },
                    oil_temperature=row['OT'],
                    load_pattern=load_pattern,
                    anomaly_score=anomaly_scores[idx]
                )
                
                samples.append(sample)
                
            self.processed_data[dataset_name] = samples
            logger.info(f"  Created {len(samples):,} samples for {dataset_name}")
            
        return self.processed_data
        
    def _detect_anomalies(self, df: pd.DataFrame, method: str = "isolation_forest") -> np.ndarray:
        """
        Detect anomalies in operational data.
        
        Args:
            df: DataFrame with ETT data
            method: Anomaly detection method
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for anomaly detection
            feature_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
            X = df[feature_columns].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            anomaly_scores = -iso_forest.score_samples(X_scaled)  # Convert to positive scores
            
            # Normalize scores to 0-1 range
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            
            logger.info(f"Anomaly detection complete: {np.sum(anomaly_labels == -1)} anomalies detected")
            
            return anomaly_scores
            
        except ImportError:
            logger.warning("sklearn not available, skipping anomaly detection")
            return np.zeros(len(df))
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return np.zeros(len(df))
            
    def extract_operational_features(self, samples: List[ETTSample], window_size: int = 24) -> pd.DataFrame:
        """
        Extract features from operational data for predictive maintenance.
        
        Args:
            samples: List of ETT samples
            window_size: Window size for rolling features (24 = 24 hours or 24*4 = 1 day for 15-min data)
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Extracting operational features from {len(samples)} samples...")
        
        # Convert samples to DataFrame for easier processing
        data_rows = []
        for sample in samples:
            row = {
                'timestamp': sample.timestamp,
                'station_id': sample.station_id,
                'oil_temperature': sample.oil_temperature,
                'load_pattern': sample.load_pattern,
                'anomaly_score': sample.anomaly_score,
                **sample.features
            }
            data_rows.append(row)
            
        df = pd.DataFrame(data_rows)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize feature dictionary
        features_list = []
        
        for idx in range(len(df)):
            features = {}
            
            # Basic features
            features['timestamp'] = df.iloc[idx]['timestamp']
            features['station_id'] = df.iloc[idx]['station_id']
            features['oil_temperature'] = df.iloc[idx]['oil_temperature']
            features['load_pattern'] = df.iloc[idx]['load_pattern']
            features['anomaly_score'] = df.iloc[idx]['anomaly_score']
            
            # Current operational state
            features['total_useful_load'] = df.iloc[idx]['HUFL'] + df.iloc[idx]['MUFL'] + df.iloc[idx]['LUFL']
            features['total_useless_load'] = df.iloc[idx]['HULL'] + df.iloc[idx]['MULL'] + df.iloc[idx]['LULL']
            features['load_efficiency'] = features['total_useful_load'] / (features['total_useful_load'] + features['total_useless_load']) if (features['total_useful_load'] + features['total_useless_load']) > 0 else 0
            
            # Rolling window features (if we have enough history)
            if idx >= window_size:
                window_data = df.iloc[idx-window_size:idx]
                
                # Oil temperature trends
                features['ot_mean_24h'] = window_data['oil_temperature'].mean()
                features['ot_std_24h'] = window_data['oil_temperature'].std()
                features['ot_min_24h'] = window_data['oil_temperature'].min()
                features['ot_max_24h'] = window_data['oil_temperature'].max()
                features['ot_trend_24h'] = window_data['oil_temperature'].iloc[-1] - window_data['oil_temperature'].iloc[0]
                
                # Load patterns
                features['useful_load_mean_24h'] = window_data[['HUFL', 'MUFL', 'LUFL']].mean().mean()
                features['useful_load_std_24h'] = window_data[['HUFL', 'MUFL', 'LUFL']].mean().std()
                features['load_variability_24h'] = window_data[['HUFL', 'MUFL', 'LUFL']].std().mean()
                
                # Anomaly trends
                features['anomaly_score_mean_24h'] = window_data['anomaly_score'].mean()
                features['anomaly_count_24h'] = (window_data['anomaly_score'] > 0.7).sum()
                
                # Temperature-load correlation
                temp_load_corr = np.corrcoef(window_data['oil_temperature'], 
                                           window_data[['HUFL', 'MUFL', 'LUFL']].mean(axis=1))[0, 1]
                features['temp_load_correlation_24h'] = temp_load_corr if not np.isnan(temp_load_corr) else 0
                
            else:
                # Fill with current values if not enough history
                features.update({
                    'ot_mean_24h': features['oil_temperature'],
                    'ot_std_24h': 0,
                    'ot_min_24h': features['oil_temperature'],
                    'ot_max_24h': features['oil_temperature'],
                    'ot_trend_24h': 0,
                    'useful_load_mean_24h': features['total_useful_load'],
                    'useful_load_std_24h': 0,
                    'load_variability_24h': 0,
                    'anomaly_score_mean_24h': features['anomaly_score'],
                    'anomaly_count_24h': 0,
                    'temp_load_correlation_24h': 0
                })
                
            # Time-based features
            dt = df.iloc[idx]['timestamp']
            features['hour'] = dt.hour
            features['day_of_week'] = dt.weekday()
            features['month'] = dt.month
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            
            # Seasonal indicators (simple)
            features['season'] = (dt.month % 12 + 3) // 3  # 1=Spring, 2=Summer, 3=Fall, 4=Winter
            
            features_list.append(features)
            
        feature_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(feature_df.columns)} operational features")
        
        return feature_df
        
    def create_maintenance_targets(self, samples: List[ETTSample], 
                                 temp_threshold: float = 40.0,
                                 anomaly_threshold: float = 0.7) -> pd.DataFrame:
        """
        Create maintenance prediction targets from operational data.
        
        Args:
            samples: List of ETT samples
            temp_threshold: Oil temperature threshold for maintenance alert
            anomaly_threshold: Anomaly score threshold for maintenance alert
            
        Returns:
            DataFrame with maintenance targets
        """
        logger.info("Creating maintenance prediction targets...")
        
        targets = []
        
        for i, sample in enumerate(samples):
            target = {
                'timestamp': sample.timestamp,
                'station_id': sample.station_id,
                'maintenance_needed': 0,  # Binary target
                'urgency_level': 0,       # 0=Normal, 1=Watch, 2=Action, 3=Critical
                'predicted_failure_days': 999  # Days until predicted failure
            }
            
            # Maintenance criteria
            needs_maintenance = False
            urgency = 0
            
            # High oil temperature
            if sample.oil_temperature > temp_threshold:
                needs_maintenance = True
                urgency = max(urgency, 2)
                
            # Very high oil temperature  
            if sample.oil_temperature > temp_threshold + 10:
                urgency = max(urgency, 3)
                
            # High anomaly score
            if sample.anomaly_score > anomaly_threshold:
                needs_maintenance = True
                urgency = max(urgency, 1)
                
            # Very high anomaly score
            if sample.anomaly_score > 0.9:
                urgency = max(urgency, 2)
                
            # Load pattern considerations
            if sample.load_pattern == "high" and sample.oil_temperature > temp_threshold - 5:
                needs_maintenance = True
                urgency = max(urgency, 1)
                
            target['maintenance_needed'] = 1 if needs_maintenance else 0
            target['urgency_level'] = urgency
            
            # Simple failure prediction (based on temperature trend)
            if sample.oil_temperature > temp_threshold:
                target['predicted_failure_days'] = max(1, int(100 - sample.oil_temperature))
            
            targets.append(target)
            
        target_df = pd.DataFrame(targets)
        
        # Log statistics
        maintenance_needed = target_df['maintenance_needed'].sum()
        logger.info(f"Maintenance targets created: {maintenance_needed}/{len(targets)} samples need maintenance")
        logger.info(f"Urgency distribution: {target_df['urgency_level'].value_counts().to_dict()}")
        
        return target_df
        
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of loaded ETT data"""
        
        summary = {
            'datasets_loaded': len(self.raw_data),
            'total_samples': sum(len(df) for df in self.raw_data.values()),
            'date_range': {},
            'feature_statistics': {},
            'load_thresholds': self.load_thresholds,
            'stations': list(set(dataset[-1] for dataset in self.raw_data.keys()))
        }
        
        # Date ranges
        all_dates = []
        for dataset_name, df in self.raw_data.items():
            dates = df[self.config.date_column]
            summary['date_range'][dataset_name] = {
                'start': dates.min(),
                'end': dates.max(),
                'samples': len(df)
            }
            all_dates.extend(dates)
            
        if all_dates:
            summary['overall_date_range'] = {
                'start': min(all_dates),
                'end': max(all_dates)
            }
            
        # Feature statistics
        if self.raw_data:
            # Combine all datasets for overall statistics
            all_data = pd.concat(self.raw_data.values(), ignore_index=True)
            
            for col in self.config.feature_columns:
                if col in all_data.columns:
                    summary['feature_statistics'][col] = {
                        'mean': float(all_data[col].mean()),
                        'std': float(all_data[col].std()),
                        'min': float(all_data[col].min()),
                        'max': float(all_data[col].max()),
                        'missing': int(all_data[col].isnull().sum())
                    }
                    
        return summary
