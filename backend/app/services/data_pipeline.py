# backend/app/services/data_pipeline.py

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from ..utils.data_processor import GridXDataProcessor, DatasetConfig, FaultCategory, TransformerType
from ..utils.feature_extractor import GridXFeatureExtractor, FeatureConfig, FeatureSelector
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_base_dir() -> Path:
    base_dir_env = os.getenv("GRIDX_BASE_DIR")
    if not base_dir_env:
        return _REPO_ROOT

    base_dir = Path(base_dir_env).expanduser()
    if not base_dir.is_absolute():
        base_dir = _REPO_ROOT / base_dir
    return base_dir


def _resolve_path_from_env(var_name: str, default: Path, base_dir: Optional[Path] = None) -> Path:
    base_dir = base_dir or _resolve_base_dir()
    env_value = os.getenv(var_name)

    if not env_value:
        return default

    path = Path(env_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path

class PipelineConfig:
    """Configuration for the complete data pipeline"""
    
    def __init__(self, production_mode=True):
        # Data paths
        self.raw_data_path = r"C:\Users\ogbonda\documents\Gridx-Datasets\Dataset for Transformer & PAR transients\Dataset for Transformer & PAR transients\data for transformer and par"
        self.processed_data_path = "./data/processed"
        self.model_data_path = "./data/interim"
        
        # Processing parameters
        self.test_size = 0.2
        self.validation_size = 0.2
        self.random_state = 42
        
        # FIXED: Scale up samples per class based on available data
        if production_mode:
            # Use much larger sample size for production models
            self.samples_per_class = 2000  # Increased from 500
            self.max_files_per_class = None  # No limit - use all available files
        else:
            # Testing mode
            self.samples_per_class = 100   # Still reasonable for testing
            self.max_files_per_class = 200 # Limit for quick tests
            
        self.max_parallel_workers = 4
        
        # Feature extraction
        self.n_top_features = 50
        self.variance_threshold = 0.01
        
        # Create directories
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.model_data_path, exist_ok=True)

class GridXDataPipeline:
    """
    Complete end-to-end data pipeline for GridX transformer diagnostics.
    Handles data loading, feature extraction, preprocessing, and model preparation.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.dataset_config = DatasetConfig(
            base_path=self.config.raw_data_path,
            parallel_workers=self.config.max_parallel_workers
        )
        
        self.feature_config = FeatureConfig(
            sampling_frequency=10000,
            statistical_features=True,
            frequency_features=True,
            time_domain_features=True,
            symmetrical_components=True
        )
        
        self.data_processor = GridXDataProcessor(self.dataset_config)
        self.feature_extractor = GridXFeatureExtractor(self.feature_config)
        self.feature_selector = FeatureSelector()
        
        # Pipeline state
        self.pipeline_state = {
            'dataset_scanned': False,
            'features_extracted': False,
            'data_preprocessed': False,
            'model_ready': False
        }
        
        self.dataset_stats = {}
        self.class_mapping = {}
        self.feature_columns = []
        self.selected_features = []
        
    def run_complete_pipeline(self, quick_test: bool = False) -> Dict[str, Any]:
        """
        Execute the complete data pipeline.
        
        Args:
            quick_test: If True, use small sample for testing
            
        Returns:
            Dictionary containing pipeline results and metadata
        """
        logger.info("Starting GridX data pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Dataset Discovery
            logger.info("Step 1: Scanning dataset...")
            self.dataset_stats = self.data_processor.scan_dataset()
            self.pipeline_state['dataset_scanned'] = True
            
            # Step 2: Create balanced dataset
            logger.info("Step 2: Creating balanced dataset...")
            if quick_test:
                samples_per_class = 10  # Small sample for testing
            else:
                samples_per_class = self.config.samples_per_class
                
            balanced_files = self.data_processor.create_balanced_dataset(samples_per_class)
            
            # Step 3: Load data and extract features
            logger.info("Step 3: Loading data and extracting features...")
            feature_data = self._extract_features_from_balanced_dataset(balanced_files)
            
            if feature_data.empty:
                raise ValueError("No features extracted - pipeline failed")
                
            self.pipeline_state['features_extracted'] = True
            
            # Step 4: Prepare labels
            logger.info("Step 4: Preparing labels...")
            labels = self._create_labels_from_features(feature_data)
            
            # Step 5: Feature selection and preprocessing
            logger.info("Step 5: Feature selection and preprocessing...")
            processed_data = self._preprocess_features(feature_data, labels)
            self.pipeline_state['data_preprocessed'] = True
            
            # Step 6: Create train/validation/test splits
            logger.info("Step 6: Creating data splits...")
            data_splits = self._create_data_splits(processed_data, labels)
            self.pipeline_state['model_ready'] = True
            
            # Step 7: Save processed data
            logger.info("Step 7: Saving processed data...")
            self._save_processed_data(data_splits, feature_data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Compile results
            results = {
                'status': 'success',
                'duration_seconds': duration,
                'dataset_stats': self.dataset_stats,
                'feature_count': len(self.feature_columns),
                'selected_features': len(self.selected_features),
                'class_count': len(self.class_mapping),
                'sample_count': len(feature_data),
                'data_splits': {
                    'train_size': len(data_splits['X_train']),
                    'validation_size': len(data_splits['X_val']),
                    'test_size': len(data_splits['X_test'])
                },
                'pipeline_state': self.pipeline_state
            }
            
            logger.info(f"Pipeline completed successfully in {duration:.1f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'pipeline_state': self.pipeline_state
            }
            
    def _extract_features_from_balanced_dataset(self, balanced_files: Dict[str, List[str]]) -> pd.DataFrame:
        """Extract features from balanced dataset"""
        
        all_feature_data = []
        total_files = sum(len(files) for files in balanced_files.values())
        processed_files = 0
        
        for class_name, file_paths in balanced_files.items():
            logger.info(f"Processing class {class_name}: {len(file_paths)} files")
            
            # Load files in batches to manage memory
            batch_size = 50
            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i:i + batch_size]
                
                # Load batch of files
                data_dict = self.data_processor.load_batch_files(batch_files)
                
                if not data_dict:
                    continue
                    
                # Extract features from batch
                batch_features = self.feature_extractor.extract_features_batch(data_dict)
                
                if not batch_features.empty:
                    # Add class information
                    batch_features['fault_class'] = class_name
                    batch_features['transformer_type'] = self._get_transformer_type_from_path(batch_files[0])
                    batch_features['fault_category'] = self._get_fault_category_from_path(batch_files[0])
                    
                    all_feature_data.append(batch_features)
                    
                processed_files += len(data_dict)
                
                if processed_files % 100 == 0:
                    logger.info(f"Processed {processed_files}/{total_files} files")
                    
        if all_feature_data:
            combined_features = pd.concat(all_feature_data, ignore_index=True)
            self.feature_columns = [col for col in combined_features.columns 
                                  if col not in ['file_path', 'fault_class', 'transformer_type', 'fault_category']]
            
            logger.info(f"Feature extraction complete: {len(combined_features)} samples, {len(self.feature_columns)} features")
            return combined_features
        else:
            return pd.DataFrame()
            
    def _create_labels_from_features(self, feature_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create label encodings from feature data"""
        
        labels = {}
        
        # Fault class labels
        fault_classes = feature_data['fault_class'].unique()
        self.class_mapping = {cls: idx for idx, cls in enumerate(fault_classes)}
        
        le_fault = LabelEncoder()
        labels['fault_class'] = le_fault.fit_transform(feature_data['fault_class'])
        
        # Transformer type labels
        le_transformer = LabelEncoder()
        labels['transformer_type'] = le_transformer.fit_transform(feature_data['transformer_type'])
        
        # Fault category labels  
        le_category = LabelEncoder()
        labels['fault_category'] = le_category.fit_transform(feature_data['fault_category'])
        
        # Save label encoders
        self._save_label_encoders({
            'fault_class': le_fault,
            'transformer_type': le_transformer,
            'fault_category': le_category
        })
        
        logger.info(f"Created labels: {len(fault_classes)} fault classes, "
                   f"{len(feature_data['transformer_type'].unique())} transformer types")
        
        return labels
        
    def _preprocess_features(self, feature_data: pd.DataFrame, labels: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Preprocess features: selection, scaling, imputation"""
        
        # Extract feature columns only
        X = feature_data[self.feature_columns].copy()
        y = labels['fault_class']  # Primary target
        
        # Handle infinite and very large values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Remove constant features
        constant_features = X.columns[X.std() == 0].tolist()
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
            self.feature_columns = [col for col in self.feature_columns if col not in constant_features]
            
        # Feature selection
        # 1. Variance threshold
        high_variance_features = self.feature_selector.select_features_variance_threshold(
            X, threshold=self.config.variance_threshold
        )
        X = X[high_variance_features]
        
        # 2. Mutual information
        self.selected_features = self.feature_selector.select_features_mutual_info(
            X, y, k=min(self.config.n_top_features, len(high_variance_features))
        )
        X_selected = X[self.selected_features]
        
        # Imputation
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Save preprocessing objects
        self._save_preprocessing_objects({
            'imputer': imputer,
            'scaler': scaler,
            'selected_features': self.selected_features
        })
        
        logger.info(f"Preprocessing complete: {len(self.selected_features)} selected features")
        return X_scaled
        
    def _create_data_splits(self, X: pd.DataFrame, labels: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create train/validation/test splits"""
        
        y = labels['fault_class']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.selected_features,
            'class_mapping': self.class_mapping
        }
        
        return data_splits
        
    def _save_processed_data(self, data_splits: Dict[str, Any], feature_data: pd.DataFrame) -> None:
        """Save processed data to disk"""
        
        # Save data splits
        splits_path = Path(self.config.model_data_path) / "data_splits.pkl"
        with open(splits_path, 'wb') as f:
            pickle.dump(data_splits, f)
            
        # Save feature data  
        features_path = Path(self.config.processed_data_path) / "feature_data.pkl"
        feature_data.to_pickle(features_path)
        
        # Save pipeline metadata
        metadata = {
            'dataset_stats': self.dataset_stats,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'class_mapping': self.class_mapping,
            'pipeline_config': {
                'test_size': self.config.test_size,
                'validation_size': self.config.validation_size,
                'samples_per_class': self.config.samples_per_class,
                'n_top_features': self.config.n_top_features
            },
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = Path(self.config.processed_data_path) / "pipeline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Processed data saved to {self.config.processed_data_path}")
        
    def _save_preprocessing_objects(self, objects: Dict[str, Any]) -> None:
        """Save preprocessing objects"""
        preprocessing_path = Path(self.config.model_data_path) / "preprocessing.pkl"
        with open(preprocessing_path, 'wb') as f:
            pickle.dump(objects, f)
            
    def _save_label_encoders(self, encoders: Dict[str, Any]) -> None:
        """Save label encoders"""
        encoders_path = Path(self.config.model_data_path) / "label_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
            
    def _get_transformer_type_from_path(self, file_path: str) -> str:
        """Extract transformer type from file path"""
        if "ispar exciting" in file_path.lower():
            return "ispar_exciting"
        elif "ispar series" in file_path.lower():
            return "ispar_series"
        else:
            return "power_transformer"
            
    def _get_fault_category_from_path(self, file_path: str) -> str:
        """Extract fault category from file path"""
        if "internal faults" in file_path.lower():
            return "internal_fault"
        else:
            return "transient_disturbance"
            
    def load_processed_data(self) -> Optional[Dict[str, Any]]:
        """Load previously processed data"""
        try:
            splits_path = Path(self.config.model_data_path) / "data_splits.pkl"
            if splits_path.exists():
                with open(splits_path, 'rb') as f:
                    data_splits = pickle.load(f)
                logger.info("Loaded processed data from disk")
                return data_splits
            else:
                logger.warning("No processed data found")
                return None
        except Exception as e:
            logger.error(f"Failed to load processed data: {str(e)}")
            return None
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_state': self.pipeline_state,
            'dataset_stats': self.dataset_stats,
            'feature_count': len(self.feature_columns),
            'selected_features_count': len(self.selected_features),
            'class_count': len(self.class_mapping)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GridXDataPipeline()
    
    # Run complete pipeline (quick test)
    results = pipeline.run_complete_pipeline(quick_test=True)
    
    print("Pipeline Results:")
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print(f"Total samples: {results['sample_count']}")
        print(f"Features extracted: {results['feature_count']}")
        print(f"Features selected: {results['selected_features']}")
        print(f"Classes: {results['class_count']}")
        print("\nData splits:")
        for split, size in results['data_splits'].items():
            print(f"  {split}: {size}")
    else:
        print(f"Error: {results['error']}")# backend/app/services/data_pipeline.py




