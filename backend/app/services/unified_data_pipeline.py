# backend/app/services/unified_data_pipeline.py

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import json
from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Import existing pipeline components
from ..utils.data_processor import GridXDataProcessor, DatasetConfig, FaultCategory, TransformerType
from ..utils.feature_extractor import GridXFeatureExtractor, FeatureConfig, FeatureSelector
from ..utils.ett_data_processor import ETTDataProcessor, ETTConfig, ETTSample
from .data_pipeline import PipelineConfig, _resolve_base_dir, _resolve_path_from_env
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

class DatasetType(Enum):
    """Types of datasets in the unified pipeline"""
    IEEE_FAULT = "ieee_fault_detection"
    ETT_OPERATIONAL = "ett_operational_data"

class UnifiedPipelineConfig:
    """Configuration for the unified multi-dataset pipeline"""
    
    def __init__(self, validate_paths: bool = True):
        base_dir = _resolve_base_dir()
        data_root = _resolve_path_from_env("GRIDX_DATA_ROOT", base_dir / "data", base_dir)

        # Dataset paths
        self.ieee_data_path: Path = _resolve_path_from_env(
            "GRIDX_IEEE_DATA_PATH",
            data_root / "raw" / "ieee_fault_detection",
            base_dir,
        )
        self.ett_data_path: Path = _resolve_path_from_env(
            "GRIDX_ETT_DATA_PATH",
            data_root / "raw" / "ett_small",
            base_dir,
        )

        
        # Processing parameters
        self.processed_data_path: Path = _resolve_path_from_env(
            "GRIDX_PROCESSED_DATA_PATH",
            data_root / "processed",
            base_dir,
        )
        self.model_data_path: Path = _resolve_path_from_env(
            "GRIDX_MODEL_DATA_PATH",
            data_root / "interim",
            base_dir,
        )
        self.test_size = 0.2
        self.validation_size = 0.2
        self.random_state = 42
        
        # IEEE dataset parameters (fault detection)
        self.ieee_samples_per_class = None
        self.ieee_top_features = 50
        
        # ETT dataset parameters (predictive maintenance)
        self.ett_window_size = 24  # 24 hours or 24*4 15-min intervals
        self.ett_temp_threshold = 40.0  # Oil temperature threshold
        self.ett_anomaly_threshold = 0.7
        
        # Integration parameters
        self.enable_cross_dataset_features = True
        self.unify_feature_space = True
        
        # Create directories
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.model_data_path.mkdir(parents=True, exist_ok=True)

        if validate_paths:
            self._validate_required_paths(require_ieee=True, require_ett=True)

    def _validate_required_paths(self, *, require_ieee: bool, require_ett: bool) -> None:
        errors: List[str] = []

        if require_ieee and not self.ieee_data_path.exists():
            errors.append(f"IEEE dataset directory not found: {self.ieee_data_path}")

        if require_ett:
            if not self.ett_data_path.exists():
                errors.append(f"ETT dataset directory not found: {self.ett_data_path}")
            elif not any(self.ett_data_path.glob('*.csv')):
                errors.append(f"No CSV files found in ETT dataset directory: {self.ett_data_path}")

        if errors:
            for message in errors:
                logger.error(message)
            raise FileNotFoundError(
                "Required dataset paths are missing or incomplete. Update your environment variables or .env configuration."
            )

    def validate_for_run(self, include_ieee: bool, include_ett: bool) -> None:
        """Validate dataset availability for the requested pipeline run."""

        self._validate_required_paths(require_ieee=include_ieee, require_ett=include_ett)

class UnifiedGridXPipeline:
    """
    Unified pipeline that handles both IEEE fault detection data 
    and ETT operational data for comprehensive transformer diagnostics.
    """
    
    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        self.config = config or UnifiedPipelineConfig(validate_paths=False)
        
        # Initialize IEEE components
        self.ieee_config = DatasetConfig(
            base_path=str(self.config.ieee_data_path),
            parallel_workers=4
        )
        self.ieee_processor = GridXDataProcessor(self.ieee_config)
        self.ieee_feature_extractor = GridXFeatureExtractor(FeatureConfig())
        
        # Initialize ETT components
        self.ett_config = ETTConfig(
            base_path=str(self.config.ett_data_path)
        )
        self.ett_processor = ETTDataProcessor(self.ett_config)
        
        # Pipeline state
        self.pipeline_state = {
            'ieee_data_loaded': False,
            'ett_data_loaded': False,
            'features_extracted': False,
            'unified_dataset_created': False,
            'models_ready': False
        }
        
        # Data storage
        self.ieee_data = {}
        self.ett_data = {}
        self.unified_features = None
        self.unified_targets = None
        
    def run_unified_pipeline(self, include_ieee: bool = True, include_ett: bool = True, 
                           quick_test: bool = False) -> Dict[str, Any]:
        """
        Execute the complete unified pipeline.
        
        Args:
            include_ieee: Whether to process IEEE fault detection data
            include_ett: Whether to process ETT operational data
            quick_test: If True, use small samples for testing
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Starting unified GridX pipeline...")
        start_time = datetime.now()
        
        try:
            self.config.validate_for_run(include_ieee, include_ett)
            results = {}
            
            # Step 1: Load IEEE fault detection data
            if include_ieee:
                logger.info("Step 1a: Loading IEEE fault detection data...")
                ieee_results = self._process_ieee_data(quick_test=quick_test)
                results['ieee'] = ieee_results
                self.pipeline_state['ieee_data_loaded'] = True
                
            # Step 2: Load ETT operational data
            if include_ett:
                logger.info("Step 1b: Loading ETT operational data...")
                ett_results = self._process_ett_data(quick_test=quick_test)
                results['ett'] = ett_results
                self.pipeline_state['ett_data_loaded'] = True
                
            # Step 3: Create unified feature space
            if include_ieee and include_ett:
                logger.info("Step 2: Creating unified feature space...")
                unified_results = self._create_unified_features()
                results['unified'] = unified_results
                self.pipeline_state['unified_dataset_created'] = True
                
            # Step 4: Prepare final datasets
            logger.info("Step 3: Preparing final model datasets...")
            final_results = self._prepare_final_datasets(include_ieee, include_ett)
            results['final'] = final_results
            self.pipeline_state['models_ready'] = True
            
            # Step 5: Save all processed data
            logger.info("Step 4: Saving processed data...")
            self._save_unified_data(results)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Compile final results
            final_summary = {
                'status': 'success',
                'duration_seconds': duration,
                'pipeline_state': self.pipeline_state,
                'datasets_processed': {
                    'ieee_included': include_ieee,
                    'ett_included': include_ett,
                    'unified_created': include_ieee and include_ett
                },
                'results': results
            }
            
            logger.info(f"Unified pipeline completed successfully in {duration:.1f} seconds")
            return final_summary
            
        except Exception as e:
            logger.error(f"Unified pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'pipeline_state': self.pipeline_state
            }
            
    def _process_ieee_data(self, quick_test: bool = False) -> Dict[str, Any]:
        """Process IEEE fault detection data"""
        
        # Scan dataset
        ieee_stats = self.ieee_processor.scan_dataset()
        
       # Create dataset with optional balancing
        samples_per_class = 10 if quick_test else self.config.ieee_samples_per_class
        balanced_files = self.ieee_processor.create_balanced_dataset(samples_per_class)
        
        # Extract features
        all_feature_data = []
        total_files = sum(len(files) for files in balanced_files.values())
        processed_files = 0
        
        for class_name, file_paths in balanced_files.items():
            # Process in batches
            batch_size = 20 if quick_test else 50
            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i:i + batch_size]
                
                # Load and extract features
                data_dict = self.ieee_processor.load_batch_files(batch_files)
                if data_dict:
                    batch_features = self.ieee_feature_extractor.extract_features_batch(data_dict)
                    if not batch_features.empty:
                        batch_features['fault_class'] = class_name
                        batch_features['dataset_type'] = 'ieee_fault'
                        all_feature_data.append(batch_features)
                        
                processed_files += len(data_dict)
                
                if processed_files % 100 == 0:
                    logger.info(f"IEEE: Processed {processed_files}/{total_files} files")
                    
        # Combine all IEEE features
        if all_feature_data:
            ieee_features = pd.concat(all_feature_data, ignore_index=True)
            self.ieee_data['features'] = ieee_features
            
            # Create labels
            le_fault = LabelEncoder()
            ieee_labels = le_fault.fit_transform(ieee_features['fault_class'])
            self.ieee_data['labels'] = ieee_labels
            self.ieee_data['label_encoder'] = le_fault
            self.ieee_data['class_mapping'] = {cls: idx for idx, cls in enumerate(le_fault.classes_)}
            
            logger.info(f"IEEE processing complete: {len(ieee_features)} samples, {len(le_fault.classes_)} fault classes")
            
            return {
                'samples': len(ieee_features),
                'features': len(ieee_features.columns) - 2,  # Exclude metadata columns
                'classes': len(le_fault.classes_),
                'class_names': le_fault.classes_.tolist()
            }
        else:
            return {'samples': 0, 'features': 0, 'classes': 0}
            
    def _process_ett_data(self, quick_test: bool = False) -> Dict[str, Any]:
        """Process ETT operational data"""
        
        # Load ETT datasets
        ett_stats = self.ett_processor.load_ett_datasets()
        
        # Create ETT samples
        samples_dict = self.ett_processor.create_ett_samples(anomaly_detection=True)
        
        # Process samples for each dataset
        all_operational_features = []
        all_maintenance_targets = []
        
        for dataset_name, samples in samples_dict.items():
            if quick_test:
                samples = samples[:500]  # Limit for testing
                
            logger.info(f"Processing ETT {dataset_name}: {len(samples)} samples")
            
            # Extract operational features
            features_df = self.ett_processor.extract_operational_features(
                samples, 
                window_size=self.config.ett_window_size
            )
            
            # Create maintenance targets
            targets_df = self.ett_processor.create_maintenance_targets(
                samples,
                temp_threshold=self.config.ett_temp_threshold,
                anomaly_threshold=self.config.ett_anomaly_threshold
            )
            
            # Add dataset identifier
            features_df['dataset_name'] = dataset_name
            features_df['dataset_type'] = 'ett_operational'
            targets_df['dataset_name'] = dataset_name
            
            all_operational_features.append(features_df)
            all_maintenance_targets.append(targets_df)
            
        # Combine all ETT data
        if all_operational_features:
            ett_features = pd.concat(all_operational_features, ignore_index=True)
            ett_targets = pd.concat(all_maintenance_targets, ignore_index=True)
            
            self.ett_data['features'] = ett_features
            self.ett_data['targets'] = ett_targets
            
            # Create maintenance labels
            maintenance_labels = ett_targets['maintenance_needed'].values
            urgency_labels = ett_targets['urgency_level'].values
            
            self.ett_data['maintenance_labels'] = maintenance_labels
            self.ett_data['urgency_labels'] = urgency_labels
            
            logger.info(f"ETT processing complete: {len(ett_features)} samples")
            logger.info(f"Maintenance needed: {np.sum(maintenance_labels)} samples")
            
            return {
                'samples': len(ett_features),
                'features': len(ett_features.columns) - 2,  # Exclude metadata
                'maintenance_needed': int(np.sum(maintenance_labels)),
                'urgency_distribution': {int(k): int(v) for k, v in pd.Series(urgency_labels).value_counts().items()}
            }
        else:
            return {'samples': 0, 'features': 0}
            
    def _create_unified_features(self) -> Dict[str, Any]:
        """Create unified feature space combining IEEE and ETT data"""
        
        if not self.ieee_data or not self.ett_data:
            logger.warning("Cannot create unified features - missing IEEE or ETT data")
            return {}
            
        logger.info("Creating unified feature space...")
        
        # Get IEEE fault detection features (numerical only)
        ieee_features = self.ieee_data['features']
        ieee_numerical = ieee_features.select_dtypes(include=[np.number])
        
        # Get ETT operational features (numerical only)  
        ett_features = self.ett_data['features']
        ett_numerical = ett_features.select_dtypes(include=[np.number])
        
        # Create cross-dataset features if enabled
        if self.config.enable_cross_dataset_features:
            # Statistical correlations between fault patterns and operational patterns  
            cross_features = {}
            
            # IEEE fault severity indicators
            ieee_severity = ieee_numerical.std(axis=1)  # Signal variability as fault severity
            cross_features['ieee_fault_severity_mean'] = ieee_severity.mean()
            cross_features['ieee_fault_severity_std'] = ieee_severity.std()
            
            # ETT operational stress indicators
            ett_stress = (ett_numerical['oil_temperature'] - ett_numerical['oil_temperature'].mean()) / ett_numerical['oil_temperature'].std()
            cross_features['ett_operational_stress_mean'] = ett_stress.mean()
            cross_features['ett_operational_stress_std'] = ett_stress.std()
            
            # Combined risk indicator (conceptual)
            cross_features['combined_risk_indicator'] = (
                cross_features['ieee_fault_severity_mean'] * 0.6 + 
                cross_features['ett_operational_stress_mean'] * 0.4
            )
            
            logger.info(f"Created {len(cross_features)} cross-dataset features")
            
        # Create unified target mapping
        unified_targets = {
            'ieee_fault_classes': self.ieee_data['class_mapping'],
            'ett_maintenance_levels': {
                0: 'Normal',
                1: 'Watch', 
                2: 'Action_Required',
                3: 'Critical'
            }
        }
        
        self.unified_targets = unified_targets
        
        return {
            'ieee_features': ieee_numerical.shape[1],
            'ett_features': ett_numerical.shape[1], 
            'cross_features': len(cross_features) if self.config.enable_cross_dataset_features else 0,
            'target_classes': unified_targets
        }
        
    def _prepare_final_datasets(self, include_ieee: bool, include_ett: bool) -> Dict[str, Any]:
        """Prepare final datasets for model training"""
        
        final_datasets = {}
        
        # IEEE fault detection dataset
        if include_ieee and self.ieee_data:
            ieee_features = self.ieee_data['features']
            ieee_labels = self.ieee_data['labels']
            
            # Select numerical features only
            feature_columns = ieee_features.select_dtypes(include=[np.number]).columns
            X_ieee = ieee_features[feature_columns]
            
            # Handle missing values and scaling
            imputer = SimpleImputer(strategy='median')
            X_ieee_imputed = pd.DataFrame(
                imputer.fit_transform(X_ieee),
                columns=X_ieee.columns
            )
            
            scaler = StandardScaler()
            X_ieee_scaled = pd.DataFrame(
                scaler.fit_transform(X_ieee_imputed),
                columns=X_ieee_imputed.columns
            )
            
            # Train/val/test splits
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_ieee_scaled, ieee_labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=ieee_labels
            )
            
            val_size_adj = self.config.validation_size / (1 - self.config.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adj,
                random_state=self.config.random_state,
                stratify=y_temp
            )

            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=self.config.ieee_top_features)
            X_train_selected = pd.DataFrame(
                selector.fit_transform(X_train, y_train),
                columns=X_train.columns[selector.get_support()],
                index=X_train.index
            )
            X_val_selected = pd.DataFrame(
                selector.transform(X_val),
                columns=X_train_selected.columns,
                index=X_val.index
            )
            X_test_selected = pd.DataFrame(
                selector.transform(X_test),
                columns=X_train_selected.columns,
                index=X_test.index
            )

            final_datasets['ieee_fault_detection'] = {
                'X_train': X_train_selected,
                'X_val': X_val_selected,
                'X_test': X_test_selected,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_names': X_train_selected.columns.tolist(),
                'class_mapping': self.ieee_data['class_mapping'],
                'preprocessing': {
                    'imputer': imputer,
                    'scaler': scaler,
                    'feature_selector': selector
                }
            }
            
            logger.info(f"IEEE dataset prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
        # ETT predictive maintenance dataset
        if include_ett and self.ett_data:
            ett_features = self.ett_data['features']
            ett_maintenance = self.ett_data['maintenance_labels']
            ett_urgency = self.ett_data['urgency_labels']
            
            # Select numerical features
            feature_columns = ett_features.select_dtypes(include=[np.number]).columns
            X_ett = ett_features[feature_columns]
            
            # Handle missing values and scaling
            imputer_ett = SimpleImputer(strategy='median')
            X_ett_imputed = pd.DataFrame(
                imputer_ett.fit_transform(X_ett),
                columns=X_ett.columns
            )
            
            scaler_ett = StandardScaler()
            X_ett_scaled = pd.DataFrame(
                scaler_ett.fit_transform(X_ett_imputed),
                columns=X_ett_imputed.columns
            )
            
            # Splits for maintenance prediction
            X_temp, X_test, y_main_temp, y_main_test, y_urg_temp, y_urg_test = train_test_split(
                X_ett_scaled, ett_maintenance, ett_urgency,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=ett_maintenance
            )
            
            val_size_adj = self.config.validation_size / (1 - self.config.test_size)
            X_train, X_val, y_main_train, y_main_val, y_urg_train, y_urg_val = train_test_split(
                X_temp, y_main_temp, y_urg_temp,
                test_size=val_size_adj,
                random_state=self.config.random_state,
                stratify=y_main_temp
            )
            
            final_datasets['ett_maintenance'] = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_maintenance_train': y_main_train,
                'y_maintenance_val': y_main_val,
                'y_maintenance_test': y_main_test,
                'y_urgency_train': y_urg_train,
                'y_urgency_val': y_urg_val,
                'y_urgency_test': y_urg_test,
                'feature_names': X_ett_scaled.columns.tolist(),
                'preprocessing': {
                    'imputer': imputer_ett,
                    'scaler': scaler_ett
                }
            }
            
            logger.info(f"ETT dataset prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
        return final_datasets
        
    def _save_unified_data(self, results: Dict[str, Any]) -> None:
        """Save all processed data"""
        
        # Save individual datasets
        if 'final' in results:
            datasets_path = Path(self.config.model_data_path) / "unified_datasets.pkl"
            with open(datasets_path, 'wb') as f:
                pickle.dump(results['final'], f)
                
        # Save pipeline metadata
        metadata = {
            'pipeline_state': self.pipeline_state,
            'config': {
                'ieee_samples_per_class': self.config.ieee_samples_per_class,
                'ett_temp_threshold': self.config.ett_temp_threshold,
                'test_size': self.config.test_size,
                'validation_size': self.config.validation_size
            },
            'results_summary': results,
            'unified_targets': self.unified_targets,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = Path(self.config.processed_data_path) / "unified_pipeline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Unified data saved to {self.config.model_data_path}")
        
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current unified pipeline status"""
        return {
            'pipeline_state': self.pipeline_state,
            'ieee_data_available': bool(self.ieee_data),
            'ett_data_available': bool(self.ett_data),
            'unified_features_created': self.unified_targets is not None
        }

# Example usage
if __name__ == "__main__":
    # Initialize unified pipeline
    pipeline = UnifiedGridXPipeline()
    
    # Run complete unified pipeline
    results = pipeline.run_unified_pipeline(
        include_ieee=True,
        include_ett=True,
        quick_test=True  # Set to False for full processing
    )
    
    print("Unified Pipeline Results:")
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"Duration: {results['duration_seconds']:.1f} seconds")
        print("\nDatasets processed:")
        for dataset, included in results['datasets_processed'].items():
            print(f"  {dataset}: {included}")
            
        if 'results' in results:
            if 'ieee' in results['results']:
                ieee_res = results['results']['ieee']
                print(f"\nIEEE Fault Detection:")
                print(f"  Samples: {ieee_res['samples']:,}")
                print(f"  Features: {ieee_res['features']}")
                print(f"  Classes: {ieee_res['classes']}")
                
            if 'ett' in results['results']:
                ett_res = results['results']['ett']
                print(f"\nETT Predictive Maintenance:")
                print(f"  Samples: {ett_res['samples']:,}")
                print(f"  Features: {ett_res['features']}")
                print(f"  Maintenance needed: {ett_res.get('maintenance_needed', 0)}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")