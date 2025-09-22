# backend/app/utils/data_processor.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerType(Enum):
    """Enumeration for transformer types"""
    ISPAR_EXCITING = "ispar_exciting"
    ISPAR_SERIES = "ispar_series" 
    POWER_TRANSFORMER = "power_transformer"

class FaultCategory(Enum):
    """Enumeration for fault categories"""
    INTERNAL_FAULT = "internal_fault"
    TRANSIENT_DISTURBANCE = "transient_disturbance"

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    base_path: str
    sampling_frequency: int = 10000  # 10kHz
    time_window: float = 0.0725      # seconds
    expected_samples: int = 726      # rows per file
    expected_columns: int = 4        # time + 3 phase currents
    parallel_workers: int = 4

@dataclass
class FileMetadata:
    """Metadata for each data file"""
    file_path: str
    transformer_type: TransformerType
    fault_category: FaultCategory
    fault_class: str
    fault_subclass: Optional[str] = None
    file_size: int = 0

class GridXDataProcessor:
    """
    Comprehensive data processor for GridX transformer diagnostic system.
    Handles loading, validation, and preprocessing of IEEE DataPort dataset.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.file_registry: List[FileMetadata] = []
        
        # Class mapping based on IEEE documentation
        self.class_mapping = {
            'Class1': 'Phase_A_to_Ground',
            'Class2': 'Phase_B_to_Ground', 
            'Class3': 'Phase_C_to_Ground',
            'Class4': 'Phase_AB_to_Ground',
            'Class5': 'Phase_AC_to_Ground',
            'Class6': 'Phase_BC_to_Ground',
            'Class7': 'Phase_ABC_to_Ground',
            'Class8': 'Phase_AB',
            'Class9': 'Phase_AC',
            'Class10': 'Phase_BC',
            'Class11': 'Phase_ABC',
            'tt': 'Turn_to_Turn_Fault',
            'ww': 'Winding_to_Winding_Fault'
        }
        
        self.transient_mapping = {
            'capacitor switching': 'Capacitor_Switching',
            'external fault with CT saturation': 'External_Fault_CT_Saturation',
            'ferroresonance': 'Ferroresonance',
            'magnetising inrush': 'Magnetizing_Inrush',
            'non-linear load switching': 'Nonlinear_Load_Switching',
            'sympathetic inrush': 'Sympathetic_Inrush'
        }
        
    def scan_dataset(self) -> Dict[str, int]:
        """
        Scan the entire dataset and build file registry.
        
        Returns:
            Dict with dataset statistics
        """
        logger.info("Starting dataset scan...")
        self.file_registry.clear()
        
        # Scan internal faults
        internal_faults_path = self.base_path / "internal faults"
        if internal_faults_path.exists():
            self._scan_internal_faults(internal_faults_path)
            
        # Scan transient disturbances  
        transient_path = self.base_path / "transient disturbances"
        if transient_path.exists():
            self._scan_transient_disturbances(transient_path)
            
        stats = self._calculate_statistics()
        logger.info(f"Dataset scan complete. Total files: {len(self.file_registry)}")
        return stats
        
    def _scan_internal_faults(self, internal_faults_path: Path) -> None:
        """Scan internal faults directory structure"""
        
        transformer_mappings = {
            "ispar exciting transformer internal faults": TransformerType.ISPAR_EXCITING,
            "ispar series transformer internal faults": TransformerType.ISPAR_SERIES,
            "power transformer internal faults": TransformerType.POWER_TRANSFORMER
        }
        
        for transformer_dir in internal_faults_path.iterdir():
            if transformer_dir.is_dir() and transformer_dir.name in transformer_mappings:
                transformer_type = transformer_mappings[transformer_dir.name]
                
                for class_dir in transformer_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        
                        # Process files in this class directory
                        for file_path in class_dir.glob("*.txt"):
                            metadata = FileMetadata(
                                file_path=str(file_path),
                                transformer_type=transformer_type,
                                fault_category=FaultCategory.INTERNAL_FAULT,
                                fault_class=self.class_mapping.get(class_name, class_name),
                                fault_subclass=class_name,
                                file_size=file_path.stat().st_size
                            )
                            self.file_registry.append(metadata)
                            
    def _scan_transient_disturbances(self, transient_path: Path) -> None:
        """Scan transient disturbances directory structure"""
        
        for transient_dir in transient_path.iterdir():
            if transient_dir.is_dir():
                transient_type = self.transient_mapping.get(
                    transient_dir.name, 
                    transient_dir.name.replace(' ', '_').title()
                )
                
                for file_path in transient_dir.glob("*.txt"):
                    metadata = FileMetadata(
                        file_path=str(file_path),
                        transformer_type=TransformerType.POWER_TRANSFORMER,  # Default for transients
                        fault_category=FaultCategory.TRANSIENT_DISTURBANCE,
                        fault_class=transient_type,
                        fault_subclass=transient_dir.name,
                        file_size=file_path.stat().st_size
                    )
                    self.file_registry.append(metadata)
                    
    def _calculate_statistics(self) -> Dict[str, int]:
        """Calculate dataset statistics"""
        stats = {
            'total_files': len(self.file_registry),
            'internal_faults': 0,
            'transient_disturbances': 0,
            'ispar_exciting': 0,
            'ispar_series': 0,
            'power_transformer': 0
        }
        
        for metadata in self.file_registry:
            if metadata.fault_category == FaultCategory.INTERNAL_FAULT:
                stats['internal_faults'] += 1
            else:
                stats['transient_disturbances'] += 1
                
            if metadata.transformer_type == TransformerType.ISPAR_EXCITING:
                stats['ispar_exciting'] += 1
            elif metadata.transformer_type == TransformerType.ISPAR_SERIES:
                stats['ispar_series'] += 1
            else:
                stats['power_transformer'] += 1
                
        return stats
        
    def load_single_file(self, file_path: str, validate: bool = True) -> Optional[pd.DataFrame]:
        """
        Load and validate a single data file.
        
        Args:
            file_path: Path to the data file
            validate: Whether to perform data validation
            
        Returns:
            DataFrame with columns [time, phase_a, phase_b, phase_c] or None if invalid
        """
        try:
            # Load data with proper column names
            df = pd.read_csv(
                file_path,
                header=None,
                names=['time', 'phase_a', 'phase_b', 'phase_c']
            )
            
            if validate and not self._validate_file_data(df, file_path):
                return None
                
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load file {file_path}: {str(e)}")
            return None
            
    def _validate_file_data(self, df: pd.DataFrame, file_path: str) -> bool:
        """Validate loaded data file"""
        
        # Check dimensions
        if df.shape[1] != self.config.expected_columns:
            logger.warning(f"File {file_path}: Expected {self.config.expected_columns} columns, got {df.shape[1]}")
            return False
            
        if abs(df.shape[0] - self.config.expected_samples) > 10:  # Allow small variation
            logger.warning(f"File {file_path}: Expected ~{self.config.expected_samples} rows, got {df.shape[0]}")
            
        # Check for missing values
        if df.isnull().any().any():
            logger.warning(f"File {file_path}: Contains missing values")
            return False
            
        # Check time column
        if not df['time'].is_monotonic_increasing:
            logger.warning(f"File {file_path}: Time column is not monotonic")
            return False
            
        return True
        
    def load_batch_files(self, file_paths: List[str], max_workers: int = None) -> Dict[str, pd.DataFrame]:
        """
        Load multiple files in parallel.
        
        Args:
            file_paths: List of file paths to load
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping file paths to DataFrames
        """
        if max_workers is None:
            max_workers = self.config.parallel_workers
            
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file loading tasks
            future_to_path = {
                executor.submit(self.load_single_file, path): path 
                for path in file_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[file_path] = df
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
        logger.info(f"Successfully loaded {len(results)}/{len(file_paths)} files")
        return results
        
    def get_files_by_category(self, 
                             fault_category: Optional[FaultCategory] = None,
                             transformer_type: Optional[TransformerType] = None,
                             fault_class: Optional[str] = None,
                             limit: Optional[int] = None) -> List[FileMetadata]:
        """
        Filter files by specified criteria.
        
        Args:
            fault_category: Filter by fault category
            transformer_type: Filter by transformer type  
            fault_class: Filter by specific fault class
            limit: Maximum number of files to return
            
        Returns:
            List of filtered file metadata
        """
        filtered_files = self.file_registry.copy()
        
        if fault_category:
            filtered_files = [f for f in filtered_files if f.fault_category == fault_category]
            
        if transformer_type:
            filtered_files = [f for f in filtered_files if f.transformer_type == transformer_type]
            
        if fault_class:
            filtered_files = [f for f in filtered_files if f.fault_class == fault_class]
            
        if limit:
            filtered_files = filtered_files[:limit]
            
        return filtered_files
        
    def create_balanced_dataset(self, samples_per_class: Optional[int] = 100) -> Dict[str, List[str]]:
        """Create a dataset with an optional per-class sample limit.
        
        Args:
            samples_per_class: Number of samples per fault class, or None to use all available
            
        Returns:
            Dictionary mapping class names to file paths
        """
        class_files: Dict[str, List[str]] = {}
        
        # Group files by fault class
        for metadata in self.file_registry:
            class_name = metadata.fault_class
            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append(metadata.file_path)
            
        # Apply sampling limit if specified
        balanced_dataset: Dict[str, List[str]] = {}
        for class_name, file_paths in class_files.items():
            if samples_per_class is None:
                selected_files = file_paths
            elif len(file_paths) >= samples_per_class:
                selected_files = np.random.choice(file_paths, size=samples_per_class, replace=False).tolist()
            else:
                selected_files = file_paths
                logger.warning(
                    f"Class {class_name}: Only {len(file_paths)} files available, requested {samples_per_class}"
                )
                
            balanced_dataset[class_name] = selected_files
            
        return balanced_dataset
        
    def export_file_registry(self, output_path: str) -> None:
        """Export file registry to CSV for analysis"""
        
        registry_data = []
        for metadata in self.file_registry:
            registry_data.append({
                'file_path': metadata.file_path,
                'transformer_type': metadata.transformer_type.value,
                'fault_category': metadata.fault_category.value,
                'fault_class': metadata.fault_class,
                'fault_subclass': metadata.fault_subclass,
                'file_size': metadata.file_size
            })
            
        df = pd.DataFrame(registry_data)
        df.to_csv(output_path, index=False)
        logger.info(f"File registry exported to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Configuration for your dataset
    config = DatasetConfig(
        base_path=r"C:\Users\ogbonda\documents\Gridx-Datasets\Dataset for Transformer & PAR transients\Dataset for Transformer & PAR transients\data for transformer and par",
        parallel_workers=4
    )
    
    # Initialize processor
    processor = GridXDataProcessor(config)
    
    # Scan dataset
    stats = processor.scan_dataset()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # Example: Get sample files from each category
    print("\nSample files by category:")
    
    # Internal faults
    internal_files = processor.get_files_by_category(
        fault_category=FaultCategory.INTERNAL_FAULT, 
        limit=5
    )
    print(f"Internal faults sample: {len(internal_files)} files")
    
    # Transient disturbances  
    transient_files = processor.get_files_by_category(
        fault_category=FaultCategory.TRANSIENT_DISTURBANCE,
        limit=5
    )
    print(f"Transient disturbances sample: {len(transient_files)} files")
    
    # Test loading a single file
    if internal_files:
        sample_file = internal_files[0].file_path
        df = processor.load_single_file(sample_file)
        if df is not None:
            print(f"\nSample file shape: {df.shape}")
            print(f"Sample data:\n{df.head()}")