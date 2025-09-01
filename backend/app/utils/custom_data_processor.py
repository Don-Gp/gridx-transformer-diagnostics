from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class DatasetCategory(Enum):
    FAULT_DETECTION = "fault_detection"      # Like IEEE dataset
    OPERATIONAL_DATA = "operational_data"    # Like ETT dataset
    CHEMICAL_ANALYSIS = "chemical_analysis"  # For DGA dataset
    CUSTOM = "custom"

@dataclass
class DatasetMetadata:
    name: str
    category: DatasetCategory
    feature_columns: List[str]
    target_columns: List[str]
    timestamp_column: Optional[str] = None
    sample_rate: Optional[str] = None  # e.g., "1H", "15min", "1D"
    description: str = ""

class BaseDataProcessor(ABC):
    """Abstract base class for all data processors"""
    
    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self.raw_data = None
        self.processed_features = None
        
    @abstractmethod
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load raw data from source"""
        pass
        
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features specific to this dataset type"""
        pass
        
    @abstractmethod
    def create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for modeling"""
        pass
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data format and quality"""
        # Check required columns
        missing_cols = set(self.metadata.feature_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for excessive missing values
        missing_pct = data.isnull().mean()
        if (missing_pct > 0.5).any():
            problematic_cols = missing_pct[missing_pct > 0.5].index.tolist()
            raise ValueError(f"Columns with >50% missing values: {problematic_cols}")
            
        return True

class CustomDataProcessor(BaseDataProcessor):
    """Processor for user-defined custom datasets"""
    
    def __init__(self, metadata: DatasetMetadata, feature_config: Dict[str, Any]):
        super().__init__(metadata)
        self.feature_config = feature_config
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load custom data based on file format"""
        path = Path(data_path)
        
        if path.suffix.lower() == '.csv':
            data = pd.read_csv(data_path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(data_path)
        elif path.suffix.lower() == '.json':
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        self.raw_data = data
        return data
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on configuration"""
        features = {}
        
        for feature_name, feature_spec in self.feature_config.items():
            if feature_spec['type'] == 'statistical':
                # Statistical features
                source_col = feature_spec['source_column']
                features[f"{feature_name}_mean"] = data[source_col].mean()
                features[f"{feature_name}_std"] = data[source_col].std()
                features[f"{feature_name}_min"] = data[source_col].min()
                features[f"{feature_name}_max"] = data[source_col].max()
                
            elif feature_spec['type'] == 'temporal':
                # Time-based features
                timestamp_col = self.metadata.timestamp_column
                if timestamp_col:
                    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                    features[f"{feature_name}_hour"] = data[timestamp_col].dt.hour
                    features[f"{feature_name}_day_of_week"] = data[timestamp_col].dt.dayofweek
                    
            elif feature_spec['type'] == 'custom':
                # User-defined feature extraction
                custom_func = feature_spec.get('function')
                if custom_func:
                    features[feature_name] = custom_func(data)
                    
        return pd.DataFrame([features])
        
    def create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables"""
        targets = {}
        
        for target_col in self.metadata.target_columns:
            if target_col in data.columns:
                targets[target_col] = data[target_col]
                
        return pd.DataFrame(targets)

class DGADataProcessor(BaseDataProcessor):
    """Processor for Dissolved Gas Analysis (DGA) data - Third dataset"""
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load DGA data"""
        data = pd.read_csv(data_path)
        self.raw_data = data
        return data
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract DGA-specific features"""
        features = {}
        
        # Gas concentration features
        gas_columns = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
        
        for gas in gas_columns:
            if gas in data.columns:
                features[f"{gas}_concentration"] = data[gas].mean()
                features[f"{gas}_max"] = data[gas].max()
                features[f"{gas}_variability"] = data[gas].std()
                
        # DGA ratios (standard fault indicators)
        if 'C2H2' in data.columns and 'C2H4' in data.columns:
            features['C2H2_C2H4_ratio'] = (data['C2H2'] / data['C2H4']).mean()
            
        if 'CH4' in data.columns and 'H2' in data.columns:
            features['CH4_H2_ratio'] = (data['CH4'] / data['H2']).mean()
            
        if 'C2H4' in data.columns and 'C2H6' in data.columns:
            features['C2H4_C2H6_ratio'] = (data['C2H4'] / data['C2H6']).mean()
            
        # Total Dissolved Combustible Gases (TDCG)
        combustible_gases = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
        available_gases = [gas for gas in combustible_gases if gas in data.columns]
        if available_gases:
            features['TDCG'] = data[available_gases].sum(axis=1).mean()
            
        return pd.DataFrame([features])
        
    def create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create DGA fault classification targets"""
        targets = {}
        
        # If fault_type column exists, use it
        if 'fault_type' in data.columns:
            targets['dga_fault_type'] = data['fault_type']
            
        # Create severity levels based on gas concentrations
        if 'C2H2' in data.columns:
            # Acetylene levels indicate discharge faults
            targets['discharge_severity'] = pd.cut(
                data['C2H2'], 
                bins=[0, 35, 100, float('inf')], 
                labels=['Normal', 'Low', 'High']
            )
            
        return pd.DataFrame(targets)