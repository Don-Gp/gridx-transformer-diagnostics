# backend/app/utils/feature_extractor.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, hilbert
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    sampling_frequency: int = 10000  # Hz
    window_length: int = 726         # samples
    frequency_bands: List[Tuple[float, float]] = None
    statistical_features: bool = True
    frequency_features: bool = True
    time_domain_features: bool = True
    symmetrical_components: bool = True
    
    def __post_init__(self):
        if self.frequency_bands is None:
            # Define standard power system frequency bands
            self.frequency_bands = [
                (0, 50),      # DC - Fundamental
                (50, 150),    # Fundamental - 3rd harmonic  
                (150, 250),   # 3rd - 5th harmonic
                (250, 500),   # 5th - 10th harmonic
                (500, 1000),  # High frequency transients
                (1000, 5000)  # Very high frequency
            ]

class GridXFeatureExtractor:
    """
    Advanced feature extraction for transformer fault diagnosis.
    Extracts comprehensive features from 3-phase differential current signals.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.dt = 1.0 / config.sampling_frequency  # Time step
        
    def extract_comprehensive_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract comprehensive features from 3-phase data.
        
        Args:
            df: DataFrame with columns [time, phase_a, phase_b, phase_c]
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract phase currents
        phases = {
            'A': df['phase_a'].values,
            'B': df['phase_b'].values, 
            'C': df['phase_c'].values
        }
        
        # Statistical features for each phase
        if self.config.statistical_features:
            features.update(self._extract_statistical_features(phases))
            
        # Time domain features
        if self.config.time_domain_features:
            features.update(self._extract_time_domain_features(phases))
            
        # Frequency domain features
        if self.config.frequency_features:
            features.update(self._extract_frequency_features(phases))
            
        # Symmetrical components
        if self.config.symmetrical_components:
            features.update(self._extract_symmetrical_components(phases))
            
        # Cross-phase features
        features.update(self._extract_cross_phase_features(phases))
        
        # Fault-specific indicators
        features.update(self._extract_fault_indicators(phases))
        
        return features
        
    def _extract_statistical_features(self, phases: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract statistical features for each phase"""
        features = {}
        
        for phase_name, signal in phases.items():
            prefix = f"{phase_name}_"
            
            # Basic statistics
            features[f"{prefix}mean"] = np.mean(signal)
            features[f"{prefix}std"] = np.std(signal)
            features[f"{prefix}var"] = np.var(signal)
            features[f"{prefix}min"] = np.min(signal)
            features[f"{prefix}max"] = np.max(signal)
            features[f"{prefix}range"] = np.max(signal) - np.min(signal)
            features[f"{prefix}rms"] = np.sqrt(np.mean(signal**2))
            
            # Distribution shape
            features[f"{prefix}skewness"] = stats.skew(signal)
            features[f"{prefix}kurtosis"] = stats.kurtosis(signal)
            
            # Percentiles
            features[f"{prefix}q25"] = np.percentile(signal, 25)
            features[f"{prefix}q50"] = np.percentile(signal, 50)  # median
            features[f"{prefix}q75"] = np.percentile(signal, 75)
            features[f"{prefix}iqr"] = features[f"{prefix}q75"] - features[f"{prefix}q25"]
            
            # Energy-based features
            features[f"{prefix}energy"] = np.sum(signal**2)
            features[f"{prefix}power"] = np.mean(signal**2)
            
        return features
        
    def _extract_time_domain_features(self, phases: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract time domain characteristic features"""
        features = {}
        
        for phase_name, signal in phases.items():
            prefix = f"{phase_name}_td_"
            
            # Zero crossings
            zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
            features[f"{prefix}zero_crossings"] = len(zero_crossings)
            
            # Peak detection
            peaks, _ = find_peaks(np.abs(signal))
            features[f"{prefix}peak_count"] = len(peaks)
            
            if len(peaks) > 0:
                peak_values = np.abs(signal[peaks])
                features[f"{prefix}peak_mean"] = np.mean(peak_values)
                features[f"{prefix}peak_max"] = np.max(peak_values)
                features[f"{prefix}peak_std"] = np.std(peak_values) if len(peak_values) > 1 else 0
            else:
                features[f"{prefix}peak_mean"] = 0
                features[f"{prefix}peak_max"] = 0
                features[f"{prefix}peak_std"] = 0
                
            # Signal variations
            diff_signal = np.diff(signal)
            features[f"{prefix}diff_mean"] = np.mean(np.abs(diff_signal))
            features[f"{prefix}diff_std"] = np.std(diff_signal)
            features[f"{prefix}diff_max"] = np.max(np.abs(diff_signal))
            
            # Signal envelope
            analytic_signal = hilbert(signal)
            envelope = np.abs(analytic_signal)
            features[f"{prefix}envelope_mean"] = np.mean(envelope)
            features[f"{prefix}envelope_std"] = np.std(envelope)
            
            # Crest factor (calculate RMS locally if not available)
            signal_rms = np.sqrt(np.mean(signal**2))
            if signal_rms > 1e-10:
                features[f"{prefix}crest_factor"] = np.max(np.abs(signal)) / signal_rms
            else:
                features[f"{prefix}crest_factor"] = 0
                
        return features
        
    def _extract_frequency_features(self, phases: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract frequency domain features"""
        features = {}
        
        for phase_name, signal in phases.items():
            prefix = f"{phase_name}_freq_"
            
            # FFT analysis
            N = len(signal)
            fft_values = fft(signal)
            fft_magnitude = np.abs(fft_values[:N//2])
            freqs = fftfreq(N, self.dt)[:N//2]
            
            # Power spectral density
            psd = fft_magnitude**2 / N
            
            # Fundamental frequency analysis (50Hz and harmonics)
            fundamental_freq = 50.0  # Hz
            freq_resolution = freqs[1] - freqs[0]
            
            # Find fundamental component
            fund_idx = int(fundamental_freq / freq_resolution)
            if fund_idx < len(fft_magnitude):
                features[f"{prefix}fundamental_mag"] = fft_magnitude[fund_idx]
                features[f"{prefix}fundamental_power"] = psd[fund_idx]
            else:
                features[f"{prefix}fundamental_mag"] = 0
                features[f"{prefix}fundamental_power"] = 0
                
            # Harmonic analysis (2nd to 10th harmonics)
            total_harmonic_power = 0
            for harmonic in range(2, 11):
                harm_freq = harmonic * fundamental_freq
                harm_idx = int(harm_freq / freq_resolution)
                if harm_idx < len(psd):
                    harmonic_power = psd[harm_idx]
                    features[f"{prefix}harmonic_{harmonic}_power"] = harmonic_power
                    total_harmonic_power += harmonic_power
                else:
                    features[f"{prefix}harmonic_{harmonic}_power"] = 0
                    
            features[f"{prefix}total_harmonic_power"] = total_harmonic_power
            
            # THD (Total Harmonic Distortion)
            if features[f"{prefix}fundamental_power"] > 1e-10:
                features[f"{prefix}thd"] = np.sqrt(total_harmonic_power) / np.sqrt(features[f"{prefix}fundamental_power"])
            else:
                features[f"{prefix}thd"] = 0
                
            # Frequency band analysis
            for i, (f_low, f_high) in enumerate(self.config.frequency_bands):
                band_mask = (freqs >= f_low) & (freqs <= f_high)
                band_power = np.sum(psd[band_mask])
                features[f"{prefix}band_{i}_power"] = band_power
                
            # Spectral features
            features[f"{prefix}spectral_centroid"] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            features[f"{prefix}spectral_rolloff"] = self._calculate_spectral_rolloff(freqs, psd, 0.85)
            features[f"{prefix}spectral_flux"] = np.sum(np.diff(psd)**2)
            
        return features
        
    def _extract_symmetrical_components(self, phases: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract symmetrical components (positive, negative, zero sequence)"""
        features = {}
        
        # Get phase signals
        a = phases['A'] + 1j * np.zeros_like(phases['A'])  # Phase A
        b_angle = -2 * np.pi / 3  # 120 degrees lag
        c_angle = -4 * np.pi / 3  # 240 degrees lag
        
        # Approximate phase B and C from the single-phase equivalent
        # In practice, these would be the actual B and C phase measurements
        b = phases['B'] * np.exp(1j * b_angle)
        c = phases['C'] * np.exp(1j * c_angle)
        
        # Symmetrical components transformation matrix
        alpha = np.exp(1j * 2 * np.pi / 3)  # 120-degree phasor
        
        # Calculate symmetrical components for each time sample
        positive_seq = (a + alpha * b + alpha**2 * c) / 3
        negative_seq = (a + alpha**2 * b + alpha * c) / 3  
        zero_seq = (a + b + c) / 3
        
        # Extract magnitude features
        features['positive_seq_rms'] = np.sqrt(np.mean(np.abs(positive_seq)**2))
        features['negative_seq_rms'] = np.sqrt(np.mean(np.abs(negative_seq)**2))
        features['zero_seq_rms'] = np.sqrt(np.mean(np.abs(zero_seq)**2))
        
        # Sequence component ratios
        total_seq = features['positive_seq_rms'] + features['negative_seq_rms'] + features['zero_seq_rms']
        if total_seq > 1e-10:
            features['negative_seq_ratio'] = features['negative_seq_rms'] / features['positive_seq_rms'] if features['positive_seq_rms'] > 1e-10 else 0
            features['zero_seq_ratio'] = features['zero_seq_rms'] / features['positive_seq_rms'] if features['positive_seq_rms'] > 1e-10 else 0
        else:
            features['negative_seq_ratio'] = 0
            features['zero_seq_ratio'] = 0
            
        return features
        
    def _extract_cross_phase_features(self, phases: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract features describing relationships between phases"""
        features = {}
        
        phase_pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        
        for p1, p2 in phase_pairs:
            signal1 = phases[p1]
            signal2 = phases[p2]
            
            prefix = f"{p1}{p2}_"
            
            # Cross-correlation
            correlation = np.corrcoef(signal1, signal2)[0, 1]
            features[f"{prefix}correlation"] = correlation if not np.isnan(correlation) else 0
            
            # Phase difference indicators
            features[f"{prefix}rms_ratio"] = np.sqrt(np.mean(signal1**2)) / np.sqrt(np.mean(signal2**2)) if np.sqrt(np.mean(signal2**2)) > 1e-10 else 0
            features[f"{prefix}max_ratio"] = np.max(np.abs(signal1)) / np.max(np.abs(signal2)) if np.max(np.abs(signal2)) > 1e-10 else 0
            
            # Differential features
            diff_signal = signal1 - signal2
            features[f"{prefix}diff_rms"] = np.sqrt(np.mean(diff_signal**2))
            features[f"{prefix}diff_max"] = np.max(np.abs(diff_signal))
            
        return features
        
    def _extract_fault_indicators(self, phases: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract specific indicators for different fault types"""
        features = {}
        
        # Calculate 3-phase signals
        signals = [phases['A'], phases['B'], phases['C']]
        
        # Ground fault indicators
        ground_current = np.mean(signals, axis=0)  # Zero-sequence approximation
        features['ground_fault_indicator'] = np.sqrt(np.mean(ground_current**2))
        
        # Phase-to-phase fault indicators
        ab_diff = phases['A'] - phases['B']
        ac_diff = phases['A'] - phases['C']
        bc_diff = phases['B'] - phases['C']
        
        features['ab_fault_indicator'] = np.sqrt(np.mean(ab_diff**2))
        features['ac_fault_indicator'] = np.sqrt(np.mean(ac_diff**2))
        features['bc_fault_indicator'] = np.sqrt(np.mean(bc_diff**2))
        
        # Transient detection indicators
        for phase_name, signal in phases.items():
            # Rate of change indicator (for switching transients)
            rate_of_change = np.diff(signal)
            features[f'{phase_name}_max_rate_change'] = np.max(np.abs(rate_of_change))
            features[f'{phase_name}_mean_rate_change'] = np.mean(np.abs(rate_of_change))
            
            # Sudden change detector
            rolling_std = pd.Series(signal).rolling(window=50, min_periods=1).std()
            features[f'{phase_name}_std_variation'] = np.std(rolling_std.dropna())
            
        # Inrush current indicators
        for phase_name, signal in phases.items():
            # DC component (characteristic of inrush)
            dc_component = np.mean(signal)
            features[f'{phase_name}_dc_component'] = abs(dc_component)
            
            # Second harmonic content (inrush characteristic)
            N = len(signal)
            fft_vals = fft(signal)
            freqs = fftfreq(N, self.dt)[:N//2]
            freq_resolution = freqs[1] - freqs[0]
            
            second_harmonic_idx = int(100.0 / freq_resolution)  # 2 * 50Hz
            if second_harmonic_idx < len(fft_vals)//2:
                second_harmonic_mag = np.abs(fft_vals[second_harmonic_idx])
                fundamental_idx = int(50.0 / freq_resolution)
                if fundamental_idx < len(fft_vals)//2:
                    fundamental_mag = np.abs(fft_vals[fundamental_idx])
                    if fundamental_mag > 1e-10:
                        features[f'{phase_name}_second_harmonic_ratio'] = second_harmonic_mag / fundamental_mag
                    else:
                        features[f'{phase_name}_second_harmonic_ratio'] = 0
                else:
                    features[f'{phase_name}_second_harmonic_ratio'] = 0
            else:
                features[f'{phase_name}_second_harmonic_ratio'] = 0
                
        return features
        
    def _calculate_spectral_rolloff(self, freqs: np.ndarray, psd: np.ndarray, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency"""
        total_energy = np.sum(psd)
        if total_energy == 0:
            return 0
            
        cumulative_energy = np.cumsum(psd)
        rolloff_idx = np.where(cumulative_energy >= threshold * total_energy)[0]
        
        if len(rolloff_idx) > 0:
            return freqs[rolloff_idx[0]]
        else:
            return freqs[-1]
            
    def extract_features_batch(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract features from multiple files in batch.
        
        Args:
            data_dict: Dictionary mapping file paths to DataFrames
            
        Returns:
            DataFrame with features for each file
        """
        feature_list = []
        
        for file_path, df in data_dict.items():
            try:
                features = self.extract_comprehensive_features(df)
                features['file_path'] = file_path  # Add file identifier
                feature_list.append(features)
            except Exception as e:
                logger.error(f"Feature extraction failed for {file_path}: {str(e)}")
                continue
                
        if feature_list:
            feature_df = pd.DataFrame(feature_list)
            logger.info(f"Extracted features from {len(feature_list)} files, {len(feature_df.columns)-1} features per file")
            return feature_df
        else:
            logger.warning("No features extracted from any files")
            return pd.DataFrame()
            
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for analysis and model interpretation"""
        
        # This will be populated after feature extraction
        # Based on the actual feature names generated
        groups = {
            'statistical': ['mean', 'std', 'rms', 'min', 'max', 'skewness', 'kurtosis'],
            'frequency': ['fundamental', 'harmonic', 'thd', 'spectral'],
            'time_domain': ['td_', 'peak', 'zero_crossings', 'crest_factor'],
            'symmetrical': ['positive_seq', 'negative_seq', 'zero_seq'],
            'cross_phase': ['AB_', 'AC_', 'BC_', 'correlation'],
            'fault_indicators': ['fault_indicator', 'dc_component', 'rate_change']
        }
        
        return groups

# Feature selection utilities
class FeatureSelector:
    """Utilities for feature selection and dimensionality reduction"""
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        
    def select_features_mutual_info(self, X: pd.DataFrame, y: np.ndarray, k: int = 50) -> List[str]:
        """Select top k features using mutual information"""
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        
        # Handle missing values
        X_clean = X.fillna(X.mean())
        
        # Select features
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X_clean, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        self.feature_scores = selector.scores_[selected_indices]
        
        return self.selected_features
        
    def select_features_variance_threshold(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove features with low variance"""
        from sklearn.feature_selection import VarianceThreshold
        
        X_clean = X.fillna(X.mean())
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X_clean)
        
        selected_indices = selector.get_support(indices=True)
        return X.columns[selected_indices].tolist()
        
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """Get ranking of selected features by importance"""
        if self.selected_features is None or self.feature_scores is None:
            return pd.DataFrame()
            
        ranking_df = pd.DataFrame({
            'feature': self.selected_features,
            'score': self.feature_scores
        }).sort_values('score', ascending=False)
        
        return ranking_df

# Example usage
if __name__ == "__main__":
    # Example feature extraction
    config = FeatureConfig(
        sampling_frequency=10000,
        statistical_features=True,
        frequency_features=True,
        time_domain_features=True,
        symmetrical_components=True
    )
    
    extractor = GridXFeatureExtractor(config)
    
    # Create sample data (726 samples, 4 columns)
    time = np.linspace(0, 0.0725, 726)
    phase_a = np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.normal(0, 1, 726)
    phase_b = np.sin(2 * np.pi * 50 * time - 2*np.pi/3) + 0.1 * np.random.normal(0, 1, 726)
    phase_c = np.sin(2 * np.pi * 50 * time - 4*np.pi/3) + 0.1 * np.random.normal(0, 1, 726)
    
    sample_df = pd.DataFrame({
        'time': time,
        'phase_a': phase_a,
        'phase_b': phase_b,
        'phase_c': phase_c
    })
    
    # Extract features
    features = extractor.extract_comprehensive_features(sample_df)
    
    print(f"Extracted {len(features)} features:")
    for i, (key, value) in enumerate(list(features.items())[:10]):  # Show first 10
        print(f"  {key}: {value:.6f}")
    print("  ...")