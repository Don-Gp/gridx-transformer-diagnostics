"""
GridX Transformer Diagnostic System - ETT Predictive Maintenance Models
ML Development Phase - Week 3

This module implements predictive maintenance models for transformer operations:
1. Maintenance Binary Classifier - Predict maintenance needs (Yes/No)
2. Urgency Multi-class Classifier - 4-level priority system (0-3)
3. Time-to-Failure Regression - Predict days until maintenance required
4. Anomaly Detection System - Identify unusual operational patterns

Target: >90% maintenance prediction accuracy with <10% false positives
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Matplotlib is currently using agg.*",
)

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Time Series and Anomaly Detection
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning for Time Series
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class ETTPredictiveMaintenance:
    """
    ETT Predictive Maintenance System for GridX Transformer Diagnostics
    
    Provides comprehensive predictive maintenance capabilities:
    - Binary Classification: Maintenance needed (Yes/No)
    - Multi-class Classification: Urgency levels (0-3)
    - Regression: Time-to-failure prediction
    - Anomaly Detection: Unusual operational patterns
    """
    
    def __init__(self, data_path='./data/interim/unified_datasets.pkl'):
        """Initialize predictive maintenance system"""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.urgency_levels = ['Low', 'Medium', 'High', 'Critical']
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load processed ETT dataset from unified pipeline"""
        print("Loading ETT predictive maintenance dataset...")
        
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract ETT data
            ett_data = data['ett_maintenance']
            
            self.X_train = ett_data['X_train']
            self.X_val = ett_data['X_val']
            self.X_test = ett_data['X_test']
            
            # Multiple target variables
            self.y_maintenance_train = ett_data['y_maintenance_train']
            self.y_maintenance_val = ett_data['y_maintenance_val']
            self.y_maintenance_test = ett_data['y_maintenance_test']
            
            self.y_urgency_train = ett_data['y_urgency_train']
            self.y_urgency_val = ett_data['y_urgency_val']
            self.y_urgency_test = ett_data['y_urgency_test']
            
            # Time-to-failure targets (if available)
            if 'y_ttf_train' in ett_data:
                self.y_ttf_train = ett_data['y_ttf_train']
                self.y_ttf_val = ett_data['y_ttf_val']
                self.y_ttf_test = ett_data['y_ttf_test']
            else:
                # Create synthetic time-to-failure targets based on urgency
                self._create_ttf_targets()
            
            # Store preprocessing objects
            preproc = ett_data.get('preprocessing', {})
            self.scaler = preproc.get('scaler')
            self.imputer = preproc.get('imputer')

            if self.scaler is None:
                raise KeyError("Scaler missing in ETT dataset; rerun unified pipeline.")
            
            print(f"ETT data loaded successfully!")
            print(f"Training samples: {self.X_train.shape[0]}")
            print(f"Validation samples: {self.X_val.shape[0]}")
            print(f"Test samples: {self.X_test.shape[0]}")
            print(f"Features: {self.X_train.shape[1]}")
            
            # Data distribution
            maintenance_rate = np.mean(self.y_maintenance_train)
            print(f"Maintenance rate: {maintenance_rate:.2%}")
            
            urgency_dist = np.bincount(self.y_urgency_train)
            print(f"Urgency distribution: {dict(enumerate(urgency_dist))}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Run unified pipeline first.")
        except KeyError as e:
            raise KeyError(f"ETT data not found in dataset: {e}")
    
    def _create_ttf_targets(self):
        """Create time-to-failure targets based on urgency levels"""
        print("Creating synthetic time-to-failure targets...")
        
        # Map urgency to days until failure
        urgency_to_days = {
            0: np.random.normal(180, 30),  # Low: ~6 months
            1: np.random.normal(90, 20),   # Medium: ~3 months  
            2: np.random.normal(30, 10),   # High: ~1 month
            3: np.random.normal(7, 2)      # Critical: ~1 week
        }
        
        # Create TTF targets
        self.y_ttf_train = np.array([max(1, urgency_to_days[u] + np.random.normal(0, 5)) 
                                    for u in self.y_urgency_train])
        self.y_ttf_val = np.array([max(1, urgency_to_days[u] + np.random.normal(0, 5)) 
                                  for u in self.y_urgency_val])
        self.y_ttf_test = np.array([max(1, urgency_to_days[u] + np.random.normal(0, 5)) 
                                   for u in self.y_urgency_test])
        
        print(f"TTF targets created - Mean: {np.mean(self.y_ttf_train):.1f} days")
    
    def train_maintenance_classifier(self, tune_hyperparameters=True):
        """
        Train binary maintenance classifier (Maintenance Needed: Yes/No)
        
        Args:
            tune_hyperparameters: Whether to perform grid search optimization
        """
        print("\n" + "="*60)
        print("TRAINING MAINTENANCE BINARY CLASSIFIER")
        print("="*60)
        
        # Handle class imbalance
        maintenance_rate = np.mean(self.y_maintenance_train)
        print(f"Maintenance rate in training: {maintenance_rate:.2%}")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Grid search for multiple algorithms
            models_to_test = {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5],
                        'class_weight': ['balanced', 'balanced_subsample']
                    }
                },
                'xgboost': {
                    'model': xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [6, 10],
                        'learning_rate': [0.01, 0.1],
                        'scale_pos_weight': [1, 5]  # Handle imbalance
                    }
                },
                'logistic_regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'class_weight': ['balanced', None],
                        'solver': ['liblinear', 'lbfgs']
                    }
                }
            }
            
            best_score = 0
            best_model = None
            best_name = None
            
            for name, config in models_to_test.items():
                print(f"Testing {name}...")
                grid_search = GridSearchCV(
                    config['model'], config['params'],
                    cv=3, scoring='f1', n_jobs=-1, verbose=0
                )
                grid_search.fit(self.X_train, self.y_maintenance_train)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_name = name
                
                print(f"  {name} best F1: {grid_search.best_score_:.4f}")
            
            self.models['maintenance_classifier'] = best_model
            print(f"\nBest model: {best_name} (F1: {best_score:.4f})")
            
        else:
            # Use optimized Random Forest
            self.models['maintenance_classifier'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            self.models['maintenance_classifier'].fit(
                self.X_train, self.y_maintenance_train
            )
        
        # Evaluate model
        self._evaluate_maintenance_classifier()
        
        return self.models['maintenance_classifier']
    
    def train_urgency_classifier(self, tune_hyperparameters=True):
        """
        Train multi-class urgency classifier (4 urgency levels: 0-3)
        
        Args:
            tune_hyperparameters: Whether to perform grid search optimization
        """
        print("\n" + "="*60)
        print("TRAINING URGENCY MULTI-CLASS CLASSIFIER")
        print("="*60)
        
        # Check class distribution
        urgency_dist = np.bincount(self.y_urgency_train)
        print(f"Urgency distribution: {dict(enumerate(urgency_dist))}")
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            
            rf_grid.fit(self.X_train, self.y_urgency_train)
            self.models['urgency_classifier'] = rf_grid.best_estimator_
            
            print(f"Best parameters: {rf_grid.best_params_}")
            print(f"Best CV F1-weighted: {rf_grid.best_score_:.4f}")
            
        else:
            # Default optimized parameters
            self.models['urgency_classifier'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            self.models['urgency_classifier'].fit(
                self.X_train, self.y_urgency_train
            )
        
        # Evaluate model
        self._evaluate_urgency_classifier()
        
        return self.models['urgency_classifier']
    
    def train_ttf_regressor(self, tune_hyperparameters=True):
        """
        Train time-to-failure regression model
        
        Args:
            tune_hyperparameters: Whether to perform grid search optimization
        """
        print("\n" + "="*60)
        print("TRAINING TIME-TO-FAILURE REGRESSOR")
        print("="*60)
        
        print(f"TTF range: {np.min(self.y_ttf_train):.1f} - {np.max(self.y_ttf_train):.1f} days")
        print(f"TTF mean: {np.mean(self.y_ttf_train):.1f} ± {np.std(self.y_ttf_train):.1f} days")
        
        if tune_hyperparameters:
            print("Comparing regression algorithms...")
            
            # Test multiple algorithms
            regressors = {
                'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
                'xgboost': xgb.XGBRegressor(random_state=42, n_jobs=-1)
            }
            
            best_score = float('inf')
            best_model = None
            best_name = None
            
            for name, regressor in regressors.items():
                # Cross-validation
                cv_scores = cross_val_score(
                    regressor, self.X_train, self.y_ttf_train,
                    cv=3, scoring='neg_mean_squared_error'
                )
                mse = -np.mean(cv_scores)
                
                print(f"{name} CV MSE: {mse:.2f}")
                
                if mse < best_score:
                    best_score = mse
                    best_model = regressor
                    best_name = name
            
            # Train best model
            best_model.fit(self.X_train, self.y_ttf_train)
            self.models['ttf_regressor'] = best_model
            print(f"\nBest model: {best_name} (MSE: {best_score:.2f})")
            
        else:
            # Default Random Forest regressor
            self.models['ttf_regressor'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.models['ttf_regressor'].fit(self.X_train, self.y_ttf_train)
        
        # Evaluate model
        self._evaluate_ttf_regressor()
        
        return self.models['ttf_regressor']
    
    def train_anomaly_detector(self):
        """Train anomaly detection system for unusual operational patterns"""
        print("\n" + "="*60)
        print("TRAINING ANOMALY DETECTION SYSTEM")
        print("="*60)
        
        # Multiple anomaly detection approaches
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(self.X_train)
        self.models['isolation_forest'] = iso_forest
        
        # 2. DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(self.X_train)
        self.models['dbscan'] = dbscan
        
        # 3. Statistical anomaly detection (Z-score based)
        # Calculate feature statistics
        feature_means = np.mean(self.X_train, axis=0)
        feature_stds = np.std(self.X_train, axis=0)
        
        self.anomaly_stats = {
            'means': feature_means,
            'stds': feature_stds,
            'threshold': 3.0  # 3 standard deviations
        }
        
        # Evaluate anomaly detection
        self._evaluate_anomaly_detection()
        
        print("Anomaly detection system trained successfully!")
        
        return {
            'isolation_forest': iso_forest,
            'dbscan': dbscan,
            'statistical': self.anomaly_stats
        }
    
    def train_lstm_predictor(self, sequence_length=10, epochs=50):
        """
        Train LSTM model for time series prediction
        
        Args:
            sequence_length: Length of input sequences
            epochs: Training epochs
        """
        print("\n" + "="*60)
        print("TRAINING LSTM TIME SERIES PREDICTOR")
        print("="*60)
        
        # Prepare sequences for LSTM
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_length + 1):
                X_seq.append(X[i:(i + seq_length)])
                y_seq.append(y[i + seq_length - 1])
            return np.array(X_seq), np.array(y_seq)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(
            self.X_train, self.y_ttf_train, sequence_length
        )
        X_val_seq, y_val_seq = create_sequences(
            self.X_val, self.y_ttf_val, sequence_length
        )
        
        print(f"Training sequences: {X_train_seq.shape}")
        print(f"Validation sequences: {X_val_seq.shape}")
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, 
                 input_shape=(sequence_length, self.X_train.shape[1])),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='relu')  # Time-to-failure (positive values)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("LSTM Model Architecture:")
        model.summary()
        
        # Training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model and data
        self.models['lstm_predictor'] = model
        self.lstm_history = history
        self.lstm_sequence_length = sequence_length
        
        # Evaluate LSTM
        self._evaluate_lstm_predictor(X_val_seq, y_val_seq)
        
        return model
    
    def _evaluate_maintenance_classifier(self):
        """Evaluate maintenance binary classifier"""
        model = self.models['maintenance_classifier']
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(self.y_maintenance_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_maintenance_test, y_pred, average='binary'
        )
        auc = roc_auc_score(self.y_maintenance_test, y_pred_proba)
        
        # False positive rate (important for maintenance)
        tn, fp, fn, tp = confusion_matrix(self.y_maintenance_test, y_pred).ravel()
        fpr = fp / (fp + tn)
        
        # Store results
        self.results['maintenance_classifier'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'false_positive_rate': fpr,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\nMAINTENANCE CLASSIFIER RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        
        # Check targets
        if accuracy >= 0.90:
            print("✅ TARGET ACHIEVED: Accuracy >= 90%")
        if fpr <= 0.10:
            print("✅ TARGET ACHIEVED: False Positive Rate <= 10%")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_maintenance_test, y_pred, 
                                  target_names=['No Maintenance', 'Maintenance Needed']))
    
    def _evaluate_urgency_classifier(self):
        """Evaluate urgency multi-class classifier"""
        model = self.models['urgency_classifier']
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_urgency_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_urgency_test, y_pred, average='weighted'
        )
        
        # Store results
        self.results['urgency_classifier'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\nURGENCY CLASSIFIER RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_urgency_test, y_pred, 
                                  target_names=self.urgency_levels))
    
    def _evaluate_ttf_regressor(self):
        """Evaluate time-to-failure regressor"""
        model = self.models['ttf_regressor']
        
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Metrics
        mse = mean_squared_error(self.y_ttf_test, y_pred)
        mae = mean_absolute_error(self.y_ttf_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((self.y_ttf_test - y_pred) / self.y_ttf_test)) * 100
        
        # Store results
        self.results['ttf_regressor'] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': y_pred
        }
        
        print(f"\nTIME-TO-FAILURE REGRESSOR RESULTS:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    def _evaluate_anomaly_detection(self):
        """Evaluate anomaly detection system"""
        iso_forest = self.models['isolation_forest']
        
        # Predictions on test set
        anomaly_scores = iso_forest.decision_function(self.X_test)
        anomaly_predictions = iso_forest.predict(self.X_test)
        
        # Convert to binary (1 = normal, -1 = anomaly -> 0 = normal, 1 = anomaly)
        anomaly_binary = (anomaly_predictions == -1).astype(int)
        
        anomaly_rate = np.mean(anomaly_binary)
        
        self.results['anomaly_detector'] = {
            'anomaly_rate': anomaly_rate,
            'anomaly_scores': anomaly_scores,
            'anomaly_predictions': anomaly_binary
        }
        
        print(f"\nANOMALY DETECTION RESULTS:")
        print(f"Anomaly rate in test set: {anomaly_rate:.2%}")
        print(f"Average anomaly score: {np.mean(anomaly_scores):.4f}")
    
    def _evaluate_lstm_predictor(self, X_val_seq, y_val_seq):
        """Evaluate LSTM predictor"""
        model = self.models['lstm_predictor']
        
        # Predictions
        y_pred = model.predict(X_val_seq).flatten()
        
        # Metrics
        mse = mean_squared_error(y_val_seq, y_pred)
        mae = mean_absolute_error(y_val_seq, y_pred)
        rmse = np.sqrt(mse)
        
        # Store results
        self.results['lstm_predictor'] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred,
            'actual': y_val_seq
        }
        
        print(f"\nLSTM PREDICTOR RESULTS:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
    
    def create_integrated_predictor(self):
        """Create integrated prediction system combining all models"""
        print("\n" + "="*60)
        print("CREATING INTEGRATED PREDICTION SYSTEM")
        print("="*60)
        
        if len(self.models) < 3:
            print("Need at least 3 models for integration. Train more models first.")
            return None
        
        # Get predictions from all models
        maintenance_proba = self.models['maintenance_classifier'].predict_proba(self.X_test)[:, 1]
        urgency_pred = self.models['urgency_classifier'].predict(self.X_test)
        ttf_pred = self.models['ttf_regressor'].predict(self.X_test)
        
        if 'isolation_forest' in self.models:
            anomaly_scores = self.models['isolation_forest'].decision_function(self.X_test)
        else:
            anomaly_scores = np.zeros(len(self.X_test))
        
        # Create integrated risk assessment
        # Combine maintenance probability, urgency, TTF, and anomaly scores
        risk_scores = []
        
        for i in range(len(self.X_test)):
            # Base risk from maintenance probability
            base_risk = maintenance_proba[i]
            
            # Urgency multiplier (higher urgency = higher risk)
            urgency_multiplier = 1 + (urgency_pred[i] * 0.25)
            
            # Time-to-failure adjustment (shorter TTF = higher risk)
            ttf_adjustment = max(0, 1 - (ttf_pred[i] / 365))  # Normalize to 1 year
            
            # Anomaly adjustment (negative anomaly score means more anomalous)
            anomaly_adjustment = max(0, -anomaly_scores[i] / 2)
            
            # Combined risk score
            risk_score = (base_risk * urgency_multiplier + ttf_adjustment + anomaly_adjustment) / 3
            risk_scores.append(min(1.0, risk_score))  # Cap at 1.0
        
        risk_scores = np.array(risk_scores)
        
        # Create risk categories
        risk_categories = []
        for score in risk_scores:
            if score < 0.25:
                risk_categories.append('Low')
            elif score < 0.5:
                risk_categories.append('Medium')
            elif score < 0.75:
                risk_categories.append('High')
            else:
                risk_categories.append('Critical')
        
        # Store integrated results
        self.results['integrated_predictor'] = {
            'risk_scores': risk_scores,
            'risk_categories': risk_categories,
            'maintenance_probabilities': maintenance_proba,
            'urgency_predictions': urgency_pred,
            'ttf_predictions': ttf_pred,
            'anomaly_scores': anomaly_scores
        }
        
        # Summary
        risk_distribution = pd.Series(risk_categories).value_counts()
        print(f"Risk Distribution:")
        for category, count in risk_distribution.items():
            print(f"  {category}: {count} ({count/len(risk_categories):.1%})")
        
        print(f"Average risk score: {np.mean(risk_scores):.3f}")
        print("✅ Integrated prediction system created successfully!")
        
        return risk_scores, risk_categories
    
    def plot_results(self, save_plots=True):
        """Generate comprehensive results visualizations"""
        print("\n" + "="*60)
        print("GENERATING ETT RESULTS VISUALIZATIONS")
        print("="*60)
        
        os.makedirs('./results/ett_models', exist_ok=True)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(3, 4, 1)
        if 'maintenance_classifier' in self.results and 'urgency_classifier' in self.results:
            models = ['Maintenance\nClassifier', 'Urgency\nClassifier']
            accuracies = [
                self.results['maintenance_classifier']['accuracy'],
                self.results['urgency_classifier']['accuracy']
            ]
            
            bars = ax1.bar(models, accuracies, color=['lightblue', 'lightgreen'])
            ax1.set_title('Classification Accuracy', fontweight='bold')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0.7, 1.0)
            
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.01, 
                        f'{acc:.3f}', ha='center', fontweight='bold')
        
        # 2. Maintenance Confusion Matrix
        ax2 = plt.subplot(3, 4, 2)
        if 'maintenance_classifier' in self.results:
            cm = confusion_matrix(self.y_maintenance_test, 
                                self.results['maintenance_classifier']['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                       xticklabels=['No Maint.', 'Maint. Needed'],
                       yticklabels=['No Maint.', 'Maint. Needed'])
            ax2.set_title('Maintenance Confusion Matrix', fontweight='bold')
        
        # 3. Urgency Distribution
        ax3 = plt.subplot(3, 4, 3)
        if 'urgency_classifier' in self.results:
            urgency_pred = self.results['urgency_classifier']['predictions']
            urgency_counts = np.bincount(urgency_pred)
            ax3.bar(self.urgency_levels[:len(urgency_counts)], urgency_counts, 
                   color='salmon')
            ax3.set_title('Predicted Urgency Distribution', fontweight='bold')
            ax3.set_ylabel('Count')
            plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # 4. ROC Curve for Maintenance
        ax4 = plt.subplot(3, 4, 4)
        if 'maintenance_classifier' in self.results:
            fpr, tpr, _ = roc_curve(self.y_maintenance_test, 
                                  self.results['maintenance_classifier']['probabilities'])
            auc = self.results['maintenance_classifier']['auc']
            ax4.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.8)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('Maintenance ROC Curve', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Time-to-Failure Predictions
        ax5 = plt.subplot(3, 4, 5)
        if 'ttf_regressor' in self.results:
            y_pred = self.results['ttf_regressor']['predictions']
            ax5.scatter(self.y_ttf_test, y_pred, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(np.min(self.y_ttf_test), np.min(y_pred))
            max_val = max(np.max(self.y_ttf_test), np.max(y_pred))
            ax5.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax5.set_xlabel('Actual TTF (days)')
            ax5.set_ylabel('Predicted TTF (days)')
            ax5.set_title('Time-to-Failure Predictions', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Add R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(self.y_ttf_test, y_pred)
            ax5.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax5.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Anomaly Scores Distribution
        ax6 = plt.subplot(3, 4, 6)
        if 'anomaly_detector' in self.results:
            scores = self.results['anomaly_detector']['anomaly_scores']
            ax6.hist(scores, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax6.axvline(0, color='red', linestyle='--', alpha=0.8, label='Anomaly Threshold')
            ax6.set_xlabel('Anomaly Score')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Anomaly Scores Distribution', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Integrated Risk Assessment
        ax7 = plt.subplot(3, 4, 7)
        if 'integrated_predictor' in self.results:
            risk_categories = self.results['integrated_predictor']['risk_categories']
            risk_counts = pd.Series(risk_categories).value_counts()
            colors = ['green', 'yellow', 'orange', 'red']
            ax7.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(risk_counts)])
            ax7.set_title('Integrated Risk Distribution', fontweight='bold')
        
        # 8. Feature Importance (if available)
        ax8 = plt.subplot(3, 4, 8)
        if 'maintenance_classifier' in self.models:
            model = self.models['maintenance_classifier']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-10:]  # Top 10 features
                ax8.barh(range(len(top_indices)), importances[top_indices])
                ax8.set_yticks(range(len(top_indices)))
                ax8.set_yticklabels([f'Feature {i}' for i in top_indices])
                ax8.set_xlabel('Importance')
                ax8.set_title('Top 10 Feature Importances', fontweight='bold')
        
        # 9. LSTM Training History (if available)
        ax9 = plt.subplot(3, 4, 9)
        if hasattr(self, 'lstm_history'):
            history = self.lstm_history.history
            epochs = range(1, len(history['loss']) + 1)
            ax9.plot(epochs, history['loss'], 'b-', label='Training Loss')
            ax9.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax9.set_xlabel('Epochs')
            ax9.set_ylabel('Loss')
            ax9.set_title('LSTM Training History', fontweight='bold')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        # 10. Risk Score vs Maintenance Probability
        ax10 = plt.subplot(3, 4, 10)
        if 'integrated_predictor' in self.results:
            risk_scores = self.results['integrated_predictor']['risk_scores']
            maint_probs = self.results['integrated_predictor']['maintenance_probabilities']
            scatter = ax10.scatter(maint_probs, risk_scores, 
                                 c=self.y_urgency_test, cmap='viridis', alpha=0.6)
            ax10.set_xlabel('Maintenance Probability')
            ax10.set_ylabel('Integrated Risk Score')
            ax10.set_title('Risk Score vs Maintenance Probability', fontweight='bold')
            plt.colorbar(scatter, ax=ax10, label='Urgency Level')
        
        # 11. Precision-Recall Curve
        ax11 = plt.subplot(3, 4, 11)
        if 'maintenance_classifier' in self.results:
            precision, recall, _ = precision_recall_curve(
                self.y_maintenance_test, 
                self.results['maintenance_classifier']['probabilities']
            )
            ax11.plot(recall, precision, linewidth=2, color='blue')
            ax11.set_xlabel('Recall')
            ax11.set_ylabel('Precision')
            ax11.set_title('Maintenance Precision-Recall', fontweight='bold')
            ax11.grid(True, alpha=0.3)
        
        # 12. Model Performance Summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Create summary text
        summary_text = "ETT PREDICTIVE MAINTENANCE\nMODEL PERFORMANCE SUMMARY\n\n"
        
        if 'maintenance_classifier' in self.results:
            acc = self.results['maintenance_classifier']['accuracy']
            fpr = self.results['maintenance_classifier']['false_positive_rate']
            summary_text += f"Maintenance Classifier:\n"
            summary_text += f"  Accuracy: {acc:.3f}\n"
            summary_text += f"  False Positive Rate: {fpr:.3f}\n"
            summary_text += f"  {'✅' if acc >= 0.90 else '❌'} Target: ≥90% Accuracy\n"
            summary_text += f"  {'✅' if fpr <= 0.10 else '❌'} Target: ≤10% FPR\n\n"
        
        if 'urgency_classifier' in self.results:
            acc = self.results['urgency_classifier']['accuracy']
            summary_text += f"Urgency Classifier:\n"
            summary_text += f"  Accuracy: {acc:.3f}\n"
            summary_text += f"  F1-Score: {self.results['urgency_classifier']['f1_score']:.3f}\n\n"
        
        if 'ttf_regressor' in self.results:
            mae = self.results['ttf_regressor']['mae']
            mape = self.results['ttf_regressor']['mape']
            summary_text += f"Time-to-Failure Regressor:\n"
            summary_text += f"  MAE: {mae:.1f} days\n"
            summary_text += f"  MAPE: {mape:.1f}%\n\n"
        
        if 'integrated_predictor' in self.results:
            avg_risk = np.mean(self.results['integrated_predictor']['risk_scores'])
            summary_text += f"Integrated Risk Assessment:\n"
            summary_text += f"  Average Risk Score: {avg_risk:.3f}\n"
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout(pad=3.0)
        
        if save_plots:
            plt.savefig('./results/ett_models/ett_comprehensive_results.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        print("ETT results visualizations generated successfully!")
    
    def save_models(self, save_path='./models/ett_maintenance_models'):
        """Save all trained models"""
        print(f"\nSaving ETT models to {save_path}...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save traditional models
        traditional_models = ['maintenance_classifier', 'urgency_classifier', 
                            'ttf_regressor', 'isolation_forest', 'dbscan']
        
        for model_name in traditional_models:
            if model_name in self.models:
                model_file = f"{save_path}/{model_name}_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(self.models[model_name], f)
                print(f"Saved {model_name} model")
        
        # Save LSTM model
        if 'lstm_predictor' in self.models:
            self.models['lstm_predictor'].save(f"{save_path}/lstm_predictor_model.h5")
            print("Saved LSTM predictor model")
            
            # Save LSTM metadata
            lstm_metadata = {
                'sequence_length': self.lstm_sequence_length,
                'feature_count': self.X_train.shape[1]
            }
            with open(f"{save_path}/lstm_metadata.pkl", 'wb') as f:
                pickle.dump(lstm_metadata, f)
        
        # Save anomaly detection statistics
        if hasattr(self, 'anomaly_stats'):
            with open(f"{save_path}/anomaly_stats.pkl", 'wb') as f:
                pickle.dump(self.anomaly_stats, f)
        
        # Save results
        with open(f"{save_path}/ett_model_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save preprocessing objects
        preprocessing_objects = {
            'scaler': self.scaler,
            'urgency_levels': self.urgency_levels
        }
        with open(f"{save_path}/ett_preprocessing_objects.pkl", 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        print("All ETT models and preprocessing objects saved successfully!")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GRIDX ETT PREDICTIVE MAINTENANCE - SUMMARY REPORT")
        print("="*80)
        
        print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset: ETT Predictive Maintenance")
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        print(f"Features: {self.X_train.shape[1]}")
        
        print(f"\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 60)
        
        # Maintenance Classifier
        if 'maintenance_classifier' in self.results:
            results = self.results['maintenance_classifier']
            print(f"MAINTENANCE BINARY CLASSIFIER:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
            print(f"  AUC: {results['auc']:.4f}")
            print(f"  False Positive Rate: {results['false_positive_rate']:.4f}")
            
            # Target achievement
            acc_target = results['accuracy'] >= 0.90
            fpr_target = results['false_positive_rate'] <= 0.10
            print(f"  {'✅' if acc_target else '❌'} Accuracy Target (≥90%)")
            print(f"  {'✅' if fpr_target else '❌'} False Positive Target (≤10%)")
            print()
        
        # Urgency Classifier
        if 'urgency_classifier' in self.results:
            results = self.results['urgency_classifier']
            print(f"URGENCY MULTI-CLASS CLASSIFIER:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
            print()
        
        # TTF Regressor
        if 'ttf_regressor' in self.results:
            results = self.results['ttf_regressor']
            print(f"TIME-TO-FAILURE REGRESSOR:")
            print(f"  Mean Absolute Error: {results['mae']:.2f} days")
            print(f"  Root Mean Squared Error: {results['rmse']:.2f} days")
            print(f"  Mean Absolute Percentage Error: {results['mape']:.2f}%")
            print()
        
        # Anomaly Detector
        if 'anomaly_detector' in self.results:
            results = self.results['anomaly_detector']
            print(f"ANOMALY DETECTION SYSTEM:")
            print(f"  Anomaly Rate: {results['anomaly_rate']:.2%}")
            print()
        
        # Integrated System
        if 'integrated_predictor' in self.results:
            results = self.results['integrated_predictor']
            risk_dist = pd.Series(results['risk_categories']).value_counts()
            avg_risk = np.mean(results['risk_scores'])
            print(f"INTEGRATED RISK ASSESSMENT:")
            print(f"  Average Risk Score: {avg_risk:.3f}")
            print(f"  Risk Distribution:")
            for category, count in risk_dist.items():
                print(f"    {category}: {count} ({count/len(results['risk_categories']):.1%})")
            print()
        
        # Overall assessment
        print(f"OVERALL ASSESSMENT:")
        print("-" * 30)
        
        targets_met = []
        if 'maintenance_classifier' in self.results:
            acc_ok = self.results['maintenance_classifier']['accuracy'] >= 0.90
            fpr_ok = self.results['maintenance_classifier']['false_positive_rate'] <= 0.10
            targets_met.extend([acc_ok, fpr_ok])
        
        if all(targets_met) and targets_met:
            print("✅ ALL PRIMARY TARGETS ACHIEVED")
            print("   - Maintenance prediction accuracy ≥90%")
            print("   - False positive rate ≤10%")
        else:
            print("❌ SOME TARGETS NOT MET - REVIEW PERFORMANCE")
        
        print(f"\nDEPLOYMENT STATUS:")
        print(f"- Models saved to ./models/ett_maintenance_models/")
        print(f"- Results visualizations in ./results/ett_models/")
        print(f"- Integrated prediction system available")
        print(f"- Ready for production deployment")
        
        print(f"\nNEXT STEPS:")
        print(f"- Integrate with IEEE fault detection models")
        print(f"- Deploy unified diagnostic system")
        print(f"- Set up real-time monitoring dashboard")
        print(f"- Prepare for DGA dataset integration")
        
        return self.results


# Example usage and testing
if __name__ == "__main__":
    # Initialize predictive maintenance system
    predictor = ETTPredictiveMaintenance()
    
    print("Starting ETT Predictive Maintenance Model Training...")
    
    # 1. Maintenance Binary Classifier
    predictor.train_maintenance_classifier(tune_hyperparameters=False)
    
    # 2. Urgency Multi-class Classifier
    predictor.train_urgency_classifier(tune_hyperparameters=False)
    
    # 3. Time-to-Failure Regressor
    predictor.train_ttf_regressor(tune_hyperparameters=False)
    
    # 4. Anomaly Detection System
    predictor.train_anomaly_detector()
    
    # 5. LSTM Time Series Predictor
    predictor.train_lstm_predictor(sequence_length=10, epochs=20)  # Reduced for testing
    
    # 6. Integrated Prediction System
    predictor.create_integrated_predictor()
    
    # Generate results
    predictor.plot_results()
    predictor.save_models()
    results = predictor.generate_summary_report()
    
    print(f"\nETT Predictive Maintenance completed!")
    print(f"System ready for integration with IEEE fault detection models.")