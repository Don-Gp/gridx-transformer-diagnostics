"""
GridX Transformer Diagnostic System - IEEE Fault Classification Models
ML Development Phase - Week 3

This module implements three ML models for transformer fault detection:
1. Random Forest Classifier - Robust ensemble method
2. XGBoost Classifier - Gradient boosting optimization  
3. CNN-LSTM Model - Deep learning for time-series patterns

Target: >95% accuracy across 17 fault classes
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Matplotlib is currently using agg.*",
)

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling1D,
    Flatten,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class IEEEFaultClassifier:
    """
    IEEE Fault Classification System for GridX Transformer Diagnostics
    
    Implements three complementary models for comprehensive fault detection:
    - Random Forest: Robust feature-based classification
    - XGBoost: Advanced gradient boosting with regularization
    - CNN-LSTM: Deep learning for temporal pattern recognition
    """
    
    def __init__(self, data_path='./data/interim/unified_datasets.pkl'):
        """Initialize classifier with data loading and basic setup"""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.expected_features = 50  # Feature count after SelectKBest
        # Get fault classes dynamically from the data
        if hasattr(self, 'class_mapping'):
            # Create fault classes list from actual class mapping
            self.fault_classes = [''] * len(self.class_mapping)
            for class_name, class_id in self.class_mapping.items():
                self.fault_classes[class_id] = class_name
        else:
            # Fallback to default list
            self.fault_classes = [f'Class_{i}' for i in range(19)]
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load processed IEEE dataset from unified pipeline"""
        print("Loading IEEE fault detection dataset...")
        
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract IEEE data
            ieee_data = data['ieee_fault_detection']
            
            self.X_train = ieee_data['X_train']
            self.X_val = ieee_data['X_val']
            self.X_test = ieee_data['X_test']
            self.y_train = ieee_data['y_train']
            self.y_val = ieee_data['y_val']
            self.y_test = ieee_data['y_test']
            
            # Store preprocessing objects for inference
            preprocessing = ieee_data['preprocessing']
            self.scaler = preprocessing['scaler']
            self.feature_selector = preprocessing.get('feature_selector', None)

            if self.X_train.shape[1] != self.expected_features:
                raise ValueError(
                    f"Expected {self.expected_features} features, got {self.X_train.shape[1]}"
                )
            
            print(f"Data loaded successfully!")
            print(f"Training samples: {self.X_train.shape[0]}")
            print(f"Validation samples: {self.X_val.shape[0]}")
            print(f"Test samples: {self.X_test.shape[0]}")
            print(f"Features: {self.expected_features}")
            print(f"Classes: {len(np.unique(self.y_train))}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Run unified pipeline first.")
        except KeyError as e:
            raise KeyError(f"IEEE data not found in dataset: {e}")
    
    def train_random_forest(self, tune_hyperparameters=True):
        """
        Train Random Forest Classifier with hyperparameter tuning
        
        Args:
            tune_hyperparameters: Whether to perform grid search optimization
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Grid search parameters
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            # Initialize base model
            rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            # Grid search with cross-validation
            rf_grid = GridSearchCV(
                estimator=rf_base,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit on training data
            rf_grid.fit(self.X_train, self.y_train)
            
            # Best model
            self.models['random_forest'] = rf_grid.best_estimator_
            print(f"Best parameters: {rf_grid.best_params_}")
            print(f"Best CV score: {rf_grid.best_score_:.4f}")
            
        else:
            # Default optimized parameters
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            self.models['random_forest'].fit(self.X_train, self.y_train)
        
        # Evaluate model
        self._evaluate_model('random_forest')
        
        return self.models['random_forest']

    def train_random_forest_cv(self, param_grid=None, n_splits=5):
        """Train Random Forest using Stratified K-Fold cross-validation.

        Args:
            param_grid (dict | None): Hyperparameters to evaluate. If ``None`` a
                small default grid is used.
            n_splits (int): Number of cross-validation folds.

        Returns:
            RandomForestClassifier: model trained on the full training set
            using the best parameters from cross-validation.
        """
        print("\n" + "=" * 60)
        print("RANDOM FOREST CROSS-VALIDATION")
        print("=" * 60)

        if param_grid is None:
            param_grid = {
                'n_estimators': [200],
                'max_depth': [20],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'max_features': ['sqrt']
            }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_params = None
        best_score = -np.inf
        best_mean = None
        best_var = None

        for params in ParameterGrid(param_grid):
            fold_metrics = []
            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                model = RandomForestClassifier(
                    **params, random_state=42, n_jobs=-1
                )
                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_val, y_pred, average='weighted'
                )
                fold_metrics.append([acc, prec, rec, f1])

            metrics_arr = np.array(fold_metrics)
            metrics_mean = metrics_arr.mean(axis=0)
            metrics_var = metrics_arr.var(axis=0, ddof=1)

            print(
                f"Params {params} -> Accuracy {metrics_mean[0]:.4f}±{np.sqrt(metrics_var[0]):.4f}"
            )

            if metrics_mean[0] > best_score:
                best_score = metrics_mean[0]
                best_params = params
                best_mean = metrics_mean
                best_var = metrics_var

        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        self.results['random_forest_cv'] = {
            'params': best_params,
            'mean': dict(zip(metric_names, best_mean)),
            'variance': dict(zip(metric_names, best_var)),
        }

        print(f"Best params: {best_params}")
        for name, mean_val in self.results['random_forest_cv']['mean'].items():
            var_val = self.results['random_forest_cv']['variance'][name]
            print(f"{name.title()}: {mean_val:.4f}±{np.sqrt(var_val):.4f}")

        # Train final model on full training data
        self.models['random_forest'] = RandomForestClassifier(
            **best_params, random_state=42, n_jobs=-1
        )
        self.models['random_forest'].fit(self.X_train, self.y_train)

        self._evaluate_model('random_forest')

        return self.models['random_forest']
    
    def train_xgboost(self, tune_hyperparameters=True):
        """
        Train XGBoost Classifier with hyperparameter tuning
        
        Args:
            tune_hyperparameters: Whether to perform grid search optimization
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST CLASSIFIER")
        print("="*60)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            
            # Grid search parameters
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 10, 15],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Initialize base model
            xgb_base = xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            # Grid search with cross-validation
            xgb_grid = GridSearchCV(
                estimator=xgb_base,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit on training data
            xgb_grid.fit(self.X_train, self.y_train)
            
            # Best model
            self.models['xgboost'] = xgb_grid.best_estimator_
            print(f"Best parameters: {xgb_grid.best_params_}")
            print(f"Best CV score: {xgb_grid.best_score_:.4f}")
            
        else:
            # Default optimized parameters
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            # Train model with early stopping
            self.models['xgboost'].fit(self.X_train, self.y_train)
        
        # Evaluate model
        self._evaluate_model('xgboost')
        
        return self.models['xgboost']
    
    def train_xgboost_cv(self, param_grid=None, n_splits=5):
        """Train XGBoost using Stratified K-Fold cross-validation."""
        print("\n" + "=" * 60)
        print("XGBOOST CROSS-VALIDATION")
        print("=" * 60)

        if param_grid is None:
            param_grid = {
                'n_estimators': [200],
                'max_depth': [10],
                'learning_rate': [0.1],
                'subsample': [0.9],
                'colsample_bytree': [0.9],
            }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_params = None
        best_score = -np.inf
        best_mean = None
        best_var = None

        for params in ParameterGrid(param_grid):
            fold_metrics = []
            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                model = xgb.XGBClassifier(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='mlogloss'
                )
                model.fit(X_tr, y_tr)

                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_val, y_pred, average='weighted'
                )
                fold_metrics.append([acc, prec, rec, f1])

            metrics_arr = np.array(fold_metrics)
            metrics_mean = metrics_arr.mean(axis=0)
            metrics_var = metrics_arr.var(axis=0, ddof=1)

            print(
                f"Params {params} -> Accuracy {metrics_mean[0]:.4f}±{np.sqrt(metrics_var[0]):.4f}"
            )

            if metrics_mean[0] > best_score:
                best_score = metrics_mean[0]
                best_params = params
                best_mean = metrics_mean
                best_var = metrics_var

        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        self.results['xgboost_cv'] = {
            'params': best_params,
            'mean': dict(zip(metric_names, best_mean)),
            'variance': dict(zip(metric_names, best_var)),
        }

        print(f"Best params: {best_params}")
        for name, mean_val in self.results['xgboost_cv']['mean'].items():
            var_val = self.results['xgboost_cv']['variance'][name]
            print(f"{name.title()}: {mean_val:.4f}±{np.sqrt(var_val):.4f}")

        # Train final model
        self.models['xgboost'] = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        self.models['xgboost'].fit(self.X_train, self.y_train)

        self._evaluate_model('xgboost')
        return self.models['xgboost']
    
    def train_cnn_lstm(
    self,
    epochs=50,
    batch_size=32,
    conv_filters=None,
    lstm_units=None,
    use_early_stopping=True,
    early_stopping_patience=5,
):
        """
        Train CNN-LSTM model for temporal pattern recognition
        
        Args:
            epochs: Maximum training epochs
            batch_size: Training batch size
            conv_filters: List of Conv1D filter sizes. Each entry creates a
                Conv1D layer. Defaults to [16].
            lstm_units: List of LSTM layer sizes. Each entry creates an LSTM
                layer. Defaults to [25].
            use_early_stopping: Whether to enable early stopping.
            early_stopping_patience: Patience for early stopping in epochs. 
        """
        print("\n" + "=" * 60)
        print("TRAINING CNN-LSTM MODEL")
        print("=" * 60)
        
        # Prepare data for CNN-LSTM (reshape for time-series)
        # Assuming features represent time-series patterns
        n_timesteps = 10
        n_features_per_step = int(self.X_train.shape[1] / n_timesteps)
        
        # Reshape data for CNN-LSTM input
        X_train_reshaped = (
            self.X_train.iloc[:, : n_timesteps * n_features_per_step]
            .to_numpy()
            .reshape(-1, n_timesteps, n_features_per_step)
        )
        X_val_reshaped = (
            self.X_val.iloc[:, : n_timesteps * n_features_per_step]
            .to_numpy()
            .reshape(-1, n_timesteps, n_features_per_step)
        )
        X_test_reshaped = (
            self.X_test.iloc[:, : n_timesteps * n_features_per_step]
            .to_numpy()
            .reshape(-1, n_timesteps, n_features_per_step)
        )
        
         # Verify shapes before training
        print(X_train_reshaped.shape)

        # Convert labels to categorical
        n_classes = len(np.unique(self.y_train))
        y_train_cat = to_categorical(self.y_train, n_classes)
        y_val_cat = to_categorical(self.y_val, n_classes)
        y_test_cat = to_categorical(self.y_test, n_classes)
        
        print(f"Output classes: {n_classes}")

        if conv_filters is None:
            conv_filters = [16]
        if lstm_units is None:
            lstm_units = [25]

        layers = []
        for i, filters in enumerate(conv_filters):
            if i == 0:
                layers.append(
                    Conv1D(
                        filters=filters,
                        kernel_size=3,
                        activation="relu",
                        input_shape=(n_timesteps, n_features_per_step),
                    )
                )
            else:
                layers.append(
                    Conv1D(filters=filters, kernel_size=3, activation="relu")
                )

        layers.extend([MaxPooling1D(pool_size=2), BatchNormalization(), Dropout(0.3)])

        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            layers.append(LSTM(units, return_sequences=return_sequences))
            layers.append(Dropout(0.3))

        layers.extend(
            [
                Dense(100, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(n_classes, activation="softmax"),
            ]
        )
        
        # Build CNN-LSTM architecture
        model = Sequential(layers)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model architecture:")
        model.summary()
    
        callbacks = [

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1,
            )
        
        ]

        if use_early_stopping:
            callbacks.insert(
                0,
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
            )
            
        # Train model
        print("Training CNN-LSTM model...")
        history = model.fit(
            X_train_reshaped, y_train_cat,
            validation_data=(X_val_reshaped, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
       

        # Store model and training data
        self.models['cnn_lstm'] = model
        self.cnn_lstm_history = history
        self.cnn_lstm_data = {
            'X_test_reshaped': X_test_reshaped,
            'y_test_cat': y_test_cat,
            'n_timesteps': n_timesteps,
            'n_features_per_step': n_features_per_step
        }
        
        # Evaluate model
        self._evaluate_cnn_lstm_model()
        
        return model
    
    def train_cnn_lstm_cv(self, epochs=50, batch_size=32, n_splits=5):
        """Cross-validate CNN-LSTM model with Stratified K-Fold."""
        print("\n" + "=" * 60)
        print("CNN-LSTM CROSS-VALIDATION")
        print("=" * 60)

        n_timesteps = 10
        n_features_per_step = int(self.X_train.shape[1] / n_timesteps)

        X = (
            self.X_train.iloc[:, : n_timesteps * n_features_per_step]
            .to_numpy()
            .reshape(-1, n_timesteps, n_features_per_step)
        )
        y = self.y_train.to_numpy()

        n_classes = len(np.unique(y))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_metrics = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            y_tr_cat = to_categorical(y_tr, n_classes)
            y_val_cat = to_categorical(y_val, n_classes)

            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu',
                       input_shape=(n_timesteps, n_features_per_step)),
                Conv1D(filters=32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(50, return_sequences=True),
                Dropout(0.3),
                LSTM(25),
                Dropout(0.3),
                Dense(100, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(n_classes, activation='softmax'),
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'],
            )

            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0,
                )
            ]

            model.fit(
                X_tr,
                y_tr_cat,
                validation_data=(X_val, y_val_cat),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            y_pred = np.argmax(model.predict(X_val), axis=1)
            acc = accuracy_score(y_val, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='weighted'
            )
            fold_metrics.append([acc, prec, rec, f1])

        metrics_arr = np.array(fold_metrics)
        metrics_mean = metrics_arr.mean(axis=0)
        metrics_var = metrics_arr.var(axis=0, ddof=1)

        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        self.results['cnn_lstm_cv'] = {
            'mean': dict(zip(metric_names, metrics_mean)),
            'variance': dict(zip(metric_names, metrics_var)),
        }

        for name, mean_val in self.results['cnn_lstm_cv']['mean'].items():
            var_val = self.results['cnn_lstm_cv']['variance'][name]
            print(f"{name.title()}: {mean_val:.4f}±{np.sqrt(var_val):.4f}")

        # After CV, train final model on full training data
        return self.train_cnn_lstm(epochs=epochs, batch_size=batch_size)
    
    def _evaluate_model(self, model_name):
        """Evaluate traditional ML models (RF, XGBoost)"""
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\n{model_name.upper()} RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=self.fault_classes[:len(np.unique(self.y_test))]))
        
    def _evaluate_cnn_lstm_model(self):
        """Evaluate CNN-LSTM model"""
        model = self.models['cnn_lstm']
        test_data = self.cnn_lstm_data
        
        # Predictions
        y_pred_proba = model.predict(test_data['X_test_reshaped'])
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        # Store results
        self.results['cnn_lstm'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\nCNN-LSTM RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=self.fault_classes[:len(np.unique(self.y_test))]))
    
    def create_ensemble_model(self):
        """Create voting ensemble from all trained models"""
        print("\n" + "="*60)
        print("CREATING ENSEMBLE MODEL")
        print("="*60)
        
        if len(self.models) < 2:
            print("Need at least 2 models for ensemble. Train more models first.")
            return None
        
        # Get predictions from all models
        predictions = {}
        
        # Traditional models
        for model_name in ['random_forest', 'xgboost']:
            if model_name in self.models:
                pred_proba = self.models[model_name].predict_proba(self.X_test)
                predictions[model_name] = pred_proba
        
        # CNN-LSTM model
        if 'cnn_lstm' in self.models:
            test_data = self.cnn_lstm_data
            pred_proba = self.models['cnn_lstm'].predict(test_data['X_test_reshaped'])
            predictions['cnn_lstm'] = pred_proba
        
        # Ensemble prediction (average probabilities)
        ensemble_proba = np.mean(list(predictions.values()), axis=0)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        # Evaluate ensemble
        accuracy = accuracy_score(self.y_test, ensemble_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, ensemble_pred, average='weighted'
        )
        
        # Store ensemble results
        self.results['ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        
        print(f"ENSEMBLE RESULTS:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, ensemble_pred, 
                                  target_names=self.fault_classes[:len(np.unique(self.y_test))]))
        
        return ensemble_pred
    
    def plot_results(self, save_plots=True, ylim=None):
        """Generate comprehensive results visualizations

        Args:
            save_plots (bool): whether to save generated plots to disk.
            ylim (tuple | None): optional y-axis limits override as (min, max).
                When None, the limits scale dynamically based on the data.
        """
        print("\n" + "="*60)
        print("GENERATING RESULTS VISUALIZATIONS")
        print("="*60)
        
        # Create results directory
        os.makedirs('./results/ieee_models', exist_ok=True)
        
        # 1. Model Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        ax1.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold'][:len(models)])
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        if ylim is not None:
            ax1.set_ylim(*ylim)
        else:
            y_min = 0.0
            y_max = max(accuracies) * 1.1
            ax1.set_ylim(y_min, min(1.0, y_max))
        for i, acc in enumerate(accuracies):
            ax1.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
        
        # F1-Score comparison
        f1_scores = [self.results[model]['f1_score'] for model in models]
        ax2.bar(models, f1_scores, color=['skyblue', 'lightgreen', 'salmon', 'gold'][:len(models)])
        ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Score')
        if ylim is not None:
            ax2.set_ylim(*ylim)
        else:
            y_min = 0.0
            y_max = max(f1_scores) * 1.1
            ax2.set_ylim(y_min, min(1.0, y_max))
        for i, f1 in enumerate(f1_scores):
            ax2.text(i, f1 + 0.01, f'{f1:.3f}', ha='center', fontweight='bold')
        
        # Confusion matrix for best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        cm = confusion_matrix(self.y_test, self.results[best_model]['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'Confusion Matrix - {best_model.title()}', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Training history (if CNN-LSTM was trained)
        if 'cnn_lstm' in self.models and hasattr(self, 'cnn_lstm_history'):
            history = self.cnn_lstm_history.history
            epochs = range(1, len(history['accuracy']) + 1)
            
            ax4.plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy')
            ax4.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
            ax4.set_title('CNN-LSTM Training History', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('Accuracy')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, 'CNN-LSTM Training\nHistory Not Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('./results/ieee_models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Results visualizations generated successfully!")
    
    def save_models(self, save_path='./models/ieee_fault_models'):
        """Save all trained models"""
        print(f"\nSaving models to {save_path}...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save traditional models
        for model_name in ['random_forest', 'xgboost']:
            if model_name in self.models:
                model_file = f"{save_path}/{model_name}_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(self.models[model_name], f)
                print(f"Saved {model_name} model")
        
        # Save CNN-LSTM model
        if 'cnn_lstm' in self.models:
            model_path_keras = f"{save_path}/cnn_lstm_model.keras"
            self.models['cnn_lstm'].save(model_path_keras, save_format="keras")
            print("Saved CNN-LSTM model")
            
            # Save CNN-LSTM metadata
            with open(f"{save_path}/cnn_lstm_metadata.pkl", 'wb') as f:
                pickle.dump(self.cnn_lstm_data, f)
        
        # Save results
        with open(f"{save_path}/model_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save preprocessing objects
        preprocessing_objects = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'fault_classes': self.fault_classes
        }
        with open(f"{save_path}/preprocessing_objects.pkl", 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        print("All models and preprocessing objects saved successfully!")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GRIDX IEEE FAULT CLASSIFICATION - SUMMARY REPORT")
        print("="*80)
        
        print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        num_classes = len(np.unique(self.y_train))
        print(f"Dataset: IEEE Fault Detection ({num_classes} classes)")
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        print(f"Features: {self.X_train.shape[1]}")
        
        print(f"\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        for model_name, results in self.results.items():
            print(f"{model_name.upper():<15} | Accuracy: {results['accuracy']:.4f} | "
                  f"F1-Score: {results['f1_score']:.4f}")
        
        # Best model
        best_model = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        best_accuracy = self.results[best_model]['accuracy']
        
        print(f"\nBEST PERFORMING MODEL: {best_model.upper()}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        # Target achievement
        target_accuracy = 0.95
        if best_accuracy >= target_accuracy:
            print(f"✅ TARGET ACHIEVED: {best_accuracy:.4f} >= {target_accuracy}")
        else:
            print(f"❌ TARGET NOT MET: {best_accuracy:.4f} < {target_accuracy}")
            print(f"   Gap to target: {target_accuracy - best_accuracy:.4f}")
        
        print(f"\nNext steps:")
        print(f"- Models saved to ./models/ieee_fault_models/")
        print(f"- Results visualizations in ./results/ieee_models/")
        print(f"- Ready for ETT predictive maintenance models")
        print(f"- Ensemble model available for deployment")
        
        return best_model, best_accuracy


# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = IEEEFaultClassifier()
    
    # Train all models
    print("Starting IEEE Fault Classification Model Training...")
    
    # 1. Random Forest
    classifier.train_random_forest(tune_hyperparameters=False)  # Set True for full tuning
    
    # 2. XGBoost
    classifier.train_xgboost(tune_hyperparameters=False)  # Set True for full tuning
    
    # 3. CNN-LSTM
    classifier.train_cnn_lstm(epochs=50)  # Reduced for testing
    
    # 4. Ensemble
    classifier.create_ensemble_model()
    
    # Generate results
    classifier.plot_results()
    classifier.save_models()
    best_model, best_accuracy = classifier.generate_summary_report()
    
    print(f"\nIEEE Fault Classification completed!")
    print(f"Best model: {best_model} with {best_accuracy:.4f} accuracy")