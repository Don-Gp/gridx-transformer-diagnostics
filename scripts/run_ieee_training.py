#!/usr/bin/env python3
"""
GridX IEEE Fault Detection Training Launcher
Runs all IEEE fault detection models and generates reports
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.ml_models.ieee_fault_models import IEEEFaultClassifier

def main():
    print("="*80)
    print("GRIDX IEEE FAULT DETECTION MODEL TRAINING")
    print("="*80)
    
    # Initialize classifier
    classifier = IEEEFaultClassifier()
    
    # Train all models
    print("\n🚀 Training Random Forest...")
    classifier.train_random_forest(tune_hyperparameters=False)
    
    print("\n🚀 Training XGBoost...")
    classifier.train_xgboost(tune_hyperparameters=False)
    
    print("\n🚀 Training CNN-LSTM...")
    classifier.train_cnn_lstm(epochs=50)
    
    print("\n🚀 Creating Ensemble...")
    classifier.create_ensemble_model()
    
    # Generate results
    print("\n📊 Generating visualizations...")
    classifier.plot_results()
    
    print("\n💾 Saving models...")
    classifier.save_models()
    
    print("\n📋 Generating summary report...")
    best_model, best_accuracy = classifier.generate_summary_report()
    
    print("\n✅ IEEE FAULT DETECTION TRAINING COMPLETED!")
    return best_model, best_accuracy

if __name__ == "__main__":
    main()