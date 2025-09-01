#!/usr/bin/env python3
"""
GridX ETT Predictive Maintenance Training Launcher
Runs all ETT predictive maintenance models and generates reports
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.ml_models.ett_predictive_models import ETTPredictiveMaintenance

def main():
    print("="*80)
    print("GRIDX ETT PREDICTIVE MAINTENANCE MODEL TRAINING")
    print("="*80)
    
    # Initialize predictor
    predictor = ETTPredictiveMaintenance()
    
    # Train all models
    print("\n🚀 Training Maintenance Classifier...")
    predictor.train_maintenance_classifier(tune_hyperparameters=False)
    
    print("\n🚀 Training Urgency Classifier...")
    predictor.train_urgency_classifier(tune_hyperparameters=False)
    
    print("\n🚀 Training TTF Regressor...")
    predictor.train_ttf_regressor(tune_hyperparameters=False)
    
    print("\n🚀 Training Anomaly Detector...")
    predictor.train_anomaly_detector()
    
    print("\n🚀 Training LSTM Predictor...")
    predictor.train_lstm_predictor(sequence_length=10, epochs=30)
    
    print("\n🚀 Creating Integrated Predictor...")
    predictor.create_integrated_predictor()
    
    # Generate results
    print("\n📊 Generating visualizations...")
    predictor.plot_results()
    
    print("\n💾 Saving models...")
    predictor.save_models()
    
    print("\n📋 Generating summary report...")
    results = predictor.generate_summary_report()
    
    print("\n✅ ETT PREDICTIVE MAINTENANCE TRAINING COMPLETED!")
    return results

if __name__ == "__main__":
    main()