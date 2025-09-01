#!/usr/bin/env python3
"""
GridX Comprehensive Testing Launcher
Runs all tests before model training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.ml_models.model_tests.ieee_model_tests import run_comprehensive_tests

def main():
    print("="*80)
    print("GRIDX COMPREHENSIVE TESTING SUITE")
    print("="*80)
    
    print("Running IEEE fault detection model tests...")
    print("This will validate data integrity, model training, and production readiness.")
    print("="*80)
    
    # Run all tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Data pipeline validated")
        print("✅ Model training verified") 
        print("✅ Production readiness confirmed")
        print("\n➡️  READY TO PROCEED WITH MODEL TRAINING")
        print("\nNext steps:")
        print("1. python scripts/run_ieee_training.py")
        print("2. python scripts/run_ett_training.py")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please review test output and fix issues before proceeding.")
        print("Check data integrity and pipeline setup.")
    
    return success

if __name__ == "__main__":
    main()