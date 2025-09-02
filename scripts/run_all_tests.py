#!/usr/bin/env python3
"""
GridX Comprehensive Testing Launcher
Runs all tests before model training
"""
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
        print("1. python -m scripts.run_ieee_training")
        print("2. python -m scripts.run_ett_training")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please review test output and fix issues before proceeding.")
        print("Check data integrity and pipeline setup.")
    
    return success

if __name__ == "__main__":
    main()