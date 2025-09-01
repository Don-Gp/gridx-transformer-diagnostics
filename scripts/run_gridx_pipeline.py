# scripts/run_gridx_pipeline.py
"""
GridX Data Pipeline Runner
Complete script to execute the GridX transformer diagnostic data pipeline
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add backend to Python path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.services.data_pipeline import GridXDataPipeline, PipelineConfig
from app.utils.data_processor import GridXDataProcessor, DatasetConfig
from app.utils.feature_extractor import GridXFeatureExtractor, FeatureConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gridx_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        "./data/raw",
        "./data/processed", 
        "./data/interim",
        "./backend/app/ml_models/trained",
        "./backend/app/ml_models/preprocessing"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_dataset_analysis():
    """Run dataset analysis and generate report"""
    logger.info("Starting dataset analysis...")
    
    # Initialize data processor
    dataset_config = DatasetConfig(
        base_path=r"C:\Users\ogbonda\documents\Gridx-Datasets\Dataset for Transformer & PAR transients\Dataset for Transformer & PAR transients\data for transformer and par"
    )
    
    processor = GridXDataProcessor(dataset_config)
    
    # Scan dataset
    stats = processor.scan_dataset()
    
    # Generate analysis report
    print("\n" + "="*60)
    print("GRIDX DATASET ANALYSIS REPORT")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value:,}")
    
    # Export file registry
    registry_path = "./data/processed/file_registry.csv"
    processor.export_file_registry(registry_path)
    print(f"\nFile registry exported to: {registry_path}")
    
    return stats

def run_feature_extraction_test():
    """Test feature extraction on sample data"""
    logger.info("Testing feature extraction...")
    
    feature_config = FeatureConfig(
        sampling_frequency=10000,
        statistical_features=True,
        frequency_features=True,
        time_domain_features=True,
        symmetrical_components=True
    )
    
    extractor = GridXFeatureExtractor(feature_config)
    
    # Load a sample file for testing
    dataset_config = DatasetConfig(
        base_path=r"C:\Users\ogbonda\documents\Gridx-Datasets\Dataset for Transformer & PAR transients\Dataset for Transformer & PAR transients\data for transformer and par"
    )
    processor = GridXDataProcessor(dataset_config)
    
    # Get sample files
    processor.scan_dataset()
    sample_files = processor.get_files_by_category(limit=3)
    
    if sample_files:
        sample_file_path = sample_files[0].file_path
        logger.info(f"Testing with file: {sample_file_path}")
        
        # Load and extract features
        df = processor.load_single_file(sample_file_path)
        if df is not None:
            features = extractor.extract_comprehensive_features(df)
            
            print("\n" + "="*60)
            print("FEATURE EXTRACTION TEST")
            print("="*60)
            print(f"File: {Path(sample_file_path).name}")
            print(f"Data shape: {df.shape}")
            print(f"Features extracted: {len(features)}")
            print("\nSample Features:")
            
            # Show first 10 features
            for i, (key, value) in enumerate(list(features.items())[:10]):
                print(f"  {key}: {value:.6f}")
            print("  ...")
            
            return True
    else:
        logger.warning("No sample files found for testing")
        return False

def run_complete_pipeline(quick_test=False):
    """Run the complete GridX pipeline"""
    logger.info("Starting complete GridX pipeline...")
    
    # Initialize pipeline
    pipeline = GridXDataPipeline()
    
    # Run pipeline
    start_time = datetime.now()
    results = pipeline.run_complete_pipeline(quick_test=quick_test)
    end_time = datetime.now()
    
    # Print results
    print("\n" + "="*60)
    print("GRIDX PIPELINE EXECUTION RESULTS")
    print("="*60)
    print(f"Execution Time: {(end_time - start_time).total_seconds():.1f} seconds")
    print(f"Status: {results['status'].upper()}")
    
    if results['status'] == 'success':
        print(f"\nDataset Processing:")
        print(f"  Total samples: {results['sample_count']:,}")
        print(f"  Classes identified: {results['class_count']}")
        print(f"  Features extracted: {results['feature_count']}")
        print(f"  Features selected: {results['selected_features']}")
        
        print(f"\nData Splits:")
        for split_name, size in results['data_splits'].items():
            print(f"  {split_name}: {size:,}")
            
        print(f"\nDataset Breakdown:")
        for stat_name, count in results['dataset_stats'].items():
            print(f"  {stat_name.replace('_', ' ').title()}: {count:,}")
            
        print(f"\nPipeline State:")
        for state, status in results['pipeline_state'].items():
            status_symbol = "✓" if status else "✗"
            print(f"  {status_symbol} {state.replace('_', ' ').title()}: {status}")
            
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    return results

def validate_dataset_path():
    """Validate that the dataset path exists"""
    dataset_path = Path(r"C:\Users\ogbonda\documents\Gridx-Datasets\Dataset for Transformer & PAR transients\Dataset for Transformer & PAR transients\data for transformer and par")
    
    if not dataset_path.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        print("\nERROR: Dataset path not found!")
        print("Please verify the dataset is located at:")
        print(f"  {dataset_path}")
        print("\nOr update the path in the script.")
        return False
    
    logger.info(f"Dataset path validated: {dataset_path}")
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="GridX Data Pipeline Runner")
    parser.add_argument("--mode", choices=["analysis", "test", "pipeline", "full"], 
                       default="full", help="Execution mode")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with small sample")
    parser.add_argument("--setup", action="store_true",
                       help="Setup directories only")
    
    args = parser.parse_args()
    
    print("GridX Transformer Diagnostic System")
    print("Data Pipeline Runner")
    print("=" * 40)
    
    # Setup directories
    if args.setup:
        setup_directories()
        return
        
    # Validate dataset
    if not validate_dataset_path():
        return
        
    setup_directories()
    
    try:
        if args.mode == "analysis":
            run_dataset_analysis()
            
        elif args.mode == "test":
            run_feature_extraction_test()
            
        elif args.mode == "pipeline":
            run_complete_pipeline(quick_test=args.quick)
            
        elif args.mode == "full":
            # Run everything
            print("\nStep 1: Dataset Analysis")
            run_dataset_analysis()
            
            print("\nStep 2: Feature Extraction Test")
            run_feature_extraction_test()
            
            print("\nStep 3: Complete Pipeline")
            results = run_complete_pipeline(quick_test=args.quick)
            
            if results['status'] == 'success':
                print("\n🎉 GridX Pipeline completed successfully!")
                print("Ready for model training and deployment.")
            else:
                print("\n❌ Pipeline failed. Check logs for details.")
                
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        logger.info("Pipeline execution interrupted")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        
    finally:
        print(f"\nExecution completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Check 'gridx_pipeline.log' for detailed logs")

if __name__ == "__main__":
    main()