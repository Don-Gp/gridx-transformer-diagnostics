# scripts/run_unified_gridx_pipeline.py
"""
GridX Unified Data Pipeline Runner
Processes both IEEE fault detection and ETT operational data
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add backend to Python path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.services.unified_data_pipeline import UnifiedGridXPipeline, UnifiedPipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gridx_unified_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def validate_datasets():
    """Validate that both datasets exist"""
    ieee_path = Path(r"C:\Users\ogbonda\documents\Gridx-Datasets\Dataset for Transformer & PAR transients\Dataset for Transformer & PAR transients\data for transformer and par")
    ett_path = Path(r"C:\Users\ogbonda\DOCUMENTS\gridx-datasets\etdataset\ETT-small")
    
    missing_datasets = []
    
    if not ieee_path.exists():
        missing_datasets.append(f"IEEE Dataset: {ieee_path}")
    
    if not ett_path.exists():
        missing_datasets.append(f"ETT Dataset: {ett_path}")
    elif not any(ett_path.glob("*.csv")):
        missing_datasets.append(f"ETT Dataset CSV files in: {ett_path}")
    
    if missing_datasets:
        print("ERROR: Missing datasets!")
        for dataset in missing_datasets:
            print(f"  Not found: {dataset}")
        return False
    
    logger.info("All datasets validated successfully")
    return True

def run_ieee_only(quick_test=False):
    """Run pipeline with IEEE data only"""
    print("\n" + "="*60)
    print("IEEE FAULT DETECTION ONLY")
    print("="*60)
    
    pipeline = UnifiedGridXPipeline()
    results = pipeline.run_unified_pipeline(
        include_ieee=True,
        include_ett=False,
        quick_test=quick_test
    )
    
    if results['status'] == 'success' and 'ieee' in results['results']:
        ieee_res = results['results']['ieee']
        print(f"IEEE Processing Complete:")
        print(f"  Samples processed: {ieee_res['samples']:,}")
        print(f"  Features extracted: {ieee_res['features']}")
        print(f"  Fault classes: {ieee_res['classes']}")
        print(f"  Class names: {', '.join(ieee_res['class_names'][:5])}{'...' if len(ieee_res['class_names']) > 5 else ''}")
    
    return results

def run_ett_only(quick_test=False):
    """Run pipeline with ETT data only"""
    print("\n" + "="*60)
    print("ETT PREDICTIVE MAINTENANCE ONLY")
    print("="*60)
    
    pipeline = UnifiedGridXPipeline()
    results = pipeline.run_unified_pipeline(
        include_ieee=False,
        include_ett=True,
        quick_test=quick_test
    )
    
    if results['status'] == 'success' and 'ett' in results['results']:
        ett_res = results['results']['ett']
        print(f"ETT Processing Complete:")
        print(f"  Samples processed: {ett_res['samples']:,}")
        print(f"  Features extracted: {ett_res['features']}")
        print(f"  Maintenance alerts: {ett_res.get('maintenance_needed', 0)}")
        print(f"  Urgency distribution: {ett_res.get('urgency_distribution', {})}")
    
    return results

def run_unified_pipeline(quick_test=False):
    """Run complete unified pipeline"""
    print("\n" + "="*60)
    print("UNIFIED GRIDX PIPELINE")
    print("="*60)
    
    pipeline = UnifiedGridXPipeline()
    results = pipeline.run_unified_pipeline(
        include_ieee=True,
        include_ett=True,
        quick_test=quick_test
    )
    
    if results['status'] == 'success':
        print(f"Unified Pipeline Complete ({results['duration_seconds']:.1f}s):")
        
        # IEEE results
        if 'ieee' in results['results']:
            ieee_res = results['results']['ieee']
            print(f"\n  IEEE Fault Detection:")
            print(f"    Samples: {ieee_res['samples']:,}")
            print(f"    Features: {ieee_res['features']}")
            print(f"    Classes: {ieee_res['classes']}")
        
        # ETT results
        if 'ett' in results['results']:
            ett_res = results['results']['ett']
            print(f"\n  ETT Predictive Maintenance:")
            print(f"    Samples: {ett_res['samples']:,}")
            print(f"    Features: {ett_res['features']}")
            print(f"    Maintenance needed: {ett_res.get('maintenance_needed', 0)}")
        
        # Unified results
        if 'unified' in results['results']:
            unified_res = results['results']['unified']
            print(f"\n  Unified Feature Space:")
            print(f"    IEEE features: {unified_res['ieee_features']}")
            print(f"    ETT features: {unified_res['ett_features']}")
            print(f"    Cross features: {unified_res['cross_features']}")
        
        # Final datasets
        if 'final' in results['results']:
            final_res = results['results']['final']
            if 'ieee_fault_detection' in final_res:
                ieee_final = final_res['ieee_fault_detection']
                print(f"\n  IEEE Model Dataset:")
                print(f"    Train: {len(ieee_final['X_train']):,}")
                print(f"    Validation: {len(ieee_final['X_val']):,}")
                print(f"    Test: {len(ieee_final['X_test']):,}")
                
            if 'ett_maintenance' in final_res:
                ett_final = final_res['ett_maintenance']
                print(f"\n  ETT Model Dataset:")
                print(f"    Train: {len(ett_final['X_train']):,}")
                print(f"    Validation: {len(ett_final['X_val']):,}")
                print(f"    Test: {len(ett_final['X_test']):,}")
    
    return results

def show_dataset_info():
    """Show information about available datasets"""
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    # Initialize pipeline to get dataset info
    pipeline = UnifiedGridXPipeline()
    
    # IEEE info
    print("\n1. IEEE DataPort - Fault Detection:")
    print(f"   Path: {pipeline.config.ieee_data_path}")
    ieee_stats = pipeline.ieee_processor.scan_dataset()
    for key, value in ieee_stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value:,}")
    
    # ETT info  
    print("\n2. ETDataset - Operational Data:")
    print(f"   Path: {pipeline.config.ett_data_path}")
    ett_stats = pipeline.ett_processor.load_ett_datasets()
    for key, value in ett_stats.items():
        if key != 'date_range':
            print(f"   {key.replace('_', ' ').title()}: {value}")
        else:
            for dataset, date_info in value.items():
                print(f"   {dataset}: {date_info['samples']:,} samples ({date_info['start']} to {date_info['end']})")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="GridX Unified Pipeline Runner")
    parser.add_argument("--mode", 
                       choices=["ieee", "ett", "unified", "info", "all"], 
                       default="unified",
                       help="Processing mode")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with small samples")
    parser.add_argument("--validate", action="store_true",
                       help="Validate datasets only")
    
    args = parser.parse_args()
    
    print("GridX Unified Transformer Diagnostic System")
    print("Multi-Dataset Pipeline Runner")
    print("=" * 50)
    
    # Validate datasets
    if args.validate:
        validate_datasets()
        return
        
    if not validate_datasets():
        return
    
    try:
        start_time = datetime.now()
        
        if args.mode == "info":
            show_dataset_info()
            
        elif args.mode == "ieee":
            results = run_ieee_only(quick_test=args.quick)
            
        elif args.mode == "ett":
            results = run_ett_only(quick_test=args.quick)
            
        elif args.mode == "unified":
            results = run_unified_pipeline(quick_test=args.quick)
            
        elif args.mode == "all":
            # Run everything
            show_dataset_info()
            
            print("\n" + "="*60)
            print("RUNNING ALL PIPELINE MODES")
            print("="*60)
            
            # IEEE only
            print("\nStep 1: IEEE Fault Detection")
            ieee_results = run_ieee_only(quick_test=args.quick)
            
            # ETT only
            print("\nStep 2: ETT Predictive Maintenance")  
            ett_results = run_ett_only(quick_test=args.quick)
            
            # Unified
            print("\nStep 3: Unified Pipeline")
            unified_results = run_unified_pipeline(quick_test=args.quick)
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "="*60)
            print("COMPLETE PIPELINE SUMMARY")
            print("="*60)
            print(f"Total execution time: {duration:.1f} seconds")
            print("All pipeline modes completed successfully!")
            print("\nReady for ML model development:")
            print("  1. IEEE fault classification models")
            print("  2. ETT predictive maintenance models")
            print("  3. Unified diagnostic system")
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        logger.info("Pipeline execution interrupted")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        
    finally:
        end_time = datetime.now()
        print(f"\nExecution completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Check 'gridx_unified_pipeline.log' for detailed logs")

if __name__ == "__main__":
    main()