from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pydantic import BaseModel

from ...services.unified_data_pipeline import UnifiedGridXPipeline, UnifiedPipelineConfig

logger = logging.getLogger(__name__)
router = APIRouter()

# Global unified pipeline instance
unified_pipeline = None

def get_unified_pipeline() -> UnifiedGridXPipeline:
    global unified_pipeline
    if unified_pipeline is None:
        unified_pipeline = UnifiedGridXPipeline()
    return unified_pipeline

class UnifiedPipelineRequest(BaseModel):
    include_ieee: bool = True
    include_ett: bool = True
    quick_test: bool = False

class DatasetIntegrationRequest(BaseModel):
    dataset_name: str
    dataset_type: str  # 'fault_detection', 'operational', 'chemical_analysis'
    feature_mapping: Dict[str, str]
    target_column: Optional[str] = None

@router.post("/run")
async def run_unified_pipeline(request: UnifiedPipelineRequest, background_tasks: BackgroundTasks):
    """Run the unified pipeline with both IEEE and ETT datasets"""
    try:
        pipeline_instance = get_unified_pipeline()
        
        def run_pipeline_task():
            result = pipeline_instance.run_unified_pipeline(
                include_ieee=request.include_ieee,
                include_ett=request.include_ett,
                quick_test=request.quick_test
            )
            logger.info(f"Unified pipeline completed with status: {result['status']}")
            
        background_tasks.add_task(run_pipeline_task)
        
        return JSONResponse(content={
            "status": "running",
            "message": "Unified pipeline started",
            "includes": {
                "ieee_fault_detection": request.include_ieee,
                "ett_predictive_maintenance": request.include_ett
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to run unified pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_unified_status():
    """Get unified pipeline status"""
    try:
        pipeline_instance = get_unified_pipeline()
        status = pipeline_instance.get_pipeline_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Failed to get unified status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/info")
async def get_datasets_info():
    """Get information about all available datasets"""
    try:
        pipeline_instance = get_unified_pipeline()
        
        # IEEE dataset info
        ieee_stats = pipeline_instance.ieee_processor.scan_dataset()
        
        # ETT dataset info
        ett_stats = pipeline_instance.ett_processor.load_ett_datasets()
        
        return JSONResponse(content={
            "ieee_dataset": ieee_stats,
            "ett_dataset": ett_stats,
            "supported_formats": [".csv", ".txt"],
            "custom_integration_available": True
        })
        
    except Exception as e:
        logger.error(f"Failed to get dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets/integrate")
async def integrate_custom_dataset(request: DatasetIntegrationRequest):
    """Integrate a custom dataset into the pipeline"""
    try:
        # This would be implemented to handle custom datasets
        # For now, return configuration for custom integration
        
        integration_config = {
            "dataset_name": request.dataset_name,
            "type": request.dataset_type,
            "feature_mapping": request.feature_mapping,
            "target": request.target_column,
            "preprocessing_steps": [
                "data_validation",
                "feature_extraction", 
                "normalization",
                "integration_with_existing_features"
            ],
            "status": "configuration_ready"
        }
        
        return JSONResponse(content=integration_config)
        
    except Exception as e:
        logger.error(f"Failed to configure custom dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))