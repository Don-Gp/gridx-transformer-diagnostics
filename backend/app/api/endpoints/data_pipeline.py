# backend/app/api/endpoints/data_pipeline.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import asyncio
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io

from ...services.data_pipeline import GridXDataPipeline, PipelineConfig
from ...utils.data_processor import GridXDataProcessor, DatasetConfig
from ...utils.feature_extractor import GridXFeatureExtractor, FeatureConfig

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class PipelineStatus(BaseModel):
    status: str
    dataset_scanned: bool
    features_extracted: bool
    data_preprocessed: bool
    model_ready: bool
    total_files: int
    feature_count: int
    class_count: int

class DatasetStats(BaseModel):
    total_files: int
    internal_faults: int
    transient_disturbances: int
    ispar_exciting: int
    ispar_series: int
    power_transformer: int

class PipelineResult(BaseModel):
    status: str
    duration_seconds: Optional[float] = None
    dataset_stats: Optional[Dict[str, int]] = None
    feature_count: Optional[int] = None
    selected_features: Optional[int] = None
    class_count: Optional[int] = None
    sample_count: Optional[int] = None
    error: Optional[str] = None

class FeatureExtractionRequest(BaseModel):
    file_data: Dict[str, List[List[float]]]  # {filename: [[time, phase_a, phase_b, phase_c], ...]}
    extract_all_features: bool = True

class DiagnosticRequest(BaseModel):
    time_series: List[List[float]]  # [[time, phase_a, phase_b, phase_c], ...]
    sampling_frequency: Optional[int] = 10000
    return_features: bool = False

router = APIRouter()

# Global pipeline instance
pipeline = None

def get_pipeline() -> GridXDataPipeline:
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = GridXDataPipeline()
    return pipeline

@router.get("/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """Get current status of the data pipeline"""
    try:
        pipeline_instance = get_pipeline()
        status = pipeline_instance.get_pipeline_status()
        
        return PipelineStatus(
            status="active",
            dataset_scanned=status['pipeline_state']['dataset_scanned'],
            features_extracted=status['pipeline_state']['features_extracted'],
            data_preprocessed=status['pipeline_state']['data_preprocessed'],
            model_ready=status['pipeline_state']['model_ready'],
            total_files=status.get('dataset_stats', {}).get('total_files', 0),
            feature_count=status['feature_count'],
            class_count=status['class_count']
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/initialize", response_model=PipelineResult)
async def initialize_pipeline(background_tasks: BackgroundTasks):
    """Initialize the data pipeline and scan dataset"""
    try:
        pipeline_instance = get_pipeline()
        
        # Run dataset scan in background
        def scan_dataset():
            stats = pipeline_instance.data_processor.scan_dataset()
            pipeline_instance.dataset_stats = stats
            pipeline_instance.pipeline_state['dataset_scanned'] = True
            
        background_tasks.add_task(scan_dataset)
        
        return PipelineResult(
            status="initializing",
            duration_seconds=None,
            dataset_stats=None,
            feature_count=0,
            selected_features=0,
            class_count=0,
            sample_count=0
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/run", response_model=PipelineResult)
async def run_pipeline(background_tasks: BackgroundTasks, quick_test: bool = False):
    """Run the complete data pipeline"""
    try:
        pipeline_instance = get_pipeline()
        
        def run_pipeline_task():
            result = pipeline_instance.run_complete_pipeline(quick_test=quick_test)
            logger.info(f"Pipeline completed with status: {result['status']}")
            
        background_tasks.add_task(run_pipeline_task)
        
        return PipelineResult(
            status="running",
            duration_seconds=None,
            dataset_stats=None,
            feature_count=0,
            selected_features=0,
            class_count=0,
            sample_count=0
        )
    except Exception as e:
        logger.error(f"Failed to run pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/stats", response_model=DatasetStats)
async def get_dataset_stats():
    """Get dataset statistics"""
    try:
        pipeline_instance = get_pipeline()
        
        if not pipeline_instance.pipeline_state['dataset_scanned']:
            raise HTTPException(status_code=400, detail="Dataset not scanned yet. Please initialize pipeline first.")
            
        stats = pipeline_instance.dataset_stats
        
        return DatasetStats(
            total_files=stats.get('total_files', 0),
            internal_faults=stats.get('internal_faults', 0),
            transient_disturbances=stats.get('transient_disturbances', 0),
            ispar_exciting=stats.get('ispar_exciting', 0),
            ispar_series=stats.get('ispar_series', 0),
            power_transformer=stats.get('power_transformer', 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/features/extract")
async def extract_features_from_data(request: FeatureExtractionRequest):
    """Extract features from provided time series data"""
    try:
        pipeline_instance = get_pipeline()
        extractor = pipeline_instance.feature_extractor
        
        results = {}
        
        for filename, time_series_data in request.file_data.items():
            # Convert to DataFrame
            df = pd.DataFrame(time_series_data, columns=['time', 'phase_a', 'phase_b', 'phase_c'])
            
            # Extract features
            features = extractor.extract_comprehensive_features(df)
            results[filename] = features
            
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/diagnostic/analyze")
async def analyze_transformer_data(request: DiagnosticRequest):
    """Analyze transformer data for fault detection"""
    try:
        pipeline_instance = get_pipeline()
        
        if not pipeline_instance.pipeline_state['model_ready']:
            raise HTTPException(status_code=400, detail="Pipeline not ready. Please run complete pipeline first.")
            
        # Convert to DataFrame
        df = pd.DataFrame(request.time_series, columns=['time', 'phase_a', 'phase_b', 'phase_c'])
        
        # Extract features
        features = pipeline_instance.feature_extractor.extract_comprehensive_features(df)
        
        # For now, return feature analysis (model integration will be added later)
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "data_quality": {
                "samples": len(df),
                "duration_ms": (df['time'].max() - df['time'].min()) * 1000,
                "sampling_rate": len(df) / (df['time'].max() - df['time'].min()) if df['time'].max() > df['time'].min() else 0
            },
            "signal_characteristics": {
                "phase_a_rms": features.get('A_rms', 0),
                "phase_b_rms": features.get('B_rms', 0),
                "phase_c_rms": features.get('C_rms', 0),
                "fundamental_frequency": 50.0,  # Hz
                "thd_phase_a": features.get('A_freq_thd', 0),
                "thd_phase_b": features.get('B_freq_thd', 0),
                "thd_phase_c": features.get('C_freq_thd', 0)
            },
            "fault_indicators": {
                "ground_fault_indicator": features.get('ground_fault_indicator', 0),
                "phase_imbalance": {
                    "ab_fault_indicator": features.get('ab_fault_indicator', 0),
                    "ac_fault_indicator": features.get('ac_fault_indicator', 0),
                    "bc_fault_indicator": features.get('bc_fault_indicator', 0)
                },
                "transient_indicators": {
                    "max_rate_change_a": features.get('A_max_rate_change', 0),
                    "max_rate_change_b": features.get('B_max_rate_change', 0),
                    "max_rate_change_c": features.get('C_max_rate_change', 0)
                }
            },
            "symmetrical_components": {
                "positive_sequence_rms": features.get('positive_seq_rms', 0),
                "negative_sequence_rms": features.get('negative_seq_rms', 0),
                "zero_sequence_rms": features.get('zero_seq_rms', 0),
                "negative_sequence_ratio": features.get('negative_seq_ratio', 0),
                "zero_sequence_ratio": features.get('zero_seq_ratio', 0)
            }
        }
        
        if request.return_features:
            analysis_result["extracted_features"] = features
            
        return JSONResponse(content=analysis_result)
        
    except Exception as e:
        logger.error(f"Diagnostic analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/upload")
async def upload_transformer_data(file: UploadFile = File(...)):
    """Upload and analyze transformer data file"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Parse CSV data
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None)
            
            # Assume 4-column format: time, phase_a, phase_b, phase_c
            if df.shape[1] != 4:
                raise HTTPException(status_code=400, detail=f"Expected 4 columns, got {df.shape[1]}")
                
            df.columns = ['time', 'phase_a', 'phase_b', 'phase_c']
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")
        
        # Validate data
        if len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data points (minimum 100 required)")
            
        # Extract features and analyze
        pipeline_instance = get_pipeline()
        features = pipeline_instance.feature_extractor.extract_comprehensive_features(df)
        
        result = {
            "filename": file.filename,
            "upload_timestamp": datetime.now().isoformat(),
            "data_summary": {
                "samples": len(df),
                "duration_seconds": df['time'].max() - df['time'].min(),
                "time_range": [float(df['time'].min()), float(df['time'].max())],
                "phase_ranges": {
                    "phase_a": [float(df['phase_a'].min()), float(df['phase_a'].max())],
                    "phase_b": [float(df['phase_b'].min()), float(df['phase_b'].max())],
                    "phase_c": [float(df['phase_c'].min()), float(df['phase_c'].max())]
                }
            },
            "extracted_features": {k: float(v) for k, v in features.items()},
            "analysis_status": "completed"
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features/importance")
async def get_feature_importance():
    """Get feature importance ranking"""
    try:
        pipeline_instance = get_pipeline()
        
        if not pipeline_instance.pipeline_state['features_extracted']:
            raise HTTPException(status_code=400, detail="Features not extracted yet")
            
        # Get feature importance from selector
        importance_df = pipeline_instance.feature_selector.get_feature_importance_ranking()
        
        if importance_df.empty:
            return JSONResponse(content={"message": "No feature importance data available"})
            
        # Convert to dictionary for JSON response
        importance_data = importance_df.to_dict('records')
        
        result = {
            "feature_importance": importance_data,
            "total_features": len(importance_data),
            "top_10_features": importance_data[:10] if len(importance_data) >= 10 else importance_data
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/classes/mapping")
async def get_class_mapping():
    """Get fault class mapping"""
    try:
        pipeline_instance = get_pipeline()
        
        if not pipeline_instance.class_mapping:
            raise HTTPException(status_code=400, detail="Class mapping not available. Please run pipeline first.")
            
        return JSONResponse(content={
            "class_mapping": pipeline_instance.class_mapping,
            "total_classes": len(pipeline_instance.class_mapping),
            "classes": list(pipeline_instance.class_mapping.keys())
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get class mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}