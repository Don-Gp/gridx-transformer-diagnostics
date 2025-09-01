"""
GRIDX - AI-Powered Transformer Fault Diagnostic System
Main FastAPI application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .api.endpoints.data_pipeline import router as pipeline_router
from .api.endpoints.unified_pipeline import router as unified_router
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GRIDX API",
    description="AI-Powered Transformer Fault Diagnostic System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "GRIDX API is running", "version": "1.0.0"}

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "GRIDX API",
        "version": "1.0.0"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
