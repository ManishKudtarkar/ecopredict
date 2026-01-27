"""
Health check utilities for production monitoring
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    memory_percent: float
    cpu_percent: float
    db_connected: bool
    error: Optional[str] = None


class ServiceHealth(BaseModel):
    api: str
    dashboard: str
    database: str
    cache: str
    timestamp: datetime


# Track startup time
_startup_time: Optional[float] = None


def set_startup_time():
    """Set the application startup time"""
    global _startup_time
    _startup_time = datetime.now().timestamp()


def get_uptime() -> float:
    """Get uptime in seconds"""
    if _startup_time is None:
        return 0
    return datetime.now().timestamp() - _startup_time


async def check_database_connection() -> bool:
    """Check if database is accessible"""
    try:
        # This would import and use your database connection
        # For now, returning True as placeholder
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def check_cache_connection() -> bool:
    """Check if cache/Redis is accessible"""
    try:
        # This would check Redis or cache connection
        return True
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return False


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint
    
    Returns:
        HealthStatus: Current health status of the application
    """
    try:
        # Get system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Check database
        db_connected = await check_database_connection()
        
        # Build response
        return HealthStatus(
            status="healthy" if db_connected else "degraded",
            timestamp=datetime.now(),
            version="1.0.0",  # Should come from your version file
            uptime_seconds=get_uptime(),
            memory_percent=memory_info.percent,
            cpu_percent=cpu_percent,
            db_connected=db_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/health/detailed", response_model=ServiceHealth)
async def detailed_health_check():
    """
    Detailed health check for all services
    
    Returns:
        ServiceHealth: Health status of all services
    """
    try:
        db_status = "healthy" if await check_database_connection() else "unhealthy"
        cache_status = "healthy" if await check_cache_connection() else "unhealthy"
        
        return ServiceHealth(
            api="healthy",
            dashboard="healthy",
            database=db_status,
            cache=cache_status,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe
    Returns 200 if service is ready to accept traffic
    """
    try:
        db_connected = await check_database_connection()
        if not db_connected:
            raise HTTPException(status_code=503, detail="Database not ready")
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/alive")
async def liveness_check():
    """
    Kubernetes liveness probe
    Returns 200 if service is alive
    """
    return {"status": "alive"}


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get application metrics
    
    Returns:
        Dict containing current metrics
    """
    try:
        process = psutil.Process()
        memory_info = psutil.virtual_memory()
        
        return {
            "uptime_seconds": get_uptime(),
            "memory": {
                "total_mb": memory_info.total / (1024 * 1024),
                "used_mb": memory_info.used / (1024 * 1024),
                "percent": memory_info.percent
            },
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count()
            },
            "process": {
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads": process.num_threads()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")
