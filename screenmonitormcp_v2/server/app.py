"""FastAPI application for ScreenMonitorMCP v2."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ..core.connection import connection_manager
from ..core.streaming import stream_manager
from ..core.performance_monitor import performance_monitor
from ..models.responses import HealthCheckResponse
from ..server.config import config
from .routes import api_router, ws_router

logger = structlog.get_logger()

# Create app factory for testing
def create_app():
    """Create FastAPI application instance."""
    return app

# Global variable to track application start time
app_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global app_start_time
    app_start_time = datetime.now()

    logger.info(
        "Starting ScreenMonitorMCP v2",
        host=config.host,
        port=config.port,
        version="2.0.0",
    )

    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_idle_connections())
    
    # Start performance monitoring
    await performance_monitor.start_monitoring()

    yield

    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Stop performance monitoring
    await performance_monitor.stop_monitoring()

    await stream_manager.cleanup()
    await connection_manager.cleanup()

    logger.info("ScreenMonitorMCP v2 stopped")


async def cleanup_idle_connections():
    """Background task to cleanup idle connections."""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            await connection_manager.cleanup_idle_connections(
                max_idle_time=timedelta(minutes=5)
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error in cleanup task", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="ScreenMonitorMCP v2",
    description="Streamable HTTP/SSE MCP Server for screen monitoring and AI vision",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v2")
app.include_router(ws_router, prefix="/ws")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ScreenMonitorMCP v2",
        "version": "2.0.0",
        "description": "Streamable HTTP/SSE MCP Server",
        "endpoints": {
            "api": "/api/v2",
            "websocket": "/ws",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint with performance monitoring."""
    try:
        # Get connection stats
        connection_stats = await connection_manager.get_stats()

        # Get stream stats
        active_streams = await stream_manager.get_active_streams()
        
        # Get performance health status
        health_status = await performance_monitor.get_health_status()

        if app_start_time is None:
            uptime = 0
        else:
            uptime = int((datetime.now() - app_start_time).total_seconds())

        return HealthCheckResponse(
            success=True,
            message=f"Service is {health_status['status']}",
            version="2.0.0",
            uptime=uptime,
            status=health_status['status'],
            system_info={"python_version": "3.9+", "platform": "cross-platform"},
            dependencies={
                "mss": True,
                "pillow": True,
                "fastapi": True,
                "uvicorn": True,
            },
            performance={
                "active_connections": connection_stats["total_connections"],
                "active_streams": len(active_streams),
                "health_score": health_status['health_score'],
                "avg_response_time": health_status['metrics']['avg_response_time'],
                "data_throughput_mb_s": health_status['metrics']['data_throughput_mb_s'],
                "issues_count": float(len(health_status['issues']))
            },
            active_streams=len(active_streams),
            active_connections=connection_stats["total_connections"],
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "screenmonitormcp_v2.server.app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level=config.log_level.lower(),
    )
