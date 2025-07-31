"""Performance monitoring and client protection for ScreenMonitorMCP v2."""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass, field

try:
    from ..server.config import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from server.config import config

from .connection import connection_manager
from .streaming import stream_manager

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring system health."""
    
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    active_streams: int = 0
    avg_response_time: float = 0.0
    failed_connections: int = 0
    data_throughput: float = 0.0  # MB/s
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Monitor system performance and protect clients from overload."""
    
    def __init__(self):
        self._metrics = PerformanceMetrics()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._response_times: list = []
        self._data_sent: float = 0.0
        self._last_data_reset = time.time()
        
    async def start_monitoring(self):
        """Start performance monitoring tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info("Performance monitoring started")
            
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cleanup monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring tasks."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            while True:
                await self._update_metrics()
                await self._check_system_health()
                await asyncio.sleep(10)  # Update every 10 seconds
        except asyncio.CancelledError:
            logger.info("Performance monitoring loop cancelled")
        except Exception as e:
            logger.error("Performance monitoring error", error=str(e), exc_info=True)
    
    async def _cleanup_loop(self):
        """Cleanup loop for idle connections and resources."""
        try:
            while True:
                await self._cleanup_idle_connections()
                await self._cleanup_failed_streams()
                await asyncio.sleep(30)  # Cleanup every 30 seconds
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
        except Exception as e:
            logger.error("Cleanup loop error", error=str(e), exc_info=True)
    
    async def _update_metrics(self):
        """Update performance metrics."""
        try:
            # Update connection and stream counts
            connections = await connection_manager.get_active_connections()
            streams = await stream_manager.get_active_streams()
            
            self._metrics.active_connections = len(connections)
            self._metrics.active_streams = len(streams)
            
            # Calculate average response time
            if self._response_times:
                self._metrics.avg_response_time = sum(self._response_times) / len(self._response_times)
                # Keep only recent response times (last 100)
                self._response_times = self._response_times[-100:]
            
            # Calculate data throughput
            current_time = time.time()
            time_diff = current_time - self._last_data_reset
            if time_diff >= 1.0:  # Calculate per second
                self._metrics.data_throughput = self._data_sent / time_diff
                self._data_sent = 0.0
                self._last_data_reset = current_time
            
            self._metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error("Failed to update metrics", error=str(e))
    
    async def _check_system_health(self):
        """Check system health and take protective actions."""
        try:
            # Check if too many connections
            if self._metrics.active_connections > config.max_connections * 0.9:
                logger.warning(
                    "High connection count detected",
                    current=self._metrics.active_connections,
                    limit=config.max_connections
                )
                await self._reduce_system_load()
            
            # Check if too many streams
            if self._metrics.active_streams > config.max_concurrent_streams * 0.8:
                logger.warning(
                    "High stream count detected",
                    current=self._metrics.active_streams,
                    limit=config.max_concurrent_streams
                )
                await self._reduce_stream_load()
            
            # Check response time
            if self._metrics.avg_response_time > 2.0:  # 2 seconds threshold
                logger.warning(
                    "High response time detected",
                    avg_response_time=self._metrics.avg_response_time
                )
                await self._optimize_performance()
                
        except Exception as e:
            logger.error("Health check failed", error=str(e))
    
    async def _reduce_system_load(self):
        """Reduce system load by cleaning up resources."""
        try:
            # Clean up idle connections more aggressively
            idle_threshold = timedelta(seconds=config.connection_timeout // 2)
            cleaned = await connection_manager.cleanup_idle_connections(idle_threshold)
            
            if cleaned > 0:
                logger.info("Cleaned up connections to reduce load", count=cleaned)
                
        except Exception as e:
            logger.error("Failed to reduce system load", error=str(e))
    
    async def _reduce_stream_load(self):
        """Reduce stream load by stopping low-priority streams."""
        try:
            streams = await stream_manager.get_active_streams()
            
            # Stop streams with no active connections
            for stream_id, stream_info in streams.items():
                connections = await connection_manager.get_stream_connections(stream_id)
                if len(connections) == 0:
                    await stream_manager.stop_stream(stream_id)
                    logger.info("Stopped stream with no connections", stream_id=stream_id)
                    
        except Exception as e:
            logger.error("Failed to reduce stream load", error=str(e))
    
    async def _optimize_performance(self):
        """Optimize performance by adjusting stream settings."""
        try:
            streams = await stream_manager.get_active_streams()
            
            for stream_id, stream_info in streams.items():
                # Reduce FPS for high-load streams
                current_fps = stream_info.get('fps', 1)
                if current_fps > 1:
                    # Reduce FPS by half, minimum 1
                    new_fps = max(1, current_fps // 2)
                    stream_info['fps'] = new_fps
                    logger.info(
                        "Reduced stream FPS for performance",
                        stream_id=stream_id,
                        old_fps=current_fps,
                        new_fps=new_fps
                    )
                    
        except Exception as e:
            logger.error("Failed to optimize performance", error=str(e))
    
    async def _cleanup_idle_connections(self):
        """Clean up idle connections."""
        try:
            idle_threshold = timedelta(seconds=config.connection_timeout)
            cleaned = await connection_manager.cleanup_idle_connections(idle_threshold)
            
            if cleaned > 0:
                logger.debug("Cleaned up idle connections", count=cleaned)
                
        except Exception as e:
            logger.error("Failed to cleanup idle connections", error=str(e))
    
    async def _cleanup_failed_streams(self):
        """Clean up failed or orphaned streams."""
        try:
            streams = await stream_manager.get_active_streams()
            
            for stream_id, stream_info in streams.items():
                # Check if stream has been running too long without activity
                created_at = stream_info.get('created_at')
                if created_at and datetime.now() - created_at > timedelta(hours=1):
                    connections = await connection_manager.get_stream_connections(stream_id)
                    if len(connections) == 0:
                        await stream_manager.stop_stream(stream_id)
                        logger.info("Cleaned up orphaned stream", stream_id=stream_id)
                        
        except Exception as e:
            logger.error("Failed to cleanup failed streams", error=str(e))
    
    def record_response_time(self, response_time: float):
        """Record a response time for metrics."""
        self._response_times.append(response_time)
    
    def record_data_sent(self, bytes_sent: int):
        """Record data sent for throughput calculation."""
        self._data_sent += bytes_sent / (1024 * 1024)  # Convert to MB
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self._metrics
    
    def is_running(self) -> bool:
        """Check if performance monitoring is running."""
        return self._monitoring_task is not None and not self._monitoring_task.done()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        metrics = self.get_metrics()
        
        # Determine health status
        health_score = 100
        issues = []
        
        if metrics.active_connections > config.max_connections * 0.8:
            health_score -= 20
            issues.append("High connection count")
        
        if metrics.active_streams > config.max_concurrent_streams * 0.7:
            health_score -= 15
            issues.append("High stream count")
        
        if metrics.avg_response_time > 1.0:
            health_score -= 25
            issues.append("Slow response times")
        
        if metrics.data_throughput > 50:  # 50 MB/s threshold
            health_score -= 10
            issues.append("High data throughput")
        
        status = "healthy"
        if health_score < 50:
            status = "critical"
        elif health_score < 70:
            status = "warning"
        elif health_score < 90:
            status = "degraded"
        
        return {
            "status": status,
            "health_score": health_score,
            "issues": issues,
            "metrics": {
                "active_connections": metrics.active_connections,
                "active_streams": metrics.active_streams,
                "avg_response_time": metrics.avg_response_time,
                "data_throughput_mb_s": metrics.data_throughput,
                "last_updated": metrics.last_updated.isoformat()
            },
            "limits": {
                "max_connections": config.max_connections,
                "max_streams": config.max_concurrent_streams,
                "connection_timeout": config.connection_timeout
            }
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()