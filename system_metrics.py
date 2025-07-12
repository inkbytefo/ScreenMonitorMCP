"""
System Metrics Manager for ScreenMonitorMCP
Provides real-time system health monitoring and performance tracking
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from cache_manager import get_cache_manager

logger = logging.getLogger(__name__)

@dataclass
class SystemSnapshot:
    """System performance snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv
        }

@dataclass
class ProviderStatus:
    """AI Provider status information"""
    name: str
    status: str  # 'active', 'inactive', 'error'
    last_request_time: Optional[float] = None
    total_requests: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status,
            'last_request_time': self.last_request_time,
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'average_response_time': self.average_response_time,
            'last_error': self.last_error
        }

class SystemMetricsManager:
    """Manages system metrics collection and monitoring"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.cache_manager = get_cache_manager()
        
        # Metrics storage
        self.system_snapshots: List[SystemSnapshot] = []
        self.provider_statuses: Dict[str, ProviderStatus] = {}
        self.max_snapshots = 1000  # Keep last 1000 snapshots
        
        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._metrics_lock = threading.RLock()
        
        # Performance counters
        self.start_time = time.time()
        self.total_screenshots = 0
        self.total_analyses = 0
        self.total_ui_detections = 0
        self.total_smart_clicks = 0
        
        logger.info(f"System metrics manager initialized - collection interval: {collection_interval}s")
    
    def start_monitoring(self):
        """Start system metrics monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return {"status": "already_running"}
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("System metrics monitoring started")
        return {"status": "started", "collection_interval": self.collection_interval}
    
    def stop_monitoring(self):
        """Stop system metrics monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("System metrics monitoring stopped")
        return {"status": "stopped"}
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring.wait(self.collection_interval):
            try:
                self._collect_system_snapshot()
            except Exception as e:
                logger.error(f"System metrics collection error: {str(e)}")
    
    def _collect_system_snapshot(self):
        """Collect current system performance snapshot"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage (root drive)
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            snapshot = SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv
            )
            
            with self._metrics_lock:
                self.system_snapshots.append(snapshot)
                
                # Keep only recent snapshots
                if len(self.system_snapshots) > self.max_snapshots:
                    self.system_snapshots = self.system_snapshots[-self.max_snapshots:]
            
            # Cache the snapshot
            self.cache_manager.set("metrics", f"snapshot_{int(time.time())}", snapshot.to_dict(), ttl=3600)
            
        except Exception as e:
            logger.error(f"Failed to collect system snapshot: {str(e)}")
    def update_provider_status(self, provider_name: str, status: str, 
                             response_time: Optional[float] = None,
                             error: Optional[str] = None):
        """Update AI provider status"""
        with self._metrics_lock:
            if provider_name not in self.provider_statuses:
                self.provider_statuses[provider_name] = ProviderStatus(
                    name=provider_name,
                    status=status
                )
            
            provider = self.provider_statuses[provider_name]
            provider.status = status
            provider.last_request_time = time.time()
            provider.total_requests += 1
            
            if response_time:
                # Update average response time
                if provider.average_response_time == 0:
                    provider.average_response_time = response_time
                else:
                    provider.average_response_time = (
                        provider.average_response_time * 0.8 + response_time * 0.2
                    )
            
            if error:
                provider.error_count += 1
                provider.last_error = error
                provider.status = "error"
    
    def increment_counter(self, counter_name: str):
        """Increment performance counter"""
        with self._metrics_lock:
            if counter_name == "screenshots":
                self.total_screenshots += 1
            elif counter_name == "analyses":
                self.total_analyses += 1
            elif counter_name == "ui_detections":
                self.total_ui_detections += 1
            elif counter_name == "smart_clicks":
                self.total_smart_clicks += 1
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        with self._metrics_lock:
            if not self.system_snapshots:
                return {"status": "no_data", "message": "No system data collected yet"}

            latest = self.system_snapshots[-1]

            # Determine health status
            health_status = "healthy"
            warnings = []

            if latest.cpu_percent > 80:
                health_status = "warning"
                warnings.append(f"High CPU usage: {latest.cpu_percent:.1f}%")

            if latest.memory_percent > 85:
                health_status = "warning"
                warnings.append(f"High memory usage: {latest.memory_percent:.1f}%")

            if latest.disk_usage_percent > 90:
                health_status = "critical"
                warnings.append(f"High disk usage: {latest.disk_usage_percent:.1f}%")

            # Ensure start_time is available
            uptime_seconds = time.time() - getattr(self, 'start_time', time.time())

            return {
                "status": health_status,
                "warnings": warnings,
                "current_metrics": latest.to_dict(),
                "uptime_seconds": uptime_seconds,
                "monitoring_active": self._monitoring_thread and self._monitoring_thread.is_alive()
            }
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics"""
        with self._metrics_lock:
            uptime = time.time() - getattr(self, 'start_time', time.time())
            
            # Calculate rates
            screenshots_per_hour = (self.total_screenshots / uptime * 3600) if uptime > 0 else 0
            analyses_per_hour = (self.total_analyses / uptime * 3600) if uptime > 0 else 0
            
            # Get cache stats
            cache_stats = self.cache_manager.get_stats()
            
            # Provider statuses
            provider_summary = {}
            for name, provider in self.provider_statuses.items():
                provider_summary[name] = {
                    "status": provider.status,
                    "total_requests": provider.total_requests,
                    "error_rate": (provider.error_count / provider.total_requests * 100) if provider.total_requests > 0 else 0,
                    "avg_response_time": provider.average_response_time
                }
            
            return {
                "uptime_hours": round(uptime / 3600, 2),
                "performance_counters": {
                    "total_screenshots": self.total_screenshots,
                    "total_analyses": self.total_analyses,
                    "total_ui_detections": self.total_ui_detections,
                    "total_smart_clicks": self.total_smart_clicks,
                    "screenshots_per_hour": round(screenshots_per_hour, 2),
                    "analyses_per_hour": round(analyses_per_hour, 2)
                },
                "cache_performance": cache_stats,
                "provider_status": provider_summary,
                "system_snapshots_count": len(self.system_snapshots)
            }
    
    def get_historical_data(self, hours: int = 1) -> Dict[str, Any]:
        """Get historical system data"""
        with self._metrics_lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_snapshots = [
                s for s in self.system_snapshots 
                if s.timestamp >= cutoff_time
            ]
            
            if not recent_snapshots:
                return {"error": "No data available for the specified time range"}
            
            # Calculate averages
            avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
            avg_memory = sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots)
            
            # Find peaks
            max_cpu = max(s.cpu_percent for s in recent_snapshots)
            max_memory = max(s.memory_percent for s in recent_snapshots)
            
            return {
                "time_range_hours": hours,
                "data_points": len(recent_snapshots),
                "averages": {
                    "cpu_percent": round(avg_cpu, 2),
                    "memory_percent": round(avg_memory, 2)
                },
                "peaks": {
                    "max_cpu_percent": round(max_cpu, 2),
                    "max_memory_percent": round(max_memory, 2)
                },
                "snapshots": [s.to_dict() for s in recent_snapshots[-20:]]  # Last 20 snapshots
            }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive system metrics report"""
        return {
            "system_health": self.get_system_health(),
            "performance_metrics": self.get_performance_metrics(),
            "historical_data": self.get_historical_data(1),
            "cache_details": self.cache_manager.get_stats(),
            "timestamp": time.time(),
            "report_generated_at": datetime.now().isoformat()
        }

# Global system metrics manager instance
_metrics_manager: Optional[SystemMetricsManager] = None

def get_metrics_manager() -> SystemMetricsManager:
    """Get global system metrics manager instance"""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = SystemMetricsManager()
        _metrics_manager.start_monitoring()  # Auto-start monitoring
    return _metrics_manager