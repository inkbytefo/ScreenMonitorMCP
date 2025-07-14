"""
Smart Error Recovery System for ScreenMonitorMCP
Provides graceful degradation, automatic recovery, and comprehensive error handling
"""

import time
import threading
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1      # Minor issues, continue operation
    MEDIUM = 2   # Moderate issues, degraded operation
    HIGH = 3     # Serious issues, limited operation
    CRITICAL = 4 # Critical issues, emergency fallback

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = ""
    function_name: str = ""
    traceback_info: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None

@dataclass
class RecoveryConfig:
    """Configuration for error recovery"""
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    fallback_function: Optional[Callable] = None
    enable_graceful_degradation: bool = True
    critical_error_threshold: int = 5  # Number of critical errors before emergency stop

class ErrorRecoveryManager:
    """Smart error recovery and management system"""
    
    def __init__(self):
        self.error_history: List[ErrorRecord] = []
        self.component_configs: Dict[str, RecoveryConfig] = {}
        self.fallback_functions: Dict[str, Callable] = {}
        self.health_checks: Dict[str, Callable] = {}
        
        # Statistics
        self.total_errors = 0
        self.recovered_errors = 0
        self.critical_errors = 0
        self.emergency_stops = 0
        
        # Threading
        self._lock = threading.RLock()
        self._health_check_thread = None
        self._stop_health_checks = threading.Event()
        
        # Default configurations
        self._setup_default_configs()
        
        # Start health monitoring
        self._start_health_monitoring()
        
        logger.info("Error recovery manager initialized")
    
    def _setup_default_configs(self):
        """Setup default recovery configurations for common components"""
        self.component_configs.update({
            "screenshot": RecoveryConfig(
                max_retries=3,
                retry_delay=0.5,
                enable_graceful_degradation=True
            ),
            "ai_analysis": RecoveryConfig(
                max_retries=2,
                retry_delay=2.0,
                exponential_backoff=True,
                enable_graceful_degradation=True
            ),
            "ocr": RecoveryConfig(
                max_retries=2,
                retry_delay=1.0,
                enable_graceful_degradation=True
            ),
            "smart_click": RecoveryConfig(
                max_retries=3,
                retry_delay=0.5,
                enable_graceful_degradation=True
            ),
            "cache": RecoveryConfig(
                max_retries=1,
                retry_delay=0.1,
                enable_graceful_degradation=True
            ),
            "monitoring": RecoveryConfig(
                max_retries=5,
                retry_delay=1.0,
                exponential_backoff=True
            )
        })
    
    def _start_health_monitoring(self):
        """Start health check monitoring"""
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
    
    def _health_check_loop(self):
        """Health check monitoring loop"""
        while not self._stop_health_checks.wait(30.0):  # Check every 30 seconds
            try:
                self._perform_health_checks()
                self._cleanup_old_errors()
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
    
    def _perform_health_checks(self):
        """Perform registered health checks"""
        with self._lock:
            for component, health_check in self.health_checks.items():
                try:
                    if not health_check():
                        logger.warning(f"Health check failed for component: {component}")
                        self._record_error(
                            error_type="HealthCheckFailure",
                            error_message=f"Health check failed for {component}",
                            severity=ErrorSeverity.MEDIUM,
                            component=component
                        )
                except Exception as e:
                    logger.error(f"Health check exception for {component}: {str(e)}")
    
    def _cleanup_old_errors(self):
        """Clean up old error records (keep last 24 hours)"""
        cutoff_time = time.time() - (24 * 3600)  # 24 hours ago
        
        with self._lock:
            original_count = len(self.error_history)
            self.error_history = [
                error for error in self.error_history 
                if error.timestamp > cutoff_time
            ]
            
            cleaned_count = original_count - len(self.error_history)
            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} old error records")
    
    def execute_with_recovery(self, 
                            function: Callable,
                            component: str,
                            *args,
                            **kwargs) -> Any:
        """Execute a function with automatic error recovery"""
        config = self.component_configs.get(component, RecoveryConfig())
        
        for attempt in range(config.max_retries + 1):
            try:
                result = function(*args, **kwargs)
                
                # Success - reset any previous error state for this component
                self._reset_component_errors(component)
                return result
                
            except Exception as e:
                error_record = self._record_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=self._determine_severity(e, component),
                    component=component,
                    function_name=function.__name__,
                    traceback_info=traceback.format_exc()
                )
                
                # Determine recovery strategy
                strategy = self._determine_recovery_strategy(error_record, attempt, config)
                error_record.recovery_strategy = strategy
                
                if strategy == RecoveryStrategy.EMERGENCY_STOP:
                    self.emergency_stops += 1
                    raise Exception(f"Emergency stop triggered for component {component}")
                
                elif strategy == RecoveryStrategy.RETRY and attempt < config.max_retries:
                    delay = self._calculate_retry_delay(config, attempt)
                    logger.warning(f"Retrying {component}.{function.__name__} in {delay}s (attempt {attempt + 1}/{config.max_retries})")
                    time.sleep(delay)
                    continue
                
                elif strategy == RecoveryStrategy.FALLBACK:
                    fallback_result = self._try_fallback(component, function, args, kwargs)
                    if fallback_result is not None:
                        error_record.recovery_attempted = True
                        error_record.recovery_successful = True
                        self.recovered_errors += 1
                        return fallback_result
                
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    degraded_result = self._graceful_degradation(component, function, e)
                    if degraded_result is not None:
                        error_record.recovery_attempted = True
                        error_record.recovery_successful = True
                        self.recovered_errors += 1
                        return degraded_result
                
                # If we get here, all recovery attempts failed
                logger.error(f"All recovery attempts failed for {component}.{function.__name__}")
                raise e
    
    def _record_error(self, 
                     error_type: str,
                     error_message: str,
                     severity: ErrorSeverity,
                     component: str,
                     function_name: str = "",
                     traceback_info: str = "") -> ErrorRecord:
        """Record an error occurrence"""
        
        error_record = ErrorRecord(
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            component=component,
            function_name=function_name,
            traceback_info=traceback_info
        )
        
        with self._lock:
            self.error_history.append(error_record)
            self.total_errors += 1
            
            if severity == ErrorSeverity.CRITICAL:
                self.critical_errors += 1
        
        logger.error(f"Error recorded: {component}.{function_name} - {error_type}: {error_message}")
        return error_record
    
    def _determine_severity(self, exception: Exception, component: str) -> ErrorSeverity:
        """Determine error severity based on exception type and component"""
        
        # Critical errors
        if isinstance(exception, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        
        # Component-specific severity
        if component in ["cache", "monitoring"]:
            return ErrorSeverity.LOW
        elif component in ["screenshot", "ocr"]:
            return ErrorSeverity.MEDIUM
        elif component in ["ai_analysis", "smart_click"]:
            return ErrorSeverity.HIGH
        
        return ErrorSeverity.MEDIUM
    
    def _determine_recovery_strategy(self, 
                                   error_record: ErrorRecord,
                                   attempt: int,
                                   config: RecoveryConfig) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy"""
        
        # Check for emergency stop conditions
        recent_critical_errors = self._count_recent_errors(
            component=error_record.component,
            severity=ErrorSeverity.CRITICAL,
            time_window=300  # 5 minutes
        )
        
        if recent_critical_errors >= config.critical_error_threshold:
            return RecoveryStrategy.EMERGENCY_STOP
        
        # Retry strategy
        if attempt < config.max_retries and error_record.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            return RecoveryStrategy.RETRY
        
        # Fallback strategy
        if error_record.component in self.fallback_functions:
            return RecoveryStrategy.FALLBACK
        
        # Graceful degradation
        if config.enable_graceful_degradation:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        return RecoveryStrategy.RETRY
    
    def _calculate_retry_delay(self, config: RecoveryConfig, attempt: int) -> float:
        """Calculate retry delay with optional exponential backoff"""
        if config.exponential_backoff:
            return config.retry_delay * (2 ** attempt)
        return config.retry_delay
    
    def _try_fallback(self, 
                     component: str,
                     original_function: Callable,
                     args: tuple,
                     kwargs: dict) -> Any:
        """Try fallback function if available"""
        if component not in self.fallback_functions:
            return None
        
        try:
            fallback_func = self.fallback_functions[component]
            logger.info(f"Attempting fallback for {component}")
            return fallback_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback failed for {component}: {str(e)}")
            return None
    
    def _graceful_degradation(self, 
                            component: str,
                            function: Callable,
                            original_error: Exception) -> Any:
        """Provide graceful degradation response"""
        
        degradation_responses = {
            "screenshot": "Screenshot capture failed - using cached or placeholder image",
            "ai_analysis": "AI analysis unavailable - providing basic analysis",
            "ocr": "OCR failed - text extraction unavailable",
            "smart_click": "Smart click failed - manual interaction required",
            "cache": "Cache unavailable - operating without cache",
            "monitoring": "Monitoring temporarily disabled"
        }
        
        if component in degradation_responses:
            logger.warning(f"Graceful degradation for {component}: {degradation_responses[component]}")
            return degradation_responses[component]
        
        return f"Service temporarily unavailable: {component}"
    
    def _count_recent_errors(self, 
                           component: str,
                           severity: ErrorSeverity,
                           time_window: int) -> int:
        """Count recent errors for a component within time window"""
        cutoff_time = time.time() - time_window
        
        with self._lock:
            return sum(
                1 for error in self.error_history
                if (error.component == component and 
                    error.severity == severity and 
                    error.timestamp > cutoff_time)
            )
    
    def _reset_component_errors(self, component: str):
        """Reset error state for a component after successful operation"""
        # This could be used to reset retry counters or other state
        pass
    
    def register_fallback(self, component: str, fallback_function: Callable):
        """Register a fallback function for a component"""
        self.fallback_functions[component] = fallback_function
        logger.info(f"Fallback function registered for component: {component}")
    
    def register_health_check(self, component: str, health_check_function: Callable):
        """Register a health check function for a component"""
        self.health_checks[component] = health_check_function
        logger.info(f"Health check registered for component: {component}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if error.timestamp > time.time() - 3600  # Last hour
            ]
            
            component_stats = {}
            for error in self.error_history:
                if error.component not in component_stats:
                    component_stats[error.component] = {
                        "total_errors": 0,
                        "recovered_errors": 0,
                        "critical_errors": 0
                    }
                
                component_stats[error.component]["total_errors"] += 1
                if error.recovery_successful:
                    component_stats[error.component]["recovered_errors"] += 1
                if error.severity == ErrorSeverity.CRITICAL:
                    component_stats[error.component]["critical_errors"] += 1
            
            return {
                "total_errors": self.total_errors,
                "recovered_errors": self.recovered_errors,
                "critical_errors": self.critical_errors,
                "emergency_stops": self.emergency_stops,
                "recovery_rate": (self.recovered_errors / self.total_errors * 100) if self.total_errors > 0 else 0,
                "recent_errors_count": len(recent_errors),
                "component_statistics": component_stats,
                "registered_fallbacks": list(self.fallback_functions.keys()),
                "registered_health_checks": list(self.health_checks.keys())
            }
    
    def shutdown(self):
        """Shutdown error recovery manager"""
        self._stop_health_checks.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        logger.info("Error recovery manager shutdown complete")

# Global error recovery manager instance
_error_recovery_manager: Optional[ErrorRecoveryManager] = None

def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance"""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager

def with_recovery(component: str):
    """Decorator for automatic error recovery"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            recovery_manager = get_error_recovery_manager()
            return recovery_manager.execute_with_recovery(func, component, *args, **kwargs)
        return wrapper
    return decorator
