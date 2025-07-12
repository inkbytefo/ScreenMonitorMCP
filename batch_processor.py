"""
Request Batching System for ScreenMonitorMCP
Provides efficient processing of multiple requests with configurable batch sizes
"""

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class BatchPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BatchRequest:
    """Individual request in a batch"""
    id: str
    function: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"req_{int(time.time() * 1000000)}"

@dataclass
class BatchResult:
    """Result of a batch request"""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    completed_at: float = field(default_factory=time.time)

class BatchProcessor:
    """Efficient batch processing system with priority support"""
    
    def __init__(self, 
                 max_batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 max_concurrent_batches: int = 3,
                 enable_priority_queue: bool = True):
        
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_priority_queue = enable_priority_queue
        
        # Request queues
        self.pending_requests: List[BatchRequest] = []
        self.processing_batches: Dict[str, List[BatchRequest]] = {}
        self.completed_results: Dict[str, BatchResult] = {}
        
        # Threading
        self._queue_lock = threading.RLock()
        self._processor_thread = None
        self._stop_processing = threading.Event()
        
        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_processing_time = 0.0
        self.average_batch_size = 0.0
        
        # Start processor
        self._start_processor()
        
        logger.info(f"Batch processor initialized - max_batch_size: {max_batch_size}, "
                   f"max_wait_time: {max_wait_time}s, max_concurrent: {max_concurrent_batches}")
    
    def _start_processor(self):
        """Start the batch processor thread"""
        self._processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self._processor_thread.start()
    
    def _processor_loop(self):
        """Main processor loop"""
        while not self._stop_processing.wait(0.1):  # Check every 100ms
            try:
                self._process_pending_requests()
            except Exception as e:
                logger.error(f"Batch processor error: {str(e)}")
    
    def _process_pending_requests(self):
        """Process pending requests into batches"""
        with self._queue_lock:
            if not self.pending_requests:
                return
            
            # Check if we can create a new batch
            if len(self.processing_batches) >= self.max_concurrent_batches:
                return
            
            # Sort by priority if enabled
            if self.enable_priority_queue:
                self.pending_requests.sort(key=lambda r: (r.priority.value, r.created_at), reverse=True)
            
            # Create batch
            batch_size = min(self.max_batch_size, len(self.pending_requests))
            oldest_request_time = self.pending_requests[0].created_at if self.pending_requests else time.time()
            
            # Check if we should process now (batch full or wait time exceeded)
            should_process = (
                batch_size >= self.max_batch_size or
                (time.time() - oldest_request_time) >= self.max_wait_time
            )
            
            if should_process and batch_size > 0:
                batch_requests = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
                
                batch_id = f"batch_{int(time.time() * 1000000)}"
                self.processing_batches[batch_id] = batch_requests
                
                # Process batch asynchronously
                threading.Thread(
                    target=self._execute_batch,
                    args=(batch_id, batch_requests),
                    daemon=True
                ).start()
    
    def _execute_batch(self, batch_id: str, requests: List[BatchRequest]):
        """Execute a batch of requests"""
        start_time = time.time()
        
        try:
            logger.debug(f"Processing batch {batch_id} with {len(requests)} requests")
            
            # Execute requests
            for request in requests:
                result = self._execute_single_request(request)
                
                # Store result
                with self._queue_lock:
                    self.completed_results[request.id] = result
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error for request {request.id}: {str(e)}")
            
            # Update statistics
            processing_time = time.time() - start_time
            with self._queue_lock:
                self.total_batches += 1
                self.total_processing_time += processing_time
                self.average_batch_size = (
                    (self.average_batch_size * (self.total_batches - 1) + len(requests)) / 
                    self.total_batches
                )
            
            logger.debug(f"Batch {batch_id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Batch {batch_id} execution failed: {str(e)}")
            
            # Mark all requests as failed
            for request in requests:
                with self._queue_lock:
                    self.completed_results[request.id] = BatchResult(
                        request_id=request.id,
                        success=False,
                        error=str(e)
                    )
        
        finally:
            # Remove from processing
            with self._queue_lock:
                if batch_id in self.processing_batches:
                    del self.processing_batches[batch_id]
    
    def _execute_single_request(self, request: BatchRequest) -> BatchResult:
        """Execute a single request"""
        start_time = time.time()
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(request.function):
                # Handle async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        request.function(*request.args, **request.kwargs)
                    )
                finally:
                    loop.close()
            else:
                # Handle sync functions
                result = request.function(*request.args, **request.kwargs)
            
            execution_time = time.time() - start_time
            
            return BatchResult(
                request_id=request.id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return BatchResult(
                request_id=request.id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def add_request(self, 
                   function: Callable,
                   args: Tuple = (),
                   kwargs: Dict[str, Any] = None,
                   priority: BatchPriority = BatchPriority.NORMAL,
                   callback: Optional[Callable] = None,
                   request_id: Optional[str] = None) -> str:
        """Add a request to the batch queue"""
        
        if kwargs is None:
            kwargs = {}
        
        request = BatchRequest(
            id=request_id or f"req_{int(time.time() * 1000000)}",
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            callback=callback
        )
        
        with self._queue_lock:
            self.pending_requests.append(request)
            self.total_requests += 1
        
        logger.debug(f"Request {request.id} added to batch queue")
        return request.id
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[BatchResult]:
        """Get result for a specific request"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._queue_lock:
                if request_id in self.completed_results:
                    return self.completed_results.pop(request_id)
            
            time.sleep(0.1)  # Wait 100ms before checking again
        
        return None  # Timeout
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processor statistics"""
        with self._queue_lock:
            return {
                "total_requests": self.total_requests,
                "total_batches": self.total_batches,
                "pending_requests": len(self.pending_requests),
                "processing_batches": len(self.processing_batches),
                "completed_results": len(self.completed_results),
                "average_batch_size": round(self.average_batch_size, 2),
                "total_processing_time": round(self.total_processing_time, 2),
                "average_processing_time": round(
                    self.total_processing_time / self.total_batches if self.total_batches > 0 else 0, 2
                ),
                "configuration": {
                    "max_batch_size": self.max_batch_size,
                    "max_wait_time": self.max_wait_time,
                    "max_concurrent_batches": self.max_concurrent_batches,
                    "priority_queue_enabled": self.enable_priority_queue
                }
            }
    
    def shutdown(self):
        """Shutdown the batch processor"""
        self._stop_processing.set()
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        logger.info("Batch processor shutdown complete")

# Global batch processor instance
_batch_processor: Optional[BatchProcessor] = None

def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor

def shutdown_batch_processor():
    """Shutdown global batch processor"""
    global _batch_processor
    if _batch_processor:
        _batch_processor.shutdown()
        _batch_processor = None
