"""
Advanced Cache Manager for ScreenMonitorMCP
Provides TTL-based caching with metrics and automatic cleanup
"""

import time
import threading
import hashlib
import pickle
import os
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() > (self.created_at + self.ttl)
    
    def access(self) -> Any:
        """Access the cached value and update stats"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0
    disk_usage_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

class CacheManager:
    """Advanced cache manager with TTL support and metrics"""
    
    def __init__(self,
                 max_memory_entries: int = 1000,
                 max_memory_size_mb: int = 100,
                 disk_cache_dir: str = None,
                 cleanup_interval: int = 300):  # 5 minutes
        
        self.max_memory_entries = max_memory_entries
        self.max_memory_size_bytes = max_memory_size_mb * 1024 * 1024

        # Set cache directory to absolute path
        if disk_cache_dir is None:
            # Get current script directory and create cache folder there
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.disk_cache_dir = os.path.join(current_dir, "cache")
        else:
            self.disk_cache_dir = os.path.abspath(disk_cache_dir)

        self.cleanup_interval = cleanup_interval
        
        # Memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Initialize disk cache directory
        try:
            os.makedirs(self.disk_cache_dir, exist_ok=True)
            logger.info(f"Cache directory initialized: {self.disk_cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"Cache manager initialized - entries: {max_memory_entries}, size: {max_memory_size_mb}MB, dir: {self.disk_cache_dir}")
    def _start_cleanup_thread(self):
        """Start the cleanup thread"""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Cleanup loop for expired entries"""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                self._cleanup_expired()
                self._enforce_memory_limits()
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    def _cleanup_expired(self):
        """Remove expired entries from memory cache"""
        with self._cache_lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items() 
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
                self.stats.evictions += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up expired cache entries: {len(expired_keys)}")
    
    def _enforce_memory_limits(self):
        """Enforce memory cache size limits"""
        with self._cache_lock:
            # Check entry count limit
            if len(self._memory_cache) > self.max_memory_entries:
                # Remove oldest entries
                sorted_entries = sorted(
                    self._memory_cache.items(),
                    key=lambda x: x[1].last_accessed
                )
                
                entries_to_remove = len(self._memory_cache) - self.max_memory_entries
                for key, _ in sorted_entries[:entries_to_remove]:
                    del self._memory_cache[key]
                    self.stats.evictions += 1            
            # Check memory size limit
            total_size = sum(entry.size_bytes for entry in self._memory_cache.values())
            if total_size > self.max_memory_size_bytes:
                # Remove largest entries first
                sorted_entries = sorted(
                    self._memory_cache.items(),
                    key=lambda x: x[1].size_bytes,
                    reverse=True
                )
                
                for key, entry in sorted_entries:
                    if total_size <= self.max_memory_size_bytes:
                        break
                    total_size -= entry.size_bytes
                    del self._memory_cache[key]
                    self.stats.evictions += 1
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            else:
                return 1024  # Default estimate
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key with namespace"""
        return f"{namespace}:{key}"    
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        with self._cache_lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                if not entry.is_expired():
                    self.stats.hits += 1
                    return entry.access()
                else:
                    # Remove expired entry
                    del self._memory_cache[cache_key]
                    self.stats.evictions += 1
            
            # Check disk cache
            disk_value = self._get_from_disk(cache_key)
            if disk_value is not None:
                self.stats.hits += 1
                return disk_value
            
            self.stats.misses += 1
            return None
    
    def set(self, namespace: str, key: str, value: Any, ttl: float = 300) -> bool:
        """Set value in cache with TTL"""
        cache_key = self._generate_key(namespace, key)
        size_bytes = self._calculate_size(value)
        
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=ttl,
            size_bytes=size_bytes
        )
        
        with self._cache_lock:
            # Store in memory cache
            self._memory_cache[cache_key] = entry
            self.stats.total_entries = len(self._memory_cache)
            
            # Also store in disk cache for persistence
            self._set_to_disk(cache_key, value, ttl)
            
            logger.debug(f"Cache entry stored - namespace: {namespace}, key: {key}, ttl: {ttl}, size: {size_bytes} bytes")
            
            return True    
    def _get_from_disk(self, cache_key: str) -> Optional[Any]:
        """Get value from disk cache"""
        try:
            file_path = os.path.join(self.disk_cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.cache")
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # Check if expired
            if time.time() > (data['created_at'] + data['ttl']):
                os.remove(file_path)
                return None
            
            return data['value']
            
        except Exception as e:
            logger.debug(f"Disk cache read error: {str(e)}")
            return None
    
    def _set_to_disk(self, cache_key: str, value: Any, ttl: float):
        """Set value to disk cache"""
        try:
            file_path = os.path.join(self.disk_cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.cache")
            
            data = {
                'value': value,
                'created_at': time.time(),
                'ttl': ttl
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.debug(f"Disk cache write error: {str(e)}")
    def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(namespace, key)
        
        with self._cache_lock:
            # Remove from memory
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                self.stats.total_entries = len(self._memory_cache)
            
            # Remove from disk
            try:
                file_path = os.path.join(self.disk_cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.cache")
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.debug(f"Disk cache delete error: {str(e)}")
            
            return True
    
    def clear(self, namespace: Optional[str] = None):
        """Clear cache entries"""
        with self._cache_lock:
            if namespace:
                # Clear specific namespace
                keys_to_remove = [
                    key for key in self._memory_cache.keys() 
                    if key.startswith(f"{namespace}:")
                ]
                for key in keys_to_remove:
                    del self._memory_cache[key]
            else:
                # Clear all
                self._memory_cache.clear()
                
            self.stats.total_entries = len(self._memory_cache)    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            memory_usage = sum(entry.size_bytes for entry in self._memory_cache.values())
            
            # Calculate disk usage
            disk_usage = 0
            try:
                for filename in os.listdir(self.disk_cache_dir):
                    if filename.endswith('.cache'):
                        file_path = os.path.join(self.disk_cache_dir, filename)
                        disk_usage += os.path.getsize(file_path)
            except:
                pass
            
            self.stats.memory_usage_bytes = memory_usage
            self.stats.disk_usage_bytes = disk_usage
            
            return {
                "hit_rate": round(self.stats.hit_rate, 2),
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
                "total_entries": self.stats.total_entries,
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "disk_usage_mb": round(disk_usage / (1024 * 1024), 2),
                "memory_entries": len(self._memory_cache)
            }
    
    def shutdown(self):
        """Shutdown cache manager"""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        logger.info("Cache manager shutdown complete")

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def shutdown_cache():
    """Shutdown global cache manager"""
    global _cache_manager
    if _cache_manager:
        _cache_manager.shutdown()
        _cache_manager = None