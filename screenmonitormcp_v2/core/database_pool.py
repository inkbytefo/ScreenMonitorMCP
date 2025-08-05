"""Database connection pool for ScreenMonitorMCP v2."""

import asyncio
import aiosqlite
import structlog
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class PoolStats:
    """Database pool statistics."""
    total_connections: int
    active_connections: int
    idle_connections: int
    total_queries: int
    failed_queries: int
    average_query_time: float
    pool_created_at: datetime
    last_cleanup: Optional[datetime] = None


class DatabasePool:
    """SQLite connection pool with async support."""
    
    def __init__(
        self,
        db_path: Path,
        max_connections: int = 10,
        min_connections: int = 2,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0  # 1 minute
    ):
        self.db_path = db_path
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval
        
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._active_connections: Dict[str, aiosqlite.Connection] = {}
        self._connection_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = PoolStats(
            total_connections=0,
            active_connections=0,
            idle_connections=0,
            total_queries=0,
            failed_queries=0,
            average_query_time=0.0,
            pool_created_at=datetime.now()
        )
        self._query_times = []
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        try:
            # Create minimum connections
            for _ in range(self.min_connections):
                conn = await self._create_connection()
                await self._pool.put(conn)
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._initialized = True
            logger.info(
                "Database pool initialized",
                db_path=str(self.db_path),
                min_connections=self.min_connections,
                max_connections=self.max_connections
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection."""
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                timeout=self.connection_timeout
            )
            
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=memory")
            
            conn_id = id(conn)
            self._connection_stats[str(conn_id)] = {
                "created_at": datetime.now(),
                "last_used": datetime.now(),
                "query_count": 0
            }
            
            self._stats.total_connections += 1
            logger.debug(f"Created new database connection: {conn_id}")
            
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        await self.initialize()
        
        conn = None
        conn_id = None
        start_time = datetime.now()
        
        try:
            # Try to get connection from pool
            try:
                conn = await asyncio.wait_for(
                    self._pool.get(),
                    timeout=self.connection_timeout
                )
            except asyncio.TimeoutError:
                # Create new connection if pool is empty and under max limit
                async with self._lock:
                    if len(self._active_connections) < self.max_connections:
                        conn = await self._create_connection()
                    else:
                        raise Exception("Connection pool exhausted")
            
            conn_id = str(id(conn))
            
            # Move to active connections
            async with self._lock:
                self._active_connections[conn_id] = conn
                self._stats.active_connections = len(self._active_connections)
                self._stats.idle_connections = self._pool.qsize()
                
                # Update connection stats
                if conn_id in self._connection_stats:
                    self._connection_stats[conn_id]["last_used"] = datetime.now()
                    self._connection_stats[conn_id]["query_count"] += 1
            
            yield conn
            
        except Exception as e:
            self._stats.failed_queries += 1
            logger.error(f"Database connection error: {e}")
            raise
            
        finally:
            # Return connection to pool or close if error
            if conn and conn_id:
                try:
                    async with self._lock:
                        if conn_id in self._active_connections:
                            del self._active_connections[conn_id]
                        
                        self._stats.active_connections = len(self._active_connections)
                    
                    # Check if connection is still valid
                    await conn.execute("SELECT 1")
                    
                    # Return to pool if not full
                    if self._pool.qsize() < self.max_connections:
                        await self._pool.put(conn)
                        self._stats.idle_connections = self._pool.qsize()
                    else:
                        await conn.close()
                        if conn_id in self._connection_stats:
                            del self._connection_stats[conn_id]
                        
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
                    try:
                        await conn.close()
                    except:
                        pass
                    if conn_id in self._connection_stats:
                        del self._connection_stats[conn_id]
            
            # Update query statistics
            query_time = (datetime.now() - start_time).total_seconds()
            self._query_times.append(query_time)
            
            # Keep only last 1000 query times for average calculation
            if len(self._query_times) > 1000:
                self._query_times = self._query_times[-1000:]
            
            self._stats.total_queries += 1
            self._stats.average_query_time = sum(self._query_times) / len(self._query_times)
    
    async def _cleanup_loop(self):
        """Cleanup idle connections periodically."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle_connections()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_idle_connections(self):
        """Remove idle connections that exceed timeout."""
        try:
            now = datetime.now()
            idle_threshold = now - timedelta(seconds=self.idle_timeout)
            
            connections_to_close = []
            
            # Check idle connections in pool
            temp_connections = []
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn_id = str(id(conn))
                    
                    if (conn_id in self._connection_stats and 
                        self._connection_stats[conn_id]["last_used"] < idle_threshold and
                        len(temp_connections) + self._pool.qsize() > self.min_connections):
                        
                        connections_to_close.append((conn, conn_id))
                    else:
                        temp_connections.append(conn)
                        
                except asyncio.QueueEmpty:
                    break
            
            # Return non-idle connections to pool
            for conn in temp_connections:
                await self._pool.put(conn)
            
            # Close idle connections
            for conn, conn_id in connections_to_close:
                try:
                    await conn.close()
                    if conn_id in self._connection_stats:
                        del self._connection_stats[conn_id]
                    logger.debug(f"Closed idle connection: {conn_id}")
                except Exception as e:
                    logger.warning(f"Error closing idle connection {conn_id}: {e}")
            
            if connections_to_close:
                self._stats.last_cleanup = now
                self._stats.idle_connections = self._pool.qsize()
                logger.info(f"Cleaned up {len(connections_to_close)} idle connections")
                
        except Exception as e:
            logger.error(f"Error in cleanup idle connections: {e}")
    
    async def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        async with self._lock:
            self._stats.active_connections = len(self._active_connections)
            self._stats.idle_connections = self._pool.qsize()
            return self._stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the pool."""
        try:
            async with self.get_connection() as conn:
                await conn.execute("SELECT 1")
            
            stats = await self.get_stats()
            
            return {
                "healthy": True,
                "total_connections": stats.total_connections,
                "active_connections": stats.active_connections,
                "idle_connections": stats.idle_connections,
                "pool_utilization": stats.active_connections / self.max_connections,
                "average_query_time": stats.average_query_time
            }
            
        except Exception as e:
            logger.error(f"Database pool health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close all connections and cleanup."""
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Close active connections
            async with self._lock:
                for conn in self._active_connections.values():
                    try:
                        await conn.close()
                    except Exception as e:
                        logger.warning(f"Error closing active connection: {e}")
                
                self._active_connections.clear()
            
            # Close pooled connections
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    await conn.close()
                except Exception as e:
                    logger.warning(f"Error closing pooled connection: {e}")
            
            self._connection_stats.clear()
            self._initialized = False
            
            logger.info("Database pool closed")
            
        except Exception as e:
            logger.error(f"Error closing database pool: {e}")


# Global database pool instance
_db_pool: Optional[DatabasePool] = None


def get_db_pool(db_path: Path) -> DatabasePool:
    """Get or create the global database pool."""
    global _db_pool
    
    if _db_pool is None:
        _db_pool = DatabasePool(db_path)
    
    return _db_pool


async def close_db_pool():
    """Close the global database pool."""
    global _db_pool
    
    if _db_pool:
        await _db_pool.close()
        _db_pool = None