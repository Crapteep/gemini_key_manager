import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from collections import deque

from gemini_key_manager.config import KeyConfig


@dataclass
class RequestMetrics:
    """Metrics for tracking request performance."""
    timestamp: float
    duration: float
    success: bool
    model: str
    error_type: Optional[str] = None


@dataclass
class KeyState:
    """Tracks the dynamic state of a single API key."""
    config: KeyConfig
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    semaphore: asyncio.Semaphore = field(init=False)
    in_flight_requests: int = 0
    is_disabled: bool = False
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    last_used: Optional[float] = None
    consecutive_errors: int = 0
    request_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=60))
    
    def __post_init__(self):
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

    def is_available(self) -> bool:
        """Checks if the key is ready for a new request."""
        now = time.monotonic()
        
        if (self.is_disabled or 
            now < self.cooldown_until or 
            self.in_flight_requests >= self.config.max_concurrent_requests):
            return False
        
        if self.config.rate_limit_rpm:
            cutoff = now - 60
            while self.request_timestamps and self.request_timestamps[0] < cutoff:
                self.request_timestamps.popleft()
            
            if len(self.request_timestamps) >= self.config.rate_limit_rpm:
                return False
        
        return True
    
    def update_stats(self, success: bool = True, duration: float = 0.0, 
                    model: str = "", error_type: Optional[str] = None) -> None:
        """Update usage statistics."""
        now = time.monotonic()
        self.total_requests += 1
        self.last_used = now
        
        if self.config.rate_limit_rpm:
            self.request_timestamps.append(now)
        
        if not success:
            self.total_errors += 1
            self.consecutive_errors += 1
            
            if self.consecutive_errors >= 5:
                self.is_disabled = True
        else:
            self.consecutive_errors = 0
        
        metric = RequestMetrics(
            timestamp=now,
            duration=duration,
            success=success,
            model=model,
            error_type=error_type
        )
        self.request_history.append(metric)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate for this key."""
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests
    
    @property
    def recent_error_rate(self) -> float:
        """Calculate error rate for recent requests."""
        if not self.request_history:
            return 0.0
        
        recent = list(self.request_history)[-20:]
        if not recent:
            return 0.0
        
        errors = sum(1 for r in recent if not r.success)
        return errors / len(recent)
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time for successful requests."""
        if not self.request_history:
            return 0.0
        
        successful_times = [r.duration for r in self.request_history 
                          if r.success and r.duration > 0]
        
        if not successful_times:
            return 0.0
        
        return sum(successful_times) / len(successful_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for this key."""
        now = time.monotonic()
        return {
            "key_id": self.config.key_id,
            "is_available": self.is_available(),
            "is_disabled": self.is_disabled,
            "in_flight_requests": self.in_flight_requests,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 3),
            "recent_error_rate": round(self.recent_error_rate, 3),
            "consecutive_errors": self.consecutive_errors,
            "cooldown_remaining": max(0, self.cooldown_until - now) if self.cooldown_until > 0 else 0,
            "last_used": self.last_used,
            "average_response_time": round(self.average_response_time, 3),
            "requests_per_minute": len([t for t in self.request_timestamps if t > now - 60])
        }
    
    def reset_stats(self) -> None:
        """Reset statistics (useful for testing or periodic cleanup)."""
        self.total_requests = 0
        self.total_errors = 0
        self.consecutive_errors = 0
        self.request_history.clear()
        self.request_timestamps.clear()