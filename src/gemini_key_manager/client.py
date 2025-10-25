import asyncio
import logging
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from gemini_key_manager.config import ManagerConfig, ModelConfig
from gemini_key_manager.exceptions import (
    NoAvailableKeysError, InvalidModelError, 
    AuthenticationError, ServerError, RateLimitError
)
from gemini_key_manager.models import KeyState

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types for better handling."""
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    BAD_REQUEST = "bad_request"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class GeminiKeyManager:
    """Manages the state and availability of multiple Gemini API keys."""

    def __init__(self, config: ManagerConfig):
        self.config = config
        self._key_states: Dict[str, KeyState] = {
            kc.key_id: KeyState(config=kc) for kc in self.config.keys
        }
        self._sorted_models = sorted(self.config.models, key=lambda m: m.priority)
        self._shutdown = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None


    async def start(self) -> None:
        """Start background tasks for key management."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        if self.config.enable_monitoring and not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_keys())

    async def stop(self) -> None:
        """Stop background tasks."""
        self._shutdown = True
        
        tasks = [t for t in [self._cleanup_task, self._monitor_task] if t]
        
        for task in tasks:
            task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._cleanup_task = None
        self._monitor_task = None

    async def _periodic_cleanup(self) -> None:
        """Periodically check and reset cooldowns."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.cleanup_interval_s)
                
                now = time.monotonic()
                for key_state in self._key_states.values():
                    async with key_state.lock:
                        if key_state.cooldown_until > 0 and now >= key_state.cooldown_until:
                            logger.info(f"Key '{key_state.config.key_id}' cooldown expired.")
                            key_state.cooldown_until = 0.0
                        
                        if key_state.is_disabled and key_state.recent_error_rate < 0.2:
                            logger.info(f"Re-enabling key '{key_state.config.key_id}' after error rate improvement.")
                            key_state.is_disabled = False
                            key_state.consecutive_errors = 0
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def _monitor_keys(self) -> None:
        """Monitor key performance and log statistics."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)
                
                stats = self.get_stats()
                available_keys = sum(1 for s in stats.values() if s['is_available'])
                total_requests = sum(s['total_requests'] for s in stats.values())
                total_errors = sum(s['total_errors'] for s in stats.values())
                
                logger.info(
                    f"Key Manager Stats - Available: {available_keys}/{len(self._key_states)}, "
                    f"Total Requests: {total_requests}, Total Errors: {total_errors}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor task: {e}")

    def _select_model(self, requested_model: Optional[str]) -> Tuple[ModelConfig, Optional[str]]:
        """Select the best available model based on priority and key availability."""
        downgraded_from: Optional[str] = None

        if self.config.downgrade_policy == "strict" and requested_model:
            model = next((m for m in self._sorted_models if m.name == requested_model), None)
            if not model:
                raise InvalidModelError(f"Model '{requested_model}' not found in configuration.")

            if not any(k.is_available() and self._is_key_allowed_for_model(k, model.name)
                      for k in self._key_states.values()):
                raise NoAvailableKeysError(f"No keys available for model '{model.name}' (strict mode).")
            return model, None

        if requested_model:
            primary_model = next((m for m in self._sorted_models if m.name == requested_model), None)
            if not primary_model:
                raise InvalidModelError(f"Model '{requested_model}' not found in configuration.")
            model_priority_list = [primary_model] + [m for m in self._sorted_models if m.name != requested_model]
        else:
            model_priority_list = self._sorted_models

        for model in model_priority_list:
            if any(k.is_available() and self._is_key_allowed_for_model(k, model.name)
                  for k in self._key_states.values()):
                if requested_model and model.name != requested_model:
                    downgraded_from = requested_model
                return model, downgraded_from

        raise NoAvailableKeysError("All keys for all eligible models are on cooldown or disabled.")

    async def acquire_key(self, model_name: str, timeout: float = 300.0) -> KeyState:
        """
        Acquire the best available key for a specific model.
        Waits (up to `timeout`) until a key becomes available instead of failing immediately.
        """
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            candidate_keys = sorted(
                [
                    ks for ks in self._key_states.values()
                    if self._is_key_allowed_for_model(ks, model_name) and not ks.is_disabled
                ],
                key=lambda k: (
                    k.recent_error_rate,
                    k.in_flight_requests,
                    k.average_response_time
                ),
            )

            for key_state in candidate_keys:
                if key_state.is_available():
                    try:
                        acquired = await asyncio.wait_for(
                            key_state.semaphore.acquire(), timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        continue

                    if acquired:
                        async with key_state.lock:
                            key_state.in_flight_requests += 1
                            logger.debug(
                                f"Acquired key '{key_state.config.key_id}' "
                                f"for model '{model_name}' (in_flight={key_state.in_flight_requests})"
                            )
                            return key_state

            await asyncio.sleep(0.2)

        raise NoAvailableKeysError(
            f"Could not acquire a key for model '{model_name}' within {timeout}s."
        )



    async def release_key(self, key_state: KeyState):
        async with key_state.lock:
            if key_state.in_flight_requests > 0:
                key_state.in_flight_requests -= 1
            try:
                key_state.semaphore.release()
            except ValueError:
                logger.warning(f"Semaphore already released for key '{key_state.config.key_id}'")
            logger.debug(
                f"Released key '{key_state.config.key_id}' "
                f"(now {key_state.in_flight_requests} in flight)"
            )





    def classify_error(self, error: Exception) -> ErrorType:
        """Classify an error to determine the appropriate handling strategy."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        patterns = {
            ErrorType.RATE_LIMIT: [
                r"rate.?limit", r"too.?many.?requests", r"429",
                r"resource.?exhausted", r"throttl"
            ],
            ErrorType.QUOTA_EXCEEDED: [
                r"quota", r"daily.?limit", r"exceed.?limit", r"billing"
            ],
            ErrorType.AUTHENTICATION: [
                r"unauthorized", r"401", r"invalid.?api.?key", 
                r"authentication", r"unauthenticated"
            ],
            ErrorType.PERMISSION_DENIED: [
                r"permission.?denied", r"403", r"forbidden", r"access.?denied"
            ],
            ErrorType.BAD_REQUEST: [
                r"bad.?request", r"400", r"invalid.?argument", 
                r"invalid.?parameter", r"malformed"
            ],
            ErrorType.SERVER_ERROR: [
                r"500", r"502", r"503", r"504", r"server.?error",
                r"internal.?error", r"unavailable", r"bad.?gateway"
            ],
            ErrorType.NETWORK_ERROR: [
                r"connection", r"network", r"dns", r"socket", r"ssl"
            ],
            ErrorType.TIMEOUT: [
                r"timeout", r"timed.?out", r"deadline.?exceeded"
            ],
        }
        
        for error_type, pattern_list in patterns.items():
            if any(re.search(pattern, error_str) or re.search(pattern, error_type_name) 
                  for pattern in pattern_list):
                return error_type
        
        return ErrorType.UNKNOWN

    async def handle_error(self, error: Exception, key_state: KeyState, req_id: str) -> None:
        """Handle an error and update key state accordingly."""
        error_type = self.classify_error(error)
        
        async with key_state.lock:
            key_state.update_stats(
                success=False, 
                error_type=error_type.value
            )
            
            cooldown = self._get_cooldown_duration(error_type, error)
            
            if error_type == ErrorType.AUTHENTICATION:
                key_state.is_disabled = True
                logger.error(f"[{req_id}] Key '{key_state.config.key_id}' is invalid. Disabling permanently.")
                raise AuthenticationError(f"Invalid API key: {key_state.config.key_id}")
            
            if error_type == ErrorType.SERVER_ERROR and "overloaded" in str(error).lower():
                model_marked_overloaded = True
                
            elif error_type == ErrorType.QUOTA_EXCEEDED:
                key_state.cooldown_until = time.monotonic() + cooldown
                logger.error(f"[{req_id}] Key '{key_state.config.key_id}' exceeded quota. Cooldown for {cooldown/3600:.1f}h.")
                raise RateLimitError(f"Quota exceeded for key: {key_state.config.key_id}")
                
            elif error_type in [ErrorType.RATE_LIMIT, ErrorType.SERVER_ERROR]:
                key_state.cooldown_until = time.monotonic() + cooldown
                logger.warning(f"[{req_id}] Key '{key_state.config.key_id}' {error_type.value}. Cooldown for {cooldown}s.")
                
            elif error_type == ErrorType.BAD_REQUEST:
                logger.warning(f"[{req_id}] Bad request for key '{key_state.config.key_id}': {error}")
                
            else:
                key_state.cooldown_until = time.monotonic() + 5.0
                logger.warning(f"[{req_id}] Unknown error for key '{key_state.config.key_id}': {type(error).__name__}: {error}")

    def _get_cooldown_duration(self, error_type: ErrorType, error: Exception) -> float:
        """Determine appropriate cooldown duration based on error type."""
        error_str = str(error).lower()
        
        match = re.search(r"retry[- ]?after[^\d]*(\d+)", error_str)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        
        cooldowns = {
            ErrorType.QUOTA_EXCEEDED: 24 * 3600,
            ErrorType.RATE_LIMIT: 60,
            ErrorType.SERVER_ERROR: 30,
            ErrorType.NETWORK_ERROR: 10,
            ErrorType.TIMEOUT: 5,
            ErrorType.UNKNOWN: 5,
        }
        
        return cooldowns.get(error_type, 5.0)

    def _is_key_allowed_for_model(self, key_state: KeyState, model_name: str) -> bool:
        """Check if a key is allowed to be used with a specific model."""
        allowed_list = key_state.config.allowed_models
        if not allowed_list:
            return True
        
        normalized_model = model_name.replace("models/", "")
        return any(
            normalized_model == allowed.replace("models/", "") 
            for allowed in allowed_list
        )

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return current statistics for all keys."""
        return {key_id: key_state.get_stats() 
                for key_id, key_state in self._key_states.items()}
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return next((m for m in self._sorted_models if m.name == model_name), None)