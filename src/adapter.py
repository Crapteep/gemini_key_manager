import asyncio
import logging
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from types import TracebackType

import google.genai as genai
from google.genai.types import (
    GenerateContentResponse, 
    Content, 
    Part,
    GenerateContentConfig,
    SafetySetting,
    Tool,
)

from client import GeminiKeyManager
from config import ManagerConfig
from exceptions import (
    MaxRetriesExceededError, 
    RateLimitError, 
    NoAvailableKeysError,
    AuthenticationError,
    ServerError,
    InvalidModelError
)

logger = logging.getLogger(__name__)


class ManagedGenAIClient:
    """A resilient, multi-key client that wraps the google-genai library."""

    def __init__(self, config: ManagerConfig):
        self._manager = GeminiKeyManager(config)
        self._is_closed = False
        self._clients_cache: Dict[str, genai.Client] = {}
        self._active_streams: List[AsyncIterator] = []

    async def __aenter__(self) -> "ManagedGenAIClient":
        """Async context manager entry."""
        await self._manager.start()
        logger.info("ManagedGenAIClient initialized and started")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if not self._is_closed:
            self._is_closed = True
            
            for stream in self._active_streams:
                try:
                    await stream.aclose()
                except Exception:
                    pass
            self._active_streams.clear()
            
            await self._manager.stop()
            
            self._clients_cache.clear()
            
            logger.info("ManagedGenAIClient closed")

    def _get_or_create_client(self, api_key: str) -> genai.Client:
        """Get or create a client for a specific API key."""
        if api_key not in self._clients_cache:
            client_config = {
                "api_key": api_key,
            }

            self._clients_cache[api_key] = genai.Client(**client_config)

            logger.info(f"Created GenAI Client for {api_key[:4]}****")

        return self._clients_cache[api_key]

  
    async def generate_content(
        self,
        contents: Union[str, List[Union[str, Content, Part, Dict[str, Any]]]],
        model: Optional[str] = None,
        generation_config: Optional[Union[GenerateContentConfig, Dict[str, Any]]] = None,
        safety_settings: Optional[List[Union[SafetySetting, Dict[str, Any]]]] = None,
        tools: Optional[List[Union[Tool, Dict[str, Any]]]] = None,
        request_id: Optional[str] = None,
        system_instruction: Optional[Union[str, Content, Part]] = None,
    ) -> GenerateContentResponse:
        """Generate content with multi-key resilience and smart fallback on overload (503)."""

        if model:
            requested_model_cfg = self._manager.get_model_config(model)
            if not requested_model_cfg:
                raise InvalidModelError(f"Requested model '{model}' not found in configuration.")
            model_priority_list = [requested_model_cfg] + [
                m for m in self._manager._sorted_models if m.name != model
            ]
            if self._manager.config.downgrade_policy == "strict":
                model_priority_list = [requested_model_cfg]
        else:
            model_priority_list = self._manager._sorted_models

        if self._is_closed:
            raise RuntimeError("Client is closed")

        req_log_id = request_id or f"req-{random.randint(10000, 99999)}"
        last_error: Optional[Exception] = None
        attempt = 0

        if isinstance(contents, str):
            contents = contents

        for model_cfg in model_priority_list:
            skip_model_due_to_overload = False
            overload_count = 0

            selected_model = None
            while True:
                try:
                    selected_model, downgraded_from = self._manager._select_model(model_cfg.name)
                    break
                except NoAvailableKeysError:
                    logger.info(
                        f"[{req_log_id}] No keys available for model '{model_cfg.name}'. Waiting..."
                    )
                    await asyncio.sleep(0.5)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"[{req_log_id}] Unexpected error when selecting key for '{model_cfg.name}': {e}"
                    )
                    break


            for _ in range(self._manager.config.retry.max_attempts):
                attempt += 1
                key_state = None

                try:
                    key_state = await self._manager.acquire_key(selected_model.name)
                    logger.info(
                        f"[{req_log_id}] Attempt {attempt}: Using key '{key_state.config.key_id}' "
                        f"for model '{selected_model.name}'"
                    )

                    client = self._get_or_create_client(key_state.config.secret.get_secret_value())

                    request_params = {"model": selected_model.name, "contents": contents}
                    if generation_config:
                        if isinstance(generation_config, dict):
                            model_conf = self._manager.get_model_config(selected_model.name)
                            if model_conf:
                                generation_config.setdefault("max_output_tokens", model_conf.max_output_tokens)
                                generation_config.setdefault("temperature", model_conf.temperature)
                            request_params["config"] = GenerateContentConfig(**generation_config)
                        else:
                            request_params["config"] = generation_config
                    if safety_settings:
                        request_params["safety_settings"] = safety_settings
                    if tools:
                        request_params["tools"] = tools
                    if system_instruction:
                        request_params["system_instruction"] = system_instruction

                    loop = asyncio.get_running_loop()
                    request_start = time.time()
                    response = await loop.run_in_executor(
                        None,
                        lambda: client.models.generate_content(**request_params),
                    )
                    duration = time.time() - request_start

                    key_state.update_stats(success=True, duration=duration, model=selected_model.name)
                    overload_count = 0

                    logger.info(
                        f"[{req_log_id}] Request successful on model '{selected_model.name}' in {duration:.2f}s"
                    )
                    return response

                except Exception as e:
                    last_error = e
                    logger.error(
                        f"[{req_log_id}] Exception type: {type(e).__name__}, "
                        f"Exception module: {type(e).__module__}, "
                        f"Exception str: {str(e)}"
                    )

                    if key_state:
                        try:
                            await self._manager.handle_error(e, key_state, req_log_id)
                        except (AuthenticationError, RateLimitError) as specific:
                            if isinstance(specific, AuthenticationError):
                                raise
                            last_error = specific

                    if self._is_model_overloaded_error(e):
                        overload_count += 1
                        if overload_count >= 2:
                            logger.warning(
                                f"[{req_log_id}] Model '{selected_model.name}' returned 503 twice. "
                                f"Falling back to next model."
                            )
                            skip_model_due_to_overload = True
                            logger.info(
                                f"[{req_log_id}] Falling back from '{selected_model.name}' "
                                f"to next model due to repeated 503s."
                            )
                            break
                        else:
                            logger.warning(
                                f"[{req_log_id}] Model '{selected_model.name}' overloaded once (503). "
                                f"Retrying one more time before fallback."
                            )
                            await asyncio.sleep(self._get_backoff_duration(attempt))
                            continue

                    backoff = self._get_backoff_duration(attempt)
                    logger.warning(
                        f"[{req_log_id}] Attempt {attempt} failed with model '{selected_model.name}'. "
                        f"Retrying in {backoff:.2f}s due to {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(backoff)

                finally:
                    if key_state:
                        await self._manager.release_key(key_state)
                        logger.debug(f"[{req_log_id}] Released key '{key_state.config.key_id}' after request")


                if skip_model_due_to_overload:
                    break

        error_type = self._manager.classify_error(last_error) if last_error else None
        if error_type and error_type.value in ["rate_limit", "quota_exceeded"]:
            raise RateLimitError(
                f"Request failed due to rate limits after {self._manager.config.retry.max_attempts} attempts."
            ) from last_error

        raise MaxRetriesExceededError(
            f"Request '{req_log_id}' failed after {attempt} attempts across all models. "
            f"Last error: {last_error}"
        ) from last_error

    def _is_model_overloaded_error(self, e: Exception) -> bool:
                    """Check if error indicates model is overloaded."""
                    if isinstance(e, ServerError):
                        error_str = str(e).lower()
                        return "503" in error_str and ("unavailable" in error_str or "overloaded" in error_str)
                    return False
    

    def _get_backoff_duration(self, attempt: int) -> float:
        """Calculate backoff duration with exponential backoff and optional jitter."""
        if attempt <= 1:
            return 0.1

        retry_config = self._manager.config.retry
        
        backoff = retry_config.initial_backoff_s * (retry_config.backoff_factor ** (attempt - 1))
        backoff = min(backoff, retry_config.max_backoff_s)
        
        if retry_config.jitter:
            jitter = random.uniform(0, min(1.0, backoff * 0.1))
            backoff += jitter
        
        return backoff

    async def count_tokens(
        self,
        model: str,
        contents: Union[str, List[Union[str, Content, Part, Dict[str, Any]]]],
        request_id: Optional[str] = None,
    ) -> int:
        """
        Count tokens for given content.
        
        Args:
            model: The model to use for token counting
            contents: The content to count tokens for
            request_id: Optional request ID for logging
            
        Returns:
            The total number of tokens
        """
        req_log_id = request_id or f"count-{random.randint(10000, 99999)}"


        for key_config in sorted(self._manager.config.keys, key=lambda k: k.priority):
            key_state = self._manager._key_states[key_config.key_id]
            
            if key_state.is_available() and self._manager._is_key_allowed_for_model(key_state, model):
                try:
                    client = self._get_or_create_client(key_config.secret.get_secret_value())
                    
                    token_contents = contents if isinstance(contents, list) else [contents]
                    
                    response = await asyncio.wait_for(
                        client.models.count_tokens(
                            model=model,
                            contents=token_contents,
                        ),
                        timeout=10.0
                    )
                    
                    logger.debug(
                        f"[{req_log_id}] Token counting successful with key '{key_config.key_id}': "
                        f"{response.total_tokens} tokens"
                    )
                    
                    return response.total_tokens
                    
                except Exception as e:
                    logger.warning(
                        f"[{req_log_id}] Token counting failed with key '{key_config.key_id}': {e}"
                    )
                    continue

        raise NoAvailableKeysError(f"No keys available for token counting with model '{model}'.")

    async def embed_content(
        self,
        model: str,
        content: Union[str, List[str]],
        task_type: Optional[str] = None,
        title: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for content.
        
        Args:
            model: The embedding model to use
            content: The content to embed
            task_type: Optional task type for the embedding
            title: Optional title for the content
            request_id: Optional request ID for logging
            
        Returns:
            List of embeddings
        """
        req_log_id = request_id or f"embed-{random.randint(10000, 99999)}"

        
        contents = [content] if isinstance(content, str) else content

        for attempt in range(1, self._manager.config.retry.max_attempts + 1):
            key_state = None
            
            try:
                key_state = await self._manager.acquire_key(model)
                client = self._get_or_create_client(key_state.config.secret.get_secret_value())
                
                request_params = {
                    "model": model,
                    "content": contents,
                }
                if task_type:
                    request_params["task_type"] = task_type
                if title:
                    request_params["title"] = title
                
                response = await asyncio.wait_for(
                    client.models.embed_content(**request_params),
                    timeout=self._manager.config.http.timeout
                )
                
                key_state.update_stats(success=True, model=model)
                
                return response.embeddings
                
            except Exception as e:
                if key_state:
                    await self._manager.handle_error(e, key_state, req_log_id)
                
                if attempt < self._manager.config.retry.max_attempts:
                    await asyncio.sleep(self._get_backoff_duration(attempt))
                else:
                    raise MaxRetriesExceededError(
                        f"Embedding generation failed after {attempt} attempts"
                    ) from e
                    
            finally:
                if key_state:
                    await self._manager.release_key(key_state)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for all keys and the client."""
        return {
            "keys": self._manager.get_stats(),
            "active_streams": len(self._active_streams),
            "cached_clients": len(self._clients_cache),
            "is_closed": self._is_closed
        }