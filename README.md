# Gemini Key Manager

A robust, async-first Python library for managing multiple Gemini API keys. It provides concurrency-aware scheduling, automatic failover for rate limits and server errors, and a "best-first" model selection strategy with fallback.

## Features

-   **Async-Only API**: Built with `asyncio` and `httpx` for high-performance, non-blocking I/O.
-   **Multi-Key Load Balancing**: Distributes requests across a pool of API keys, prioritizing the least-loaded keys.
-   **Intelligent Failover**: Automatically retries requests on other keys upon encountering rate limits (429), quota errors, or transient server errors (5xx).
-   **Per-Key Cooldowns**: Honors `Retry-After` headers for rate-limited keys, taking them temporarily out of rotation.
-   **Model Fallback**: Uses a configurable model priority list. If keys for the best model are unavailable, it can gracefully downgrade to the next best model.
-   **Resilience**: Implements exponential backoff with jitter for retrying transient errors and permanently disables invalid keys (401/403).
-   **Strongly-Typed Configuration**: Uses Pydantic for clear, validated, and self-documenting configuration.
-   **Clean & Minimal**: Small dependency footprint (`httpx`, `pydantic`) and a simple public API.

## Installation

```bash
pip install pydantic httpx
```
*(Note: As this is a custom library, you would typically install it from a git repository or a private package index. For this example, just save the library code in a `gemini_manager` directory.)*

## Quickstart

### 1. Configure the Manager

Create a configuration object using the Pydantic models. You can load this from a file, environment variables, or define it directly.

```python
from gemini_manager.config import ManagerConfig, KeyConfig, ModelConfig

API_KEY_1_SECRET = "your_first_api_key_here"
API_KEY_2_SECRET = "your_second_api_key_here"

config = ManagerConfig(
    keys=[
        KeyConfig(
            key_id="primary-key-1",
            secret=API_KEY_1_SECRET,
            priority=0,
            max_concurrent_requests=5,
        ),
        KeyConfig(
            key_id="backup-key-2",
            secret=API_KEY_2_SECRET,
            priority=1,
            max_concurrent_requests=10,
        ),
    ],
    models=[
        ModelConfig(name="gemini-1.5-pro-latest", priority=0),
        ModelConfig(name="gemini-1.5-flash-latest", priority=1),
    ]
)
```

### 2. Use the Client

Use the manager as an async context manager to handle multiple concurrent requests.

```python
import asyncio
import logging
from gemini_manager.client import GeminiKeyManager
from example_config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    prompt = "Explain the importance of project management in 50 words."
    payload = [{"parts": [{"text": prompt}]}]

    async with GeminiKeyManager(config) as manager:
        
        tasks = [manager.generate(contents=payload, request_id=f"req_{i}") for i in range(10)]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logging.error(f"Request {i} failed: {res}")
            elif res.success:
            
                logging.info(f"Request {i} succeeded on key '{res.used_key_id}' with model '{res.used_model}'.")
            else:
                logging.warning(f"Request {i} failed with error: {res.error}")

if __name__ == "__main__":
    asyncio.run(main())

```

This example will send 10 concurrent requests, and the manager will distribute them across your two configured keys according to their `max_concurrent_requests` and `in_flight` status.