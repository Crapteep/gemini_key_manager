# Gemini Key Manager

A robust, async-first Python library for managing multiple Gemini API keys.  
It provides concurrency-aware scheduling, automatic failover for rate limits and server errors,  
and a "best-first" model selection strategy with fallback.

## Features

- **Async-Only API** – built with `asyncio` and `httpx` for high-performance, non-blocking I/O.  
- **Multi-Key Load Balancing** – distributes requests across a pool of API keys, prioritizing the least-loaded keys.  
- **Intelligent Failover** – automatically retries requests on other keys upon encountering rate limits (429), quota errors, or transient server errors (5xx).  
- **Per-Key Cooldowns** – honors `Retry-After` headers for rate-limited keys, taking them temporarily out of rotation.  
- **Model Fallback** – uses a configurable model priority list; if keys for the best model are unavailable, it gracefully downgrades to the next best one.  
- **Resilience** – implements exponential backoff with jitter for transient errors and permanently disables invalid keys (401/403).  
- **Strongly-Typed Configuration** – uses Pydantic for clear, validated, and self-documenting configuration.  
- **Clean & Minimal** – small dependency footprint (`httpx`, `pydantic`) and a simple public API.

## Installation

This project uses **Poetry** for dependency management and packaging.

```bash
poetry install
```

If you prefer, you can also install dependencies manually:

```bash
pip install pydantic httpx
```

*(Note: As this is a custom library, you would typically install it from a git repository or a private package index.  
For local use, just save the library code in a `gemini_manager/` directory and install with Poetry as shown above.)*

## Usage

Example usage scripts are available in `examples.py`.

They demonstrate:
- how to configure the manager,
- how to run multiple concurrent requests,
- and how automatic failover and model fallback are handled internally.

To run the examples:

```bash
poetry run python examples.py
```

## License

MIT License – see `LICENSE` file for details.
