import asyncio
import logging
from typing import List
from gemini_key_manager.config import ManagerConfig, KeyConfig, ModelConfig, HTTPConfig, RetryConfig
from gemini_key_manager.adapter import ManagedGenAIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def basic_example():
    """Basic usage example with multiple API keys."""
    
    config = ManagerConfig(
        
        models=[
            ModelConfig(
                name="gemini-2.5-pro",
                priority=0,
                max_output_tokens=2048,
                temperature=0.7,
            ),
            ModelConfig(
                name="gemini-2.5-flash",
                priority=1,
                max_output_tokens=1024,
                temperature=0.5,
            ),
        ],
        http=HTTPConfig(
            timeout=60.0,
            max_connections=100,
        ),
        retry=RetryConfig(
            max_attempts=3,
            initial_backoff_s=1.0,
            max_backoff_s=30.0,
            backoff_factor=2.0,
            jitter=True,
        ),
        downgrade_policy="warn",
        enable_monitoring=True,
    )
    
    async with ManagedGenAIClient(config) as client:
        
        response = await client.generate_content(
            model="gemini-1.5-pro",
            contents="Explain quantum computing in simple terms.",
            generation_config={
                "temperature": 0.8,
                "max_output_tokens": 500,
            }
        )
        print(f"Response: {response.text}")
        
        token_count = await client.count_tokens(
            model="gemini-1.5-pro",
            contents="This is a test message to count tokens."
        )
        print(f"Token count: {token_count}")
        
        stats = client.get_stats()
        print(f"Client stats: {stats}")



async def parallel_requests_example():
    """Example of making parallel requests with multiple keys."""
    
    config = ManagerConfig(
        keys=[
            KeyConfig(
                key_id="primary_key",
                secret="secret_key_1",
                priority=0,
                max_concurrent_requests=1,
            ),
            KeyConfig(
                key_id="secondary_key", 
                secret="secret_key_2",
                priority=1,
                max_concurrent_requests=1,
            ),
        ],
        models=[
            ModelConfig(name="gemini-2.0-flash", priority=0),
        ],
    )
    
    async with ManagedGenAIClient(config) as client:
        
        prompts = [
            "Write a haiku about the ocean",
            "Write a haiku about mountains",
            "Write a haiku about forests",
            "Write a haiku about deserts",
            "Write a haiku about cities",
        ]
        
        tasks = [
            client.generate_content(
                contents=prompt,
                request_id=f"parallel-{i}",
            )
            for i, prompt in enumerate(prompts)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for prompt, response in zip(prompts, responses):
            if isinstance(response, Exception):
                print(f"Failed for '{prompt}': {response}")
            else:
                print(f"Prompt: {prompt}")
                print(f"Response: {response.text}\n")


# ============= .env.example =============
"""
# Example environment file for API keys
# Copy this to .env and fill in your actual API keys

# Primary API keys
PRIMARY_API_KEY=your-primary-gemini-api-key-here
SECONDARY_API_KEY=your-secondary-gemini-api-key-here

# Additional backup keys
BACKUP_KEY_1=your-backup-key-1-here
BACKUP_KEY_2=your-backup-key-2-here

# Model preferences
DEFAULT_MODEL=gemini-1.5-pro
FALLBACK_MODEL=gemini-1.5-flash

# Configuration
MAX_RETRIES=3
TIMEOUT_SECONDS=60
ENABLE_MONITORING=true
"""


if __name__ == "__main__":
    asyncio.run(parallel_requests_example())