from typing import List, Literal, Optional
from pydantic import BaseModel, Field, SecretStr, field_validator


class HTTPConfig(BaseModel):
    """HTTP client settings."""
    timeout: float = Field(60.0, description="Default request timeout in seconds.")
    max_connections: int = Field(100, ge=1, description="Maximum number of connections.")
    max_keepalive_connections: int = Field(20, ge=1, description="Maximum keepalive connections.")
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class RetryConfig(BaseModel):
    """Retry and backoff policy settings."""
    max_attempts: int = Field(3, ge=1, description="Maximum number of attempts per request.")
    initial_backoff_s: float = Field(1.0, ge=0, description="Initial backoff delay for transient errors.")
    max_backoff_s: float = Field(30.0, ge=0, description="Maximum backoff delay.")
    backoff_factor: float = Field(2.0, ge=1, description="Multiplier for exponential backoff.")
    jitter: bool = Field(True, description="Add random jitter to backoff delays.")
    
    @field_validator('max_backoff_s')
    @classmethod
    def validate_max_backoff(cls, v: float, info) -> float:
        if 'initial_backoff_s' in info.data and v < info.data['initial_backoff_s']:
            raise ValueError("max_backoff_s must be >= initial_backoff_s")
        return v


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str = Field(..., description="The full name of the model, e.g., 'gemini-1.5-pro'.")
    priority: int = Field(0, description="Priority for selection (lower is higher priority).")
    max_output_tokens: Optional[int] = Field(None, description="Maximum tokens in response.")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature.")
    
    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        
        valid_prefixes = ['gemini-', 'models/gemini-']
        if not any(v.lower().startswith(prefix) for prefix in valid_prefixes):
            import warnings
            warnings.warn(f"Model name '{v}' doesn't follow standard Gemini naming convention")
        
        return v.strip()


class KeyConfig(BaseModel):
    """Configuration for a single API key."""
    key_id: str = Field(..., description="A unique, friendly identifier for the key.")
    secret: SecretStr = Field(..., description="The Gemini API key.")
    priority: int = Field(0, description="Priority for selection (lower is higher priority).")
    max_concurrent_requests: int = Field(5, ge=1, le=100, description="Maximum number of in-flight requests for this key.")
    allowed_models: Optional[List[str]] = Field(None, description="Optional list of model names this key can be used with.")
    rate_limit_rpm: Optional[int] = Field(None, ge=1, description="Rate limit in requests per minute.")
    
    @field_validator('key_id')
    @classmethod
    def validate_key_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Key ID cannot be empty")
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v.strip()):
            raise ValueError("Key ID must be alphanumeric (with optional - or _)")
        return v.strip()


class ManagerConfig(BaseModel):
    """Root configuration for the GeminiKeyManager."""
    keys: List[KeyConfig] = Field(..., min_length=1)
    models: List[ModelConfig] = Field(..., min_length=1)
    http: HTTPConfig = Field(default_factory=HTTPConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    downgrade_policy: Literal["warn", "allow", "strict"] = Field(
        "warn", 
        description="Policy for falling back to lower-priority models. 'strict' prevents fallback."
    )
    enable_monitoring: bool = Field(True, description="Enable performance monitoring and statistics.")
    cleanup_interval_s: float = Field(60.0, ge=10, description="Interval for cleanup tasks in seconds.")
    
    @field_validator('keys')
    @classmethod
    def validate_unique_key_ids(cls, v: List[KeyConfig]) -> List[KeyConfig]:
        key_ids = [k.key_id for k in v]
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("All key_id values must be unique")
        return v
    
    @field_validator('models')
    @classmethod
    def validate_unique_model_names(cls, v: List[ModelConfig]) -> List[ModelConfig]:
        model_names = [m.name for m in v]
        if len(model_names) != len(set(model_names)):
            raise ValueError("All model names must be unique")
        return v