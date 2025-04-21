import os
import logging
from typing import Dict, Any, Optional, List, Union
import time
from pydantic import BaseModel, Field, model_validator

class OpenSearchConfig(BaseModel):
    """Configuration for OpenSearch connection."""
    host: str = Field(..., description="OpenSearch server host")
    port: int = Field(..., description="OpenSearch server port")
    user: Optional[str] = Field(None, description="OpenSearch username")
    password: Optional[str] = Field(None, description="OpenSearch password")
    use_ssl: Optional[bool] = Field(False, description="Use SSL for OpenSearch connection")
    verify_certs: Optional[bool] = Field(False, description="Verify SSL certificates")
    timeout: Optional[int] = Field(30, description="Connection timeout in seconds")
    retry_on_timeout: Optional[bool] = Field(True, description="Retry on timeout")
    max_retries: Optional[int] = Field(3, description="Maximum number of retries")

    
def create_opensearch_client(config: OpenSearchConfig) -> Any:
    """Create an OpenSearch client based on the provided configuration."""

    from opensearchpy import OpenSearch
    
    auth = (config.user, config.password) if config.user and config.password else None

    return OpenSearch(
            hosts=[config.host],
            port=config.port,
            http_auth=auth,
            use_ssl=config.use_ssl,
            verify_certs=config.verify_certs,
            timeout=config.timeout,
            retry_on_timeout=config.retry_on_timeout,
            max_retries=config.max_retries,
            ssl_show_warn=config.verify_certs
        )
        

def create_async_opensearch_client(config: OpenSearchConfig) -> Any:
    """Create an asynchronous OpenSearch client based on the provided configuration."""

    try:
        from opensearchpy import AsyncOpenSearch
    except ImportError:
        raise ImportError("""If you want to use async OpenSearch, please install opensearch-py with async support. 
        (pip install 'opensearch-py[async]')""")

    
    auth = (config.user, config.password) if config.user and config.password else None

    return AsyncOpenSearch(
            hosts=[config.host],
            port=config.port,
            http_auth=auth,
            use_ssl=config.use_ssl,
            verify_certs=config.verify_certs,
            timeout=config.timeout,
            retry_on_timeout=config.retry_on_timeout,
            max_retries=config.max_retries,
            ssl_show_warn=config.verify_certs
        )