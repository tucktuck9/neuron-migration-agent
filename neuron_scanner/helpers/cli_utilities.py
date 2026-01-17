"""
CLI utilities for entry points (MCP server and Click CLI).

ORGANIZATION:
    1. LoggingConfigurator - Logging setup
    2. ServerRunner - MCP server runner
    3. CLI helper functions - Input parsing and validation utilities

ARCHITECTURE:
    Shared utilities used by both mcp_server.py and cli.py entry points.
    Separated from entry point files to keep them focused on their specific frameworks.
"""

import os
import logging
from typing import Callable, Optional, Protocol, Tuple


# =============================================================================
# Protocol for MCP Server
# =============================================================================

class MCPServerProtocol(Protocol):
    """Protocol for MCP server instances."""
    def run(self) -> None:
        """Run the server."""
        ...


# =============================================================================
# CLI Helper Functions
# =============================================================================

def parse_input_shape(input_shape: str | None) -> tuple[int, ...] | None:
    """
    Parse CLI --input-shape argument into a tuple of ints.
    
    Args:
        input_shape: Comma-separated string of integers (e.g., "1,3,224,224")
    
    Returns:
        Tuple of integers or None if input_shape is None
    """
    if not input_shape:
        return None
    return tuple(int(x.strip()) for x in input_shape.split(","))


def extract_bucket_name(bucket_input: str) -> str:
    """
    Extract bucket name from either a bucket name or S3 URI.
    
    Args:
        bucket_input: Either a bucket name (e.g., "my-bucket") or S3 URI (e.g., "s3://my-bucket/path")
    
    Returns:
        Bucket name without s3:// prefix or path
    """
    if bucket_input.startswith("s3://"):
        # Extract bucket name from S3 URI: s3://bucket-name/path -> bucket-name
        return bucket_input.split("/")[2]
    return bucket_input


def apply_validate_model_env_overrides(
    bucket: str | None,
    role_arn: str | None,
    region: str | None,
) -> None:
    """
    Apply validate-model CLI flags as env vars for the SageMaker validator.

    ARCHITECTURE:
        The model validation service uses Pydantic BaseSettings that reads env vars.
        CLI flags override env vars so users don't need to export variables manually.
    
    Args:
        bucket: S3 bucket name or URI (will extract bucket name if URI provided)
        role_arn: IAM role ARN for SageMaker
        region: AWS region
    """
    if bucket is not None:
        # Extract bucket name if S3 URI was provided
        bucket_name = extract_bucket_name(bucket)
        os.environ["NEURON_VALIDATION_BUCKET"] = bucket_name
    if role_arn is not None:
        os.environ["NEURON_VALIDATION_ROLE_ARN"] = role_arn
    if region is not None:
        os.environ["AWS_REGION"] = region


# =============================================================================
# LoggingConfigurator
# =============================================================================

class LoggingConfigurator:
    """Configures logging for the application."""

    def configure(self):
        """Configure logging with default settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )




# =============================================================================
# ServerRunner
# =============================================================================

class ServerRunner:
    """Handles MCP server startup and execution."""

    def __init__(
        self, 
        fast_mcp_class: Optional[type], 
        create_server_func: Callable[[], MCPServerProtocol]
    ):
        """
        Initialize the server runner.
        
        Args:
            fast_mcp_class: The FastMCP class (None if not installed)
            create_server_func: Factory function that creates the MCP server
        """
        self.fast_mcp_class = fast_mcp_class
        self.create_server_func = create_server_func
        self.logger = logging.getLogger(__name__)

    def run(self) -> bool:
        """
        Run the MCP server.
        
        Returns:
            True if server started successfully, False otherwise.
        """
        if self.fast_mcp_class is None:
            self.logger.error("MCP package not installed. Install with: pip install mcp")
            return False
        
        mcp = self.create_server_func()
        mcp.run()
        return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocol
    "MCPServerProtocol",
    
    # CLI Helper Functions
    "parse_input_shape",
    "extract_bucket_name",
    "apply_validate_model_env_overrides",
    
    # Utilities
    "LoggingConfigurator",
    "ServerRunner",
]
