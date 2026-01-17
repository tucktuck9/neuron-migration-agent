"""
CLI utilities for entry points (MCP server and Click CLI).

ORGANIZATION:
    1. LoggingConfigurator - Logging setup
    2. ServerRunner - MCP server runner

ARCHITECTURE:
    Shared utilities used by both mcp_server.py and cli.py entry points.
    Separated from entry point files to keep them focused on their specific frameworks.
"""

import logging
from typing import Callable, Optional, Protocol


# =============================================================================
# Protocol for MCP Server
# =============================================================================

class MCPServerProtocol(Protocol):
    """Protocol for MCP server instances."""
    def run(self) -> None:
        """Run the server."""
        ...


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
    
    # Utilities
    "LoggingConfigurator",
    "ServerRunner",
]
