"""
Jinja2 template engine for model validation module.

ARCHITECTURE:
    Centralized template loading and rendering.
    All templates in templates/ directory as .j2 files.
    
BENEFITS:
    - Template validation at load time (not runtime)
    - Conditional rendering support
    - Better error messages
    - Familiar syntax for non-Python developers
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


# =============================================================================
# Template Engine
# =============================================================================

@lru_cache(maxsize=1)
def get_template_env() -> Environment:
    """
    Get the Jinja2 environment for model validation templates.
    
    Returns:
        Configured Jinja2 Environment instance
        
    Notes:
        - Uses FileSystemLoader to load templates from templates/ directory
        - Caches Environment for performance (lru_cache)
        - trim_blocks/lstrip_blocks for cleaner output
    """
    templates_dir = Path(__file__).parent / "templates"
    
    if not templates_dir.exists():
        logger.warning(f"Templates directory not found: {templates_dir}")
        templates_dir.mkdir(parents=True, exist_ok=True)
    
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_template(template_name: str, **context: Any) -> str:
    """
    Render a template with the given context.
    
    Args:
        template_name: Name of the template file (e.g., "sagemaker_validation.j2")
        **context: Variables to pass to the template
        
    Returns:
        Rendered template as string
        
    Raises:
        TemplateNotFound: If template file doesn't exist
    """
    try:
        env = get_template_env()
        template = env.get_template(template_name)
        return template.render(**context)
    except TemplateNotFound:
        logger.error(f"Template not found: {template_name}")
        raise


def render_template_string(template_str: str, **context: Any) -> str:
    """
    Render a template string directly (for inline templates).
    
    Args:
        template_str: Jinja2 template string
        **context: Variables to pass to the template
        
    Returns:
        Rendered string
    """
    env = get_template_env()
    template = env.from_string(template_str)
    return template.render(**context)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "get_template_env",
    "render_template",
    "render_template_string",
]
