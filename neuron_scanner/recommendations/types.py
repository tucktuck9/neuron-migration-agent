"""
Shared types for model validation and SageMaker compilation.

ARCHITECTURE:
    Pydantic-first design - centralized validation result types.
    Used by both name-based model lookup and S3-based SageMaker validation.

ORGANIZATION:
    1. ValidationStatus - Enum for validation outcomes
    2. ValidationResult - Pydantic model for validation results
"""

import logging
from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Status Enum
# =============================================================================

class ValidationStatus(str, Enum):
    """Status of Neuron validation."""
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    SKIPPED = "skipped"
    ERROR = "error"


# =============================================================================
# Pydantic Models for Validation Results
# =============================================================================

class ValidationResult(BaseModel):
    """
    Result of Neuron API validation for a single model.
    
    ARCHITECTURE:
        Pydantic model storing ground-truth compilation result.
        Immutable for safety.
        Used by both name-based lookup and SageMaker validation.
        
    FIELDS:
        Core fields are always set; optional fields for error details and metadata.
    """
    
    model_config = {"frozen": True}
    
    # Core result (always set)
    status: ValidationStatus = Field(..., description="Validation outcome")
    model_name: str = Field(
        ..., 
        min_length=1, 
        max_length=256, 
        description="Name of the model"
    )
    file_path: str = Field(
        ..., 
        min_length=1, 
        description="S3 URI or file path identifier"
    )
    
    # Human-readable message
    message: str = Field(
        default="", 
        max_length=2048, 
        description="Human-readable result message"
    )
    
    # Error details (set when status is INCOMPATIBLE or ERROR)
    error_details: Optional[str] = Field(
        None, 
        max_length=4096,
        description="Full error message from SageMaker"
    )
    
    # Internal categorization (used for display logic, not shown to user)
    error_category: Optional[str] = Field(
        None,
        max_length=64,
        pattern=r"^[A-Z_]+$",
        description="Internal category for display routing (e.g., MISSING_CONFIG, NO_MODELS_FOUND)"
    )
    
    # Compilation metadata
    input_shape: Optional[Tuple[int, ...]] = Field(
        None, 
        description="Input shape used for compilation"
    )
    compilation_time_ms: Optional[float] = Field(
        None, 
        ge=0,
        description="Compilation time in milliseconds"
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ValidationStatus",
    "ValidationResult",
]
