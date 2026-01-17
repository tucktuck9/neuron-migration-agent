"""
Model support recommendation display formatting.

This module formats model classification results for user-friendly output.

ORGANIZATION:
    1. SageMakerValidationDisplayFormatter (formats SageMaker validation results)

ARCHITECTURE:
    Jinja2 template-based display - templates in templates/ directory.
    Clean separation of data processing from template rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List

from neuron_scanner.model_validation.types import ValidationResult, ValidationStatus
from neuron_scanner.model_validation.template_engine import get_template_env

logger = logging.getLogger(__name__)


# =============================================================================
# SageMaker validation display formatter
# =============================================================================

class SageMakerValidationDisplayFormatter:
    """
    Format SageMaker validation results for display.
    
    RESPONSIBILITY:
        Takes validation results from SageMaker validator
        and formats them into user-friendly markdown using Jinja2 templates.
    
    ARCHITECTURE:
        Jinja2 template-based - loads sagemaker_validation.j2 template
    """
    
    def __init__(self):
        """Initialize formatter with Jinja2 environment."""
        self._env = get_template_env()
        self._template = self._env.get_template("sagemaker_validation.j2")
    
    def format(self, results: List[ValidationResult]) -> str:
        """
        Format SageMaker validation results using Jinja2 templates.
        
        Args:
            results: List of ValidationResult from SageMaker validation
            
        Returns:
            Formatted markdown section
        """
        if not results:
            return ""
        
        try:
            # Check if validation was not configured (error status with MISSING_CONFIG)
            is_not_configured = any(
                result.error_category == "MISSING_CONFIG" for result in results
            )
            
            # Check if no models were found
            no_models_found = (
                len(results) == 1 and 
                results[0].error_category == "NO_MODELS_FOUND"
            )
            
            if no_models_found:
                return self._template.module.no_models_found()
            
            # Select appropriate mode notice
            if is_not_configured:
                mode_notice = self._template.module.mode_not_configured()
            else:
                mode_notice = self._template.module.mode_configured()
            
            # Format individual results
            result_sections = []
            compatible_count = 0
            incompatible_count = 0
            skipped_count = 0
            
            for result in results:
                if result.status == ValidationStatus.COMPATIBLE:
                    compatible_count += 1
                    
                    # Format compilation time if available
                    compilation_time = None
                    if result.compilation_time_ms:
                        compilation_time = f"{result.compilation_time_ms:.0f}"
                    
                    section = self._template.module.result_compatible(
                        model_name=result.model_name,
                        file_path=self._display_path(result.file_path),
                        compilation_time=compilation_time,
                    )
                    result_sections.append(section)
                
                elif result.status == ValidationStatus.INCOMPATIBLE:
                    incompatible_count += 1
                    
                    section = self._template.module.result_incompatible(
                        model_name=result.model_name,
                        file_path=self._display_path(result.file_path),
                        error_details=result.error_details or result.message or "No error details",
                    )
                    result_sections.append(section)
                
                else:  # SKIPPED or ERROR
                    skipped_count += 1
                    
                    section = self._template.module.result_skipped(
                        model_name=result.model_name,
                        file_path=self._display_path(result.file_path),
                        message=result.message or "Unknown reason",
                    )
                    result_sections.append(section)
            
            # Build recommendation message from template
            recommendation = self._template.module.recommendation_message(
                compatible=compatible_count,
                incompatible=incompatible_count,
            ).strip()
            
            # Render summary
            summary = self._template.module.summary(
                compatible=compatible_count,
                incompatible=incompatible_count,
                skipped=skipped_count,
                recommendation=recommendation,
            )
            
            # Combine into final section
            return self._template.module.validation_section(
                mode_notice=mode_notice,
                results="".join(result_sections),
                summary=summary,
            )
            
        except Exception as e:
            logger.error(f"Error formatting validation results: {e}")
            return f"Error formatting validation results: {str(e)}"
    
    def _display_path(self, file_path: str) -> str:
        """Get display-friendly path."""
        if not file_path:
            return "unknown"
        
        # Try to make it relative to common roots
        path = Path(file_path)
        
        # Just return the last 3 parts for brevity
        parts = path.parts
        if len(parts) > 3:
            return "/".join(parts[-3:])
        
        return str(path)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SageMakerValidationDisplayFormatter",
]
