"""
Model validation service - orchestration layer.

ORGANIZATION:
    1. AsyncJobTracker - Manages async SageMaker validation jobs
    2. ModelValidationService - Main service for validating model support

ARCHITECTURE:
    Service layer - orchestrates classification and display formatting.
    
    Three modes:
    1. Name-based lookup: validate_model("meta-llama/Llama-2-7b-hf")
    2. S3-based SageMaker Notebook Instance (sync): validate_s3_uri("s3://bucket/model.tar.gz")
    3. S3-based SageMaker Notebook Instance (async): start_async_validation() + get_job_status()
"""

import asyncio
import logging
import time
import uuid
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field

from neuron_scanner.model_validation.display import (
    SageMakerValidationDisplayFormatter,
)
from neuron_scanner.model_validation.template_engine import get_template_env
from neuron_scanner.model_validation.types import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Async Job Tracker
# =============================================================================

class JobStatus(str, Enum):
    """Status of an async validation job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =============================================================================
# Pydantic Models for Job Tracking
# =============================================================================

class ValidationJob(BaseModel):
    """
    Represents an async validation job.
    
    ARCHITECTURE:
        Pydantic model for job state tracking.
        Mutable during job lifecycle (not frozen).
        Uses computed_field for derived properties.
    """
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Required fields
    job_id: str = Field(..., min_length=1, description="Unique job identifier")
    s3_uri: str = Field(..., min_length=5, description="S3 URI to model.tar.gz")
    
    # Status tracking
    status: JobStatus = Field(default=JobStatus.PENDING, description="Current job status")
    created_at: float = Field(default_factory=time.time, description="Job creation timestamp")
    started_at: Optional[float] = Field(None, description="Job start timestamp")
    completed_at: Optional[float] = Field(None, description="Job completion timestamp")
    
    # Result data
    result: Optional[ValidationResult] = Field(None, description="Validation result when completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Configuration
    instance_type: Optional[str] = Field(None, description="SageMaker instance type")
    input_shape: Optional[Tuple[int, ...]] = Field(None, description="Input tensor shape")
    
    @computed_field
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since job creation."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return time.time() - self.created_at
    
    @computed_field
    @property
    def elapsed_formatted(self) -> str:
        """Get elapsed time as formatted string."""
        elapsed = int(self.elapsed_seconds)
        if elapsed < 60:
            return f"{elapsed}s"
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes}m {seconds}s"


# =============================================================================
# Pydantic Response Models (for API/MCP responses)
# =============================================================================

class JobResultSummary(BaseModel):
    """Summary of validation result for API responses."""
    
    model_config = {"frozen": True}
    
    compatible: bool = Field(..., description="Whether the model is Neuron-compatible")
    status: str = Field(..., description="Validation status value")
    message: str = Field(..., description="Human-readable result message")
    model_name: str = Field(..., description="Name of the validated model")


class JobStatusResponse(BaseModel):
    """
    Response model for job status queries.
    
    ARCHITECTURE:
        Immutable Pydantic model returned by get_status().
        Replaces raw Dict[str, Any] for type safety.
    """
    
    model_config = {"frozen": True}
    
    # Always present
    job_id: str = Field(..., description="Unique job identifier")
    s3_uri: str = Field(..., description="S3 URI being validated")
    status: str = Field(..., description="Current job status")
    elapsed: str = Field(..., description="Formatted elapsed time")
    elapsed_seconds: int = Field(..., ge=0, description="Elapsed time in seconds")
    
    # Conditional fields
    message: Optional[str] = Field(None, description="Status message for pending/running jobs")
    error: Optional[str] = Field(None, description="Error message for failed jobs")
    result: Optional[JobResultSummary] = Field(None, description="Result summary for completed jobs")


class JobListItem(BaseModel):
    """Summary item for job listing."""
    
    model_config = {"frozen": True}
    
    job_id: str = Field(..., description="Unique job identifier")
    s3_uri: str = Field(..., description="S3 URI being validated")
    status: str = Field(..., description="Current job status")
    elapsed: str = Field(..., description="Formatted elapsed time")


class AsyncJobTracker:
    """
    Tracks async SageMaker validation jobs.
    
    ARCHITECTURE:
        In-memory job storage with asyncio task management.
        Jobs are tracked by job_id and can be queried for status.
    
    USAGE:
        tracker = AsyncJobTracker()
        job_id = tracker.start_job(s3_uri, run_validation_func)
        status = tracker.get_status(job_id)
    """
    
    def __init__(self):
        self._jobs: Dict[str, ValidationJob] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
    
    def start_job(
        self,
        s3_uri: str,
        validation_func: callable,
        input_shape: Optional[Tuple[int, ...]] = None,
        instance_type: Optional[str] = None,
    ) -> str:
        """
        Start an async validation job.
        
        Args:
            s3_uri: S3 URI to model.tar.gz
            validation_func: Sync function to run (called in thread pool)
            input_shape: Optional input shape override
            instance_type: Optional instance type override
        
        Returns:
            job_id for tracking
        """
        job_id = f"neuron-val-{uuid.uuid4().hex[:8]}"
        
        job = ValidationJob(
            job_id=job_id,
            s3_uri=s3_uri,
            status=JobStatus.PENDING,
            input_shape=input_shape,
            instance_type=instance_type,
        )
        self._jobs[job_id] = job
        
        # Start background task
        task = asyncio.create_task(
            self._run_validation(job_id, validation_func)
        )
        self._tasks[job_id] = task
        
        logger.info(f"Started async validation job: {job_id}")
        return job_id
    
    async def _run_validation(self, job_id: str, validation_func: callable) -> None:
        """Run validation in background and update job status."""
        job = self._jobs.get(job_id)
        if not job:
            return
        
        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            
            # Run sync validation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Default thread pool
                validation_func,
                job.s3_uri,
                job.input_shape,
                None,  # model_name (extracted from URI)
                job.instance_type,
            )
            
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            logger.info(f"Job {job_id} completed: {result.status.value}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            logger.error(f"Job {job_id} failed: {e}")
    
    def get_job(self, job_id: str) -> Optional[ValidationJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def get_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Get job status as Pydantic model.
        
        Args:
            job_id: Unique job identifier
        
        Returns:
            JobStatusResponse with status, elapsed time, and result if completed.
            None if job not found.
        """
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        # Build optional fields based on status
        message: Optional[str] = None
        error: Optional[str] = None
        result_summary: Optional[JobResultSummary] = None
        
        if job.status == JobStatus.COMPLETED and job.result:
            result_summary = JobResultSummary(
                compatible=job.result.status == ValidationStatus.COMPATIBLE,
                status=job.result.status.value,
                message=job.result.message,
                model_name=job.result.model_name,
            )
        elif job.status == JobStatus.FAILED:
            error = job.error
        elif job.status == JobStatus.RUNNING:
            message = (
                f"Validation in progress ({job.elapsed_formatted} elapsed). "
                "SageMaker Notebook Instance validation typically takes 5-15 minutes."
            )
        elif job.status == JobStatus.PENDING:
            message = "Job queued, starting shortly..."
        
        return JobStatusResponse(
            job_id=job.job_id,
            s3_uri=job.s3_uri,
            status=job.status.value,
            elapsed=job.elapsed_formatted,
            elapsed_seconds=int(job.elapsed_seconds),
            message=message,
            error=error,
            result=result_summary,
        )
    
    def list_jobs(self, limit: int = 10) -> List[JobListItem]:
        """
        List recent validation jobs.
        
        Args:
            limit: Maximum number of jobs to return (default 10)
        
        Returns:
            List of JobListItem models, sorted by creation time (newest first)
        """
        jobs = sorted(
            self._jobs.values(),
            key=lambda job: job.created_at,
            reverse=True
        )[:limit]
        
        return [
            JobListItem(
                job_id=job.job_id,
                s3_uri=job.s3_uri,
                status=job.status.value,
                elapsed=job.elapsed_formatted,
            )
            for job in jobs
        ]
    
    def cleanup_old_jobs(self, max_age_seconds: int = 3600) -> int:
        """Remove completed jobs older than max_age_seconds."""
        now = time.time()
        to_remove = [
            job_id for job_id, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            and (now - job.created_at) > max_age_seconds
        ]
        
        for job_id in to_remove:
            del self._jobs[job_id]
            if job_id in self._tasks:
                del self._tasks[job_id]
        
        return len(to_remove)


# =============================================================================
# Singleton Job Tracker (thread-safe via lru_cache)
# =============================================================================

@lru_cache(maxsize=1)
def get_job_tracker() -> AsyncJobTracker:
    """
    Get singleton job tracker instance.
    
    ARCHITECTURE:
        Uses lru_cache for thread-safe singleton pattern.
        No global mutable state - instance created on first call and cached.
        
    Returns:
        AsyncJobTracker singleton instance
    """
    return AsyncJobTracker()


class ModelValidationService:
    """Service for validating AWS Neuron support for ML models."""

    def __init__(self):
        """Initialize service with formatters and templates."""
        self._sagemaker_formatter = SageMakerValidationDisplayFormatter()
        
        # Load Jinja2 templates for service messages
        self._env = get_template_env()
        self._messages_template = self._env.get_template("service_messages.j2")

    # =========================================================================
    # Mode 1: Name-based lookup (DEPRECATED)
    # =========================================================================

    def validate_model(self, model_name: str) -> str:
        """
        Validate AWS Neuron support for a specific model by name.
        
        Args:
            model_name: HuggingFace model ID or model name
        
        Returns:
            Formatted message pointing to documentation
        """
        return self._messages_template.module.deprecated_model_lookup(
            model_name=model_name
        ).strip()

    # =========================================================================
    # Mode 2: S3 URI SageMaker validation (simplified)
    # =========================================================================
    
    def validate_s3_uri(
        self,
        s3_uri: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        model_name: Optional[str] = None,
        instance_type: Optional[str] = None,
    ) -> str:
        """
        Validate a model in S3 using SageMaker Notebook Instance.
        
        The model must already be staged in S3 as a .tar.gz file containing
        a TorchScript model (model.pt) or HuggingFace model files.
        
        Input shapes are auto-detected from config.json in the model tarball.
        You can override with input_shape if needed.
        
        Args:
            s3_uri: S3 URI to model.tar.gz (e.g., s3://bucket/path/model.tar.gz)
            input_shape: Optional input tensor shape (auto-detected if not provided)
            model_name: Display name for the model (extracted from URI if not provided)
            instance_type: Optional SageMaker instance type override
        
        Returns:
            Formatted validation report
        """
        try:
            result = self._run_s3_validation(s3_uri, input_shape, model_name, instance_type)
            return self._sagemaker_formatter.format([result])
        except Exception as e:
            logger.error(f"Error in SageMaker validation: {e}")
            return self._messages_template.module.validation_error(
                error_message=str(e)
            ).strip()

    def _get_configured_validator(self):
        """
        Get configured SageMaker validator or None if not available.
        
        Returns:
            SageMakerNeuronValidator instance or None if boto3 not installed
            or environment not configured.
        """
        try:
            from neuron_scanner.model_validation.sagemaker_validator import (
                create_validator_from_env,
            )
            return create_validator_from_env()
        except ImportError:
            logger.warning("boto3 not installed for SageMaker validation")
            return None

    def _create_error_result(
        self,
        s3_uri: str,
        model_name: Optional[str],
        error_category: str,
        message: str,
    ) -> ValidationResult:
        """Create a ValidationResult for error cases."""
        return ValidationResult(
            status=ValidationStatus.ERROR,
            model_name=model_name or f"({error_category})",
            file_path=s3_uri,
            message=message,
            error_category=error_category,
        )

    def _run_s3_validation(
        self,
        s3_uri: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        model_name: Optional[str] = None,
        instance_type: Optional[str] = None,
    ) -> ValidationResult:
        """
        Run Neuron validation via SageMaker Notebook Instance for an S3 model.
        
        Works from any laptop! Requires AWS credentials and environment variables:
          - NEURON_VALIDATION_BUCKET: S3 bucket for compiled model output
          - NEURON_VALIDATION_ROLE_ARN: IAM role ARN with SageMaker permissions
        
        Args:
            s3_uri: S3 URI to model.tar.gz
            input_shape: Optional input tensor shape (auto-detected from config.json)
            model_name: Display name for the model
            instance_type: Optional SageMaker instance type override
            
        Returns:
            ValidationResult with compatibility status
        """
        try:
            # Get configured validator (handles ImportError internally)
            validator = self._get_configured_validator()
            
            if validator is None:
                logger.error(
                    "SageMaker validation not configured. "
                    "Set NEURON_VALIDATION_BUCKET and NEURON_VALIDATION_ROLE_ARN."
                )
                return self._create_error_result(
                    s3_uri=s3_uri,
                    model_name=model_name,
                    error_category="MISSING_CONFIG",
                    message=(
                        "SageMaker validation requires environment variables: "
                        "NEURON_VALIDATION_BUCKET and NEURON_VALIDATION_ROLE_ARN. "
                        "See README for prerequisites."
                    ),
                )
            
            # Log validation parameters
            logger.info("Validating S3 model via SageMaker Notebook Instance...")
            logger.info(f"  URI: {s3_uri}")
            if input_shape:
                logger.info(f"  Input shape (override): {input_shape}")
            else:
                logger.info("  Input shape: auto-detect from config.json")
            if instance_type:
                logger.info(f"  Instance type (override): {instance_type}")
            logger.info("This may take 5-15 minutes on first run...")
            
            # Call SageMaker validator
            result = validator.validate_s3_model(
                s3_uri=s3_uri,
                model_name=model_name,
                input_shape=input_shape,
                instance_type_override=instance_type,
            )
            
            # Convert SageMaker result to our ValidationResult
            if result.is_compatible:
                status = ValidationStatus.COMPATIBLE
                message = f"Model compiled successfully for {result.target_device}"
                logger.info("✅ Model is COMPATIBLE with Neuron!")
            else:
                status = ValidationStatus.INCOMPATIBLE
                message = result.error_message or "Compilation failed"
                logger.warning(f"❌ Model is INCOMPATIBLE: {message}")
            
            return ValidationResult(
                status=status,
                model_name=result.model_name,
                file_path=s3_uri,
                message=message,
                error_details=result.error_message,
            )
            
        except Exception as e:
            logger.error(f"SageMaker validation failed: {e}")
            return self._create_error_result(
                s3_uri=s3_uri,
                model_name=model_name,
                error_category="VALIDATION_ERROR",
                message=str(e),
            )

    # =========================================================================
    # Mode 3: Async S3 validation (for MCP - non-blocking)
    # =========================================================================

    def start_async_validation(
        self,
        s3_uri: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        instance_type: Optional[str] = None,
    ) -> str:
        """
        Start an async SageMaker validation job.
        
        Returns immediately with a job_id. Use get_job_status() to check progress.
        Designed for MCP tools where blocking for 10-15 minutes is not acceptable.
        
        Args:
            s3_uri: S3 URI to model.tar.gz
            input_shape: Optional input shape override
            instance_type: Optional instance type override
        
        Returns:
            job_id for tracking
        """
        tracker = get_job_tracker()
        job_id = tracker.start_job(
            s3_uri=s3_uri,
            validation_func=self._run_s3_validation,
            input_shape=input_shape,
            instance_type=instance_type,
        )
        return job_id

    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Get status of an async validation job.
        
        Args:
            job_id: Job ID returned by start_async_validation()
        
        Returns:
            JobStatusResponse with status, elapsed time, and result (if completed).
            None if job_id not found.
        """
        tracker = get_job_tracker()
        return tracker.get_status(job_id)

    def get_job_result_formatted(self, job_id: str) -> str:
        """
        Get formatted result for a completed job.
        
        Args:
            job_id: Job ID returned by start_async_validation()
        
        Returns:
            Formatted markdown result or status message
        """
        tracker = get_job_tracker()
        job = tracker.get_job(job_id)
        
        if not job:
            return self._messages_template.module.job_not_found(job_id=job_id).strip()
        
        if job.status == JobStatus.PENDING:
            return self._messages_template.module.job_pending(
                job_id=job_id,
                s3_uri=job.s3_uri,
            ).strip()
        
        if job.status == JobStatus.RUNNING:
            return self._messages_template.module.job_running(
                job_id=job_id,
                s3_uri=job.s3_uri,
                elapsed=job.elapsed_formatted,
            ).strip()
        
        if job.status == JobStatus.FAILED:
            return self._messages_template.module.job_failed(
                job_id=job_id,
                s3_uri=job.s3_uri,
                error=job.error,
                elapsed=job.elapsed_formatted,
            ).strip()
        
        if job.status == JobStatus.COMPLETED and job.result:
            return self._sagemaker_formatter.format([job.result])
        
        return self._messages_template.module.job_unknown_status(job_id=job_id).strip()

    def list_jobs(self, limit: int = 10) -> List[JobListItem]:
        """
        List recent validation jobs.
        
        Args:
            limit: Maximum number of jobs to return (default 10)
        
        Returns:
            List of JobListItem models, sorted by creation time (newest first)
        """
        tracker = get_job_tracker()
        return tracker.list_jobs(limit)
