"""
SageMaker Notebook Instance validator for Neuron compatibility.

ARCHITECTURE:
    Uses AWS SageMaker Notebook Instances with lifecycle configurations to
    validate models for Neuron targets. Runs on Neuron-enabled instances
    (ml.inf2.xlarge, ml.trn1.2xlarge) with the full Neuron SDK.
    Works from any laptop - NO Neuron SDK installation needed locally!
    
FLOW:
    1. Customer provides S3 URI to their model.tar.gz
    2. Upload compile_script.py to S3
    3. Create lifecycle configuration with embedded S3 paths
    4. Create Notebook Instance with lifecycle config attached
    5. Instance boots, lifecycle script runs torch_neuronx.trace()
    6. Poll S3 for result.json
    7. Stop and delete notebook instance
    
COST:
    ~$1.30/hour for ml.inf2.xlarge (typically 5-15 min per model = $0.20-0.50)
    
PREREQUISITES:
    - Model already in S3 as model.tar.gz (TorchScript or HuggingFace)
    - AWS credentials configured
    - Environment variables set (see create_validator_from_env)
"""

import base64
import json
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from rich.console import Console

from .template_engine import get_template_env

logger = logging.getLogger(__name__)

# Rich console for progress output (stderr so it doesn't interfere with main output)
_progress_console = Console(stderr=True)


# =============================================================================
# Types and Constants
# =============================================================================

class NeuronTarget(str, Enum):
    """Available instance types for Sagemaker Notebook Instances for Neuron compilation."""
    # 
    # Inferentia 2 (Inference)
    INF2_XLARGE = "ml.inf2.xlarge"
    INF2_8XLARGE = "ml.inf2.8xlarge"
    INF2_24XLARGE = "ml.inf2.24xlarge"
    INF2_48XLARGE = "ml.inf2.48xlarge"
    # Trainium 1 (Training & Inference)
    TRN1_2XLARGE = "ml.trn1.2xlarge"
    TRN1_32XLARGE = "ml.trn1.32xlarge"
    TRN1N_32XLARGE = "ml.trn1n.32xlarge"
    # Trainium 2 (Next-Gen Training)
    TRN2_48XLARGE = "ml.trn2.48xlarge"


class SageMakerValidationStatus(str, Enum):
    """Validation job status."""
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    IN_PROGRESS = "IN_PROGRESS"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class SageMakerValidationResult(BaseModel):
    """Result of SageMaker Notebook Instance validation."""
    
    model_config = {"frozen": True}
    
    status: SageMakerValidationStatus
    model_name: str
    target_device: str = Field(default="ml.inf2.xlarge")
    job_name: Optional[str] = None
    notebook_instance_name: Optional[str] = None
    output_location: Optional[str] = None
    error_message: Optional[str] = None
    detected_input_shape: Optional[str] = None
    detected_model_type: Optional[str] = None
    compilation_time_seconds: Optional[float] = None
    
    @property
    def is_compatible(self) -> bool:
        return self.status == SageMakerValidationStatus.COMPATIBLE


# =============================================================================
# Configuration Settings (Pydantic BaseSettings)
# =============================================================================

# Legacy target name mapping for backward compatibility
_LEGACY_TARGET_MAP = {
    "ml_inf2": "ml.inf2.xlarge",
    "ml_trn1": "ml.trn1.2xlarge",
    "ml_trn1n": "ml.trn1n.32xlarge",
    "ml_trn2": "ml.trn2.48xlarge",
}


class SageMakerValidatorSettings(BaseSettings):
    """
    Configuration for SageMaker Neuron Validator.
    
    ARCHITECTURE:
        Pydantic BaseSettings model that loads configuration from environment
        variables. Centralizes all validation and default values.
        
    ENVIRONMENT VARIABLES:
        Required:
            NEURON_VALIDATION_BUCKET: S3 bucket for compiled model output
            NEURON_VALIDATION_ROLE_ARN: IAM role ARN with SageMaker permissions
        Optional:
            AWS_REGION: AWS region (default: us-east-1)
            NEURON_TARGET_DEVICE: Target instance type (default: ml.inf2.xlarge)
    """
    
    model_config = {
        "env_prefix": "",  # No prefix - use exact env var names
        "frozen": True,
    }
    
    # Required fields (no defaults)
    bucket: str = Field(
        ...,
        min_length=3,
        max_length=63,
        description="S3 bucket for compiled model output",
        alias="NEURON_VALIDATION_BUCKET",
    )
    role_arn: str = Field(
        ...,
        min_length=20,
        pattern=r"^arn:aws:iam::\d{12}:role/.+",
        description="IAM role ARN with SageMaker permissions",
        alias="NEURON_VALIDATION_ROLE_ARN",
    )
    
    # Optional fields with defaults
    region: str = Field(
        default="us-east-1",
        min_length=5,
        description="AWS region",
        alias="AWS_REGION",
    )
    target_device: str = Field(
        default="ml.inf2.xlarge",
        description="Target Neuron instance type",
        alias="NEURON_TARGET_DEVICE",
    )
    
    @field_validator("target_device", mode="before")
    @classmethod
    def normalize_target_device(cls, value: str) -> str:
        """Normalize legacy target names to current format."""
        if value in _LEGACY_TARGET_MAP:
            logger.info(f"Mapping legacy target '{value}' to '{_LEGACY_TARGET_MAP[value]}'")
            return _LEGACY_TARGET_MAP[value]
        return value
    
    @field_validator("bucket")
    @classmethod
    def validate_bucket_name(cls, value: str) -> str:
        """Validate S3 bucket naming rules."""
        value_clean = value.replace("-", "").replace(".", "")
        if not value_clean.isalnum():
            raise ValueError("Bucket name contains invalid characters")
        return value.lower()


# =============================================================================
# SageMaker Notebook Instance Validator
# =============================================================================

class SageMakerNeuronValidator:
    """
    Validate Neuron compatibility using SageMaker Notebook Instances.
    
    ARCHITECTURE:
        Creates a temporary Notebook Instance with lifecycle configuration
        that runs torch_neuronx.trace() on Neuron-enabled instances.
        Supports modern PyTorch (2.5+) via Neuron SDK on inf2/trn1.
        Customer provides S3 URI to their pre-staged model.
        Input shapes auto-detected from config.json (optional override).
        
    RESPONSIBILITIES (Single Purpose: Neuron Validation via SageMaker):
        - S3 Operations: Upload compile scripts, extract model names
        - Lifecycle Config: Create/cleanup SageMaker lifecycle configurations
        - Notebook Instance: Create/wait/cleanup notebook instances
        - Result Polling: Poll S3 for compilation results
        
        All methods work together for one cohesive purpose: validating a model's
        Neuron compatibility using SageMaker infrastructure. The class is kept
        unified because splitting would create tight coupling between components.
        
    USAGE:
        validator = SageMakerNeuronValidator(
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            output_bucket="my-bucket"
        )
        result = validator.validate_s3_model("s3://my-bucket/model.tar.gz")
    """
    
    def __init__(
        self,
        role_arn: str,
        output_bucket: str,
        region: str = "us-east-1",
        target_device: NeuronTarget = NeuronTarget.INF2_XLARGE,
    ):
        """
        Initialize SageMaker validator.
        
        Args:
            role_arn: IAM role ARN with SageMaker permissions
            output_bucket: S3 bucket for compiled model output and scripts
            region: AWS region
            target_device: Neuron target (inf2 recommended)
        """
        try:
            import boto3
            self.sagemaker_client = boto3.client('sagemaker', region_name=region)
            self.s3_client = boto3.client('s3', region_name=region)
        except ImportError:
            raise ImportError(
                "boto3 is required for SageMaker validation. "
                "Install with: pip install 'neuron-scanner[sagemaker]'"
            )
        
        self.role_arn = role_arn
        self.output_bucket = output_bucket
        self.region = region
        self.target_device = target_device if isinstance(target_device, NeuronTarget) else NeuronTarget(target_device)
        
        # Script location in this package
        self._script_path = Path(__file__).parent / "compile_script.py"
        
        # Load Jinja2 templates for console output and lifecycle script
        self._env = get_template_env()
        self._console_template = self._env.get_template("console.j2")
        self._lifecycle_template = self._env.get_template("lifecycle_script.j2")
    
    def _render_console(self, section: str, **kwargs) -> str:
        """Render console message from template."""
        return self._console_template.render(section=section, **kwargs).strip()
    
    # -------------------------------------------------------------------------
    # Public API - Entry point for Neuron validation
    # -------------------------------------------------------------------------
    
    def validate_s3_model(
        self,
        s3_uri: str,
        model_name: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        timeout_seconds: int = 1800,
        instance_type_override: Optional[str] = None,
        retain_instance: bool = False,
    ) -> SageMakerValidationResult:
        """
        Validate a model already in S3 using SageMaker Notebook Instance.
        
        Args:
            s3_uri: S3 URI to model.tar.gz (e.g., s3://bucket/path/model.tar.gz)
            model_name: Name for logging (extracted from URI if not provided)
            input_shape: Optional input tensor shape (auto-detected if not provided)
            timeout_seconds: Max time to wait for compilation (default 30 min)
            instance_type_override: Override the default instance type
            retain_instance: Retain notebook instance after compilation (default: False)
            
        Returns:
            SageMakerValidationResult with compatibility status
        """
        start_time = time.time()
        
        # Extract model name from URI if not provided
        if model_name is None:
            model_name = self._extract_model_name(s3_uri)
        
        # Validate S3 URI format
        if not s3_uri.startswith("s3://"):
            return SageMakerValidationResult(
                status=SageMakerValidationStatus.ERROR,
                model_name=model_name,
                target_device=self.target_device.value,
                error_message=f"Invalid S3 URI: {s3_uri}. Must start with s3://",
            )
        
        # Generate unique names
        timestamp = int(time.time())
        safe_name = model_name[:20].replace('_', '-').replace('.', '-').lower()
        notebook_name = f"neuron-val-{safe_name}-{timestamp}"
        lifecycle_config_name = f"neuron-lc-{safe_name}-{timestamp}"
        output_prefix = f"neuron-validation/output/{notebook_name}"
        
        try:
            # Upload compile script to S3
            script_s3_uri = self._upload_compile_script()
            
            # Determine instance type
            if instance_type_override:
                instance_type = instance_type_override
            else:
                instance_type = self.target_device.value
            
            _progress_console.print()
            _progress_console.print(self._render_console(
                "start_validation",
                job_name=notebook_name,
                model_uri=s3_uri,
                instance_type=instance_type,
                target_device=self.target_device.value,
                input_shape=input_shape
            ))
            
            # Create lifecycle configuration
            _progress_console.print(self._render_console("lifecycle_config"))
            self._create_lifecycle_config(
                config_name=lifecycle_config_name,
                model_s3_uri=s3_uri,
                script_s3_uri=script_s3_uri,
                output_s3_uri=f"s3://{self.output_bucket}/{output_prefix}",
                input_shape=input_shape,
            )
            
            # Create notebook instance
            _progress_console.print(self._render_console("create_notebook", instance_type=instance_type))
            self._create_notebook_instance(
                notebook_name=notebook_name,
                instance_type=instance_type,
                lifecycle_config_name=lifecycle_config_name,
            )
            
            # Wait for notebook to reach InService status
            _progress_console.print(self._render_console("wait_start"))
            self._wait_for_notebook_status(notebook_name, "InService", timeout_seconds // 2)
            
            # Poll S3 for result.json
            _progress_console.print(self._render_console("wait_results"))
            result = self._wait_for_compilation_result(
                notebook_name=notebook_name,
                model_name=model_name,
                output_prefix=output_prefix,
                timeout_seconds=timeout_seconds,
            )
            
            # Add compilation time
            result_dict = {k: v for k, v in result.model_dump().items() if k != 'compilation_time_seconds'}
            result_dict['compilation_time_seconds'] = time.time() - start_time
            return SageMakerValidationResult(**result_dict)
            
        except Exception as e:
            logger.error(f"Notebook validation failed: {e}")
            return SageMakerValidationResult(
                status=SageMakerValidationStatus.ERROR,
                model_name=model_name,
                target_device=self.target_device.value,
                notebook_instance_name=notebook_name,
                error_message=str(e),
                compilation_time_seconds=time.time() - start_time,
            )
        finally:
            # Clean up resources
            if not retain_instance:
                self._cleanup_notebook_instance(notebook_name)
                self._cleanup_lifecycle_config(lifecycle_config_name)
            else:
                logger.info(f"Retaining notebook instance: {notebook_name}")
                logger.info(f"Access at: https://console.aws.amazon.com/sagemaker/home?region={self.region}#/notebook-instances/{notebook_name}")
                _progress_console.print(f"\n[yellow]⚠️  Notebook instance retained: {notebook_name}[/yellow]")
                _progress_console.print(f"[yellow]⚠️  Cost: ~$1.30/hour for ml.inf2.xlarge - remember to delete manually[/yellow]")
                _progress_console.print(f"[dim]Access: https://console.aws.amazon.com/sagemaker/home?region={self.region}#/notebook-instances/{notebook_name}[/dim]")
    
    # -------------------------------------------------------------------------
    # Internal Methods - S3 Operations
    # Handles: Upload compile scripts, extract model names from URIs
    # -------------------------------------------------------------------------
    
    def _extract_model_name(self, s3_uri: str) -> str:
        """Extract a model name from S3 URI for logging."""
        parts = s3_uri.rstrip('/').split('/')
        filename = parts[-1] if parts else "model"
        if filename.endswith('.tar.gz'):
            filename = filename[:-7]
        elif '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        return filename or "model"
    
    def _upload_compile_script(self) -> str:
        """Upload compile_script.py to S3 and return S3 URI."""
        script_key = "neuron-validation/scripts/compile_script.py"
        
        with open(self._script_path, 'r') as f:
            script_content = f.read()
        
        self.s3_client.put_object(
            Bucket=self.output_bucket,
            Key=script_key,
            Body=script_content.encode('utf-8'),
        )
        
        s3_uri = f"s3://{self.output_bucket}/{script_key}"
        logger.info(f"Uploaded compile script to: {s3_uri}")
        return s3_uri
    
    # -------------------------------------------------------------------------
    # Internal Methods - Lifecycle Configuration
    # Handles: Create/update/delete SageMaker notebook lifecycle configs
    # -------------------------------------------------------------------------
    
    def _create_lifecycle_config(
        self,
        config_name: str,
        model_s3_uri: str,
        script_s3_uri: str,
        output_s3_uri: str,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Create a lifecycle configuration for the notebook instance."""
        
        # Prepare template variables
        input_shape_str = ",".join(str(dim) for dim in input_shape) if input_shape else ""
        container_registry = f"763104351884.dkr.ecr.{self.region}.amazonaws.com"
        container_image = f"{container_registry}/pytorch-training-neuronx:2.8.0-neuronx-py311-sdk2.26.0-ubuntu22.04"
        
        # Render lifecycle script from Jinja2 template
        on_start_script = self._lifecycle_template.render(
            model_s3_uri=model_s3_uri,
            output_s3_uri=output_s3_uri,
            script_s3_uri=script_s3_uri,
            input_shape_str=input_shape_str,
            region=self.region,
            container_registry=container_registry,
            container_image=container_image,
        )
        
        # Base64 encode the script (required by SageMaker)
        encoded_script = base64.b64encode(on_start_script.encode('utf-8')).decode('utf-8')
        
        # Create lifecycle configuration
        try:
            self.sagemaker_client.create_notebook_instance_lifecycle_config(
                NotebookInstanceLifecycleConfigName=config_name,
                OnStart=[{'Content': encoded_script}],
            )
            logger.info(f"Created lifecycle config: {config_name}")
        except self.sagemaker_client.exceptions.ClientError as e:
            if 'ResourceInUse' in str(e):
                logger.warning(f"Lifecycle config {config_name} already exists, updating...")
                self.sagemaker_client.update_notebook_instance_lifecycle_config(
                    NotebookInstanceLifecycleConfigName=config_name,
                    OnStart=[{'Content': encoded_script}],
                )
            else:
                raise
    
    def _cleanup_lifecycle_config(self, config_name: str) -> None:
        """Delete lifecycle configuration."""
        try:
            self.sagemaker_client.delete_notebook_instance_lifecycle_config(
                NotebookInstanceLifecycleConfigName=config_name
            )
            logger.info(f"Deleted lifecycle config: {config_name}")
        except Exception as e:
            logger.warning(f"Failed to delete lifecycle config {config_name}: {e}")
    
    # -------------------------------------------------------------------------
    # Internal Methods - Notebook Instance
    # Handles: Create/wait/stop/delete SageMaker notebook instances
    # -------------------------------------------------------------------------
    
    def _create_notebook_instance(
        self,
        notebook_name: str,
        instance_type: str,
        lifecycle_config_name: str,
    ) -> None:
        """Create a SageMaker Notebook Instance."""
        
        self.sagemaker_client.create_notebook_instance(
            NotebookInstanceName=notebook_name,
            InstanceType=instance_type,
            RoleArn=self.role_arn,
            VolumeSizeInGB=50,
            LifecycleConfigName=lifecycle_config_name,
            DirectInternetAccess='Enabled',
            RootAccess='Enabled',
        )
        logger.info(f"Created notebook instance: {notebook_name} ({instance_type})")
    
    def _wait_for_notebook_status(
        self,
        notebook_name: str,
        target_status: str,
        timeout_seconds: int = 600,
        poll_interval: int = 30,
    ) -> None:
        """Wait for notebook instance to reach target status."""
        
        start_time = time.time()
        poll_count = 0
        
        while time.time() - start_time < timeout_seconds:
            response = self.sagemaker_client.describe_notebook_instance(
                NotebookInstanceName=notebook_name
            )
            current_status = response['NotebookInstanceStatus']
            elapsed = int(time.time() - start_time)
            
            if current_status == target_status:
                _progress_console.print(self._render_console("notebook_ready", elapsed=elapsed))
                return
            
            if current_status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                raise RuntimeError(f"Notebook instance failed: {failure_reason}")
            
            poll_count += 1
            # Show progress with spinner-like feedback
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'][poll_count % 10]
            _progress_console.print(self._render_console("status_spinner", spinner=spinner, status=current_status, elapsed=elapsed))
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Notebook {notebook_name} did not reach {target_status} within {timeout_seconds}s")
    
    def _cleanup_notebook_instance(self, notebook_name: str) -> None:
        """Stop and delete notebook instance."""
        try:
            # Check current status
            response = self.sagemaker_client.describe_notebook_instance(
                NotebookInstanceName=notebook_name
            )
            status = response['NotebookInstanceStatus']
            
            # Stop if running
            if status in ['InService', 'Pending']:
                logger.info(f"Stopping notebook instance: {notebook_name}")
                self.sagemaker_client.stop_notebook_instance(
                    NotebookInstanceName=notebook_name
                )
                self._wait_for_notebook_status(notebook_name, 'Stopped', timeout_seconds=300)
            
            # Delete
            logger.info(f"Deleting notebook instance: {notebook_name}")
            self.sagemaker_client.delete_notebook_instance(
                NotebookInstanceName=notebook_name
            )
            logger.info(f"Deleted notebook instance: {notebook_name}")
            
        except self.sagemaker_client.exceptions.ClientError as e:
            if 'ResourceNotFound' in str(e):
                logger.info(f"Notebook instance {notebook_name} not found (already deleted)")
            else:
                logger.warning(f"Failed to cleanup notebook instance {notebook_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to cleanup notebook instance {notebook_name}: {e}")
    
    # -------------------------------------------------------------------------
    # Internal Methods - Result Polling
    # Handles: Poll S3 for compilation results, parse result JSON
    # -------------------------------------------------------------------------
    
    def _wait_for_compilation_result(
        self,
        notebook_name: str,
        model_name: str,
        output_prefix: str,
        timeout_seconds: int = 1800,
        poll_interval: int = 30,
    ) -> SageMakerValidationResult:
        """Poll S3 for compilation result."""
        
        result_key = f"{output_prefix}/result.json"
        start_time = time.time()
        
        logger.info(f"Polling for result at: s3://{self.output_bucket}/{result_key}")
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.s3_client.get_object(
                    Bucket=self.output_bucket,
                    Key=result_key,
                )
                result_data = json.loads(response['Body'].read().decode('utf-8'))
                
                status_val = result_data.get('status')
                _progress_console.print(self._render_console("compilation_complete", status=status_val))
                
                # Parse result
                compile_status = result_data.get('status', 'ERROR')
                
                if compile_status == 'COMPATIBLE':
                    return SageMakerValidationResult(
                        status=SageMakerValidationStatus.COMPATIBLE,
                        model_name=model_name,
                        target_device=self.target_device.value,
                        notebook_instance_name=notebook_name,
                        output_location=f"s3://{self.output_bucket}/{output_prefix}",
                        detected_input_shape=str(result_data.get('input_shape')),
                        detected_model_type=result_data.get('model_type'),
                    )
                else:
                    return SageMakerValidationResult(
                        status=SageMakerValidationStatus.INCOMPATIBLE,
                        model_name=model_name,
                        target_device=self.target_device.value,
                        notebook_instance_name=notebook_name,
                        error_message=result_data.get('error', result_data.get('message', 'Unknown error')),
                        detected_input_shape=str(result_data.get('input_shape')),
                        detected_model_type=result_data.get('model_type'),
                    )
                    
            except self.s3_client.exceptions.NoSuchKey:
                # Result not ready yet
                elapsed = int(time.time() - start_time)
                poll_num = elapsed // poll_interval + 1
                spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'][poll_num % 10]
                _progress_console.print(self._render_console("compilation_spinner", spinner=spinner, elapsed=elapsed))
            except Exception as e:
                logger.warning(f"Error checking for result: {e}")
            
            time.sleep(poll_interval)
        
        # Timeout
        return SageMakerValidationResult(
            status=SageMakerValidationStatus.ERROR,
            model_name=model_name,
            target_device=self.target_device.value,
            notebook_instance_name=notebook_name,
            error_message=f"Compilation timed out after {timeout_seconds} seconds. Check CloudWatch logs for notebook instance.",
        )


# =============================================================================
# CLI Integration Helper
# =============================================================================

def create_validator_from_env() -> Optional[SageMakerNeuronValidator]:
    """
    Create validator from environment variables using Pydantic BaseSettings.
    
    Required env vars:
        NEURON_VALIDATION_BUCKET: S3 bucket for compiled model output
        NEURON_VALIDATION_ROLE_ARN: IAM role ARN with SageMaker permissions
        
    Optional env vars:
        AWS_REGION: AWS region (default: us-east-1)
        NEURON_TARGET_DEVICE: Target device (default: ml.inf2.xlarge)
        
    Returns:
        SageMakerNeuronValidator instance or None if configuration is missing/invalid
    """
    try:
        settings = SageMakerValidatorSettings()
        
        return SageMakerNeuronValidator(
            role_arn=settings.role_arn,
            output_bucket=settings.bucket,
            region=settings.region,
            target_device=NeuronTarget(settings.target_device),
        )
    except Exception as e:
        # Pydantic will raise ValidationError for missing/invalid fields
        logger.warning(f"SageMaker validation not configured: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NeuronTarget",
    "SageMakerValidationStatus",
    "SageMakerValidationResult",
    "SageMakerValidatorSettings",
    "SageMakerNeuronValidator",
    "create_validator_from_env",
]
