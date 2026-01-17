"""
MCP Server for Neuron Scanner - enables integration with Claude, Cursor, and other MCP-compatible tools.
"""
import os
import json
import sys
import logging
import time
from typing import Optional

# Ensure project root is in path for direct execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# MCP imports
try:
    from mcp.fastmcp import FastMCP
except Exception:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        FastMCP = None

from neuron_scanner.model_validation.service import ModelValidationService
from neuron_scanner.helpers.cli_utilities import (
    LoggingConfigurator,
    ServerRunner,
)

# Configure logging
logger = logging.getLogger(__name__)

# Store start time for health check
sys._neuron_scanner_start_time = time.time()

# =================== Logging Configuration =================== #
# Suppress noisy logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# =================== MCP Server =================== #

def create_mcp_server():
    """Create and configure the MCP server."""
    if FastMCP is None:
        raise ImportError("MCP package not installed. Install with: pip install mcp")

    # Create FastMCP server
    mcp = FastMCP("neuron-scanner")

    # Create services once (reused across all tool calls)
    model_service = ModelValidationService()

    @mcp.tool()
    async def validate_model(
        s3_uri: str,
        instance_type: Optional[str] = None
    ) -> str:
        """
        Start an async SageMaker Neuron validation job (returns immediately).
        
        This tool starts a real Neuron compilation test on AWS infrastructure.
        The job runs in the background and typically takes 5-15 minutes.
        Use check_validation_status() to poll for results.
        
        PREREQUISITES:
        - Model must be uploaded to S3 as .tar.gz (use prepare_model.py)
        - AWS credentials configured
        - Environment variables set:
          - NEURON_VALIDATION_BUCKET: Your S3 bucket for output
          - NEURON_VALIDATION_ROLE_ARN: IAM role with SageMaker permissions
        
        Args:
            s3_uri: S3 URI to model.tar.gz 
                   (e.g., "s3://my-bucket/models/bert-base.tar.gz")
            instance_type: Optional instance type override
                          (e.g., "ml.inf2.xlarge", "ml.trn1.2xlarge")
        
        Returns:
            Job ID and instructions for checking status.
            Example: "Job started: neuron-val-abc123. Check with check_validation_status('neuron-val-abc123')"
        
        Workflow:
            1. Creates a SageMaker Notebook Instance (ml.inf2.xlarge)
            2. Downloads your model from S3
            3. Runs torch_neuronx.trace() to compile for Neuron
            4. Uploads results to S3
            5. Cleans up the notebook instance
        
        Example:
            validate_model("s3://my-bucket/models/my-model.tar.gz")
        """
        try:
            job_id = model_service.start_async_validation(
                s3_uri=s3_uri,
                instance_type=instance_type,
            )
            return (
                f"✅ **Validation Job Started**\n\n"
                f"**Job ID:** `{job_id}`\n"
                f"**S3 URI:** `{s3_uri}`\n"
                f"**Instance:** {instance_type or 'ml.inf2.xlarge (default)'}\n\n"
                f"⏳ This typically takes 5-15 minutes.\n\n"
                f"**Check status with:**\n"
                f"```\n"
                f"check_validation_status(\"{job_id}\")\n"
                f"```\n\n"
                f"The job will:\n"
                f"1. Create a SageMaker Notebook Instance (~2-3 min)\n"
                f"2. Run Neuron compilation (~2-5 min)\n"
                f"3. Upload results and cleanup (~1 min)"
            )
        except Exception as e:
            return f"❌ Failed to start validation: {str(e)}"

    @mcp.tool()
    async def check_validation_status(
        job_id: str
    ) -> str:
        """
        Check the status of a SageMaker validation job.
        
        Use this to poll for results after calling start_sagemaker_validation().
        
        Args:
            job_id: Job ID returned by start_sagemaker_validation()
                   (e.g., "neuron-val-abc123")
        
        Returns:
            Current status with details:
            - PENDING: Job queued, starting shortly
            - RUNNING: Compilation in progress (shows elapsed time)
            - COMPLETED: Result with COMPATIBLE/INCOMPATIBLE status
            - FAILED: Error details (includes specific compiler error message)
        
        Example:
            check_validation_status("neuron-val-abc123")
        """
        result = model_service.get_job_result_formatted(job_id)
        if result:
            return result
        return f"❌ Job not found: `{job_id}`\n\nRun `list_validation_jobs()` to see active jobs."

    @mcp.tool()
    async def list_validation_jobs(limit: int = 10) -> str:
        """
        List recent SageMaker validation jobs.
        
        Shows the last 10 validation jobs with their status.
        Useful for finding job IDs to check with check_validation_status().
        
        Args:
            limit: Maximum number of jobs to list (default: 10)
        
        Returns:
            List of recent jobs with ID, status, and elapsed time.
        """
        jobs = model_service.list_jobs(limit=limit)
        
        if not jobs:
            return (
                "No validation jobs found.\n\n"
                "Start one with:\n"
                "```\n"
                "start_sagemaker_validation(\"s3://bucket/model.tar.gz\")\n"
                "```"
            )
        
        lines = ["## Recent Validation Jobs\n"]
        for job in jobs:
            status_emoji = {
                "PENDING": "⏳",
                "RUNNING": "🔄",
                "COMPLETED": "✅",
                "FAILED": "❌",
            }.get(job["status"], "❓")
            
            lines.append(
                f"- {status_emoji} **{job['job_id']}** - {job['status']} ({job['elapsed']})\n"
                f"  `{job['s3_uri']}`"
            )
        
        lines.append("\n\nCheck a job with: `check_validation_status(\"job-id\")`")
        return "\n".join(lines)

    @mcp.tool()
    async def get_server_health() -> str:
        """
        Get MCP server health status and metadata.
        
        Use this to verify connection and server version.
        
        Returns:
            JSON string with status, version, uptime, and project root.
        """
        import time
        import sys
        
        # Calculate uptime (approximate, based on import time)
        # In a real long-running process, we'd store start time in a global variable
        uptime_seconds = time.time() - getattr(sys, "_neuron_scanner_start_time", time.time())
        if uptime_seconds < 0: uptime_seconds = 0
        
        health = {
            "status": "OK",
            "version": "0.1.0",
            "uptime_seconds": int(uptime_seconds),
            "project_root": project_root,
            "python_version": sys.version.split()[0],
        }
        return json.dumps(health, indent=2)

    return mcp


def main():
    """Run the MCP server."""
    logging_configurator = LoggingConfigurator()
    logging_configurator.configure()

    server_runner = ServerRunner(FastMCP, create_mcp_server)
    server_runner.run()


if __name__ == "__main__":
    main()
