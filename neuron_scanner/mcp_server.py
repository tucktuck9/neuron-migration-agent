"""
MCP Server for Neuron Scanner - enables integration with Claude, Cursor, and other MCP-compatible tools.
"""
import os
import json
import asyncio
import sys
import logging
import time
from typing import Any, Sequence, Optional
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parent.parent)
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

from neuron_scanner.application.scanner import NeuronCodeScanner
from neuron_scanner.application.service import AnalyzeCodebaseService
from neuron_scanner.kubernetes.service import KubernetesValidatorService
from neuron_scanner.recommendations.service import ModelValidationService
from neuron_scanner.helpers.cli_utilities import (
    LoggingConfigurator,
    ArgumentParser,
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
    from neuron_scanner.core.model_manager import MLModelManager
    
    scanner = NeuronCodeScanner
    model_manager = MLModelManager()  # Shared ML infrastructure
    
    analyze_service = AnalyzeCodebaseService(scanner)
    k8s_service = KubernetesValidatorService(model_manager)  # ML-based workload classification
    model_service = ModelValidationService()

    @mcp.tool()
    async def analyze_codebase(
        repo_path: str,
        analyze_operators: bool = False,
        discover_patterns: bool = False
    ) -> str:
        """
        Analyze repository code for Neuron migration compatibility.

        Scans Python source code, Dockerfiles, and Kubernetes manifests to assess
        migration compatibility. Identifies CUDA patterns, resource requirements, and
        provides actionable code context for planning migration strategies.

        Args:
            repo_path: Path to the repository to analyze
            analyze_operators: Whether to run torch_neuronx operator analysis (requires torch_neuronx installed)
            discover_patterns: Whether to use ML to discover custom CUDA patterns (requires Phi model, optional)

        Returns:
            Comprehensive analysis report with the following sections:

            ALWAYS SHOWN:
            - Summary: Readiness score (0-100), effort estimate, critical blockers
            - Key Findings: CUDA patterns count, torch.compile usage, custom kernels
            - Top Recommendations: Prioritized actions with time estimates
            - Action Details: Affected files, before/after examples, guide section links
            - Reference Documentation: Relevant AWS Neuron docs (deduplicated URLs)

            CONDITIONALLY SHOWN (only when findings exist):
            - 🔎 Code Context: Line-numbered code windows around CUDA patterns
              → Appears when CUDA device placement detected (.to('cuda'), .cuda(), torch.cuda.*)
            - 🐳 Container Context: Dockerfile excerpts around FROM statements
              → Appears when Dockerfiles found in repository
            - ☸️ Kubernetes Context: YAML snippets around resource definitions
              → Appears when manifests contain GPU resources (nvidia.com/gpu, resources:)
            - 📦 Dependency Context: requirements.txt/pyproject.toml excerpts
              → Appears when CUDA-specific libraries detected (cupy, pycuda, triton)
            - 🤖 ML-Discovered Patterns: Custom CUDA patterns found via Phi model (only with discover_patterns=True)
              → Appears when discover_patterns enabled and Phi identifies project-specific patterns

            INTERPRETATION:
            - Readiness Score: 0-30 = major work, 30-70 = moderate, 70-100 = nearly ready
            - Missing sections = no issues found in that category (positive signal)
            - Code windows provide enough context to write migration code directly

        Example:
            "Analyze ~/projects/ml-training for Neuron migration"
        """
        return await analyze_service.analyze_codebase(
            repo_path=repo_path,
            analyze_operators=analyze_operators,
            discover_patterns=discover_patterns,
        )

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
    async def validate_kubernetes_manifests(
        manifest_path: str,
        instance_type: Optional[str] = None,
        strict_mode: bool = False,
        k8s_version: Optional[str] = None
    ) -> str:
        """
        Validate Kubernetes manifests for Neuron deployment compatibility.

        Analyzes Kubernetes manifests (YAML/JSON) for Trainium & Inferentia deployment readiness.
        Validates resource declarations, node affinity, and provides actionable fixes for migration
        from GPU to Neuron workloads. Perfect for AI coding assistants to validate manifests
        during development.

        IMPORTANT - INSTANCE TYPE RECOMMENDATION:
            Specifying instance_type is STRONGLY RECOMMENDED for accurate fix recommendations.
            Without it, instance types are inferred via ML classification which may be inaccurate.
            
            Valid Neuron instance types:
              Inferentia: inf2.xlarge, inf2.8xlarge, inf2.24xlarge, inf2.48xlarge
              Trainium:   trn1.2xlarge, trn1.32xlarge, trn1n.32xlarge

        Args:
            manifest_path: Path to Kubernetes manifests (single file, directory, or glob pattern)
                         Examples: "k8s/", "deployment.yaml", "manifests/*.yaml"
            instance_type: Target Neuron instance type for fix recommendations (STRONGLY RECOMMENDED).
                          Ensures accurate nodeSelector and scheduling recommendations.
                          Examples: "inf2.xlarge", "inf2.24xlarge", "trn1.32xlarge"
                          If not specified, instance type is inferred via ML which may be inaccurate.
            strict_mode: Enable strict Kubernetes schema validation (slower but more thorough)
            k8s_version: Specific Kubernetes version to validate against (e.g., "1.28", "1.29")
                        If not specified, uses the default version from kubernetes-validate package

        Returns:
            Comprehensive validation report with the following sections:

            ALWAYS INCLUDED:
            - 📊 Summary: Manifest count, Neuron readiness status, resource validity
            - 🔍 Detailed Issues: Categorized by severity (Critical, High, Medium, Low)
            - 🔧 Quick Fixes: Actionable commands to resolve critical issues

            CONDITIONALLY INCLUDED:
            - 🚨 Critical Issues: Deployment-blocking problems (invalid resources, missing affinity)
            - ⚠️ High Priority: Performance/cost impacting issues (GPU resources detected)
            - 🔄 Migration Opportunities: GPU workloads that could benefit from Neuron
            - 📚 Reference Documentation: Relevant AWS Neuron docs

            INTERPRETATION:
            - ✅ = Ready for Neuron deployment
            - ❌ = Requires fixes before deployment
            - 🎯 = Migration opportunity identified
            - Issues include file paths, line numbers, and specific fix suggestions

            VALIDATION FEATURES:
            ✓ Neuron Resource Validation: aws.amazon.com/neuron, aws.amazon.com/neuroncore
            ✓ Node Affinity Checks: Trainium/Inferentia instance targeting
            ✓ Schema Compliance: Kubernetes API version compatibility
            ✓ Migration Detection: Identifies GPU workloads for Neuron conversion
            ✓ Auto-fix Suggestions: kubectl commands and YAML patches

        Examples:
            "Validate with instance type: k8s/ instance_type=inf2.xlarge"
            "Training workload: k8s/training.yaml instance_type=trn1.32xlarge"
            "Check all manifests: k8s/ instance_type=inf2.24xlarge"
            "Strict validation: k8s/ strict_mode=true instance_type=inf2.xlarge"

        Integration Tips:
        - ALWAYS specify instance_type for accurate fix recommendations
        - Use after analyze_codebase to validate manifests match code requirements
        - Call during development to catch issues before deployment
        """
        return await k8s_service.validate_manifests(
            manifest_path=manifest_path,
            strict_mode=strict_mode,
            k8s_version=k8s_version,
            instance_type=instance_type
        )

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
    
    argument_parser = ArgumentParser()
    mode_result = argument_parser.parse_mode()
    
    console = Console()

    if mode_result:
        mode, query, output_json = mode_result
        
        if mode == "--analyze-codebase":
            # Parse query for path and options (support discover_patterns flag)
            repo_path = query
            discover_patterns = False
            
            # Check if query contains flags (space-separated)
            if " --discover-patterns" in query:
                repo_path = query.replace(" --discover-patterns", "").strip()
                discover_patterns = True
            
            if output_json:
                service = AnalyzeCodebaseService(NeuronCodeScanner)
                payload = asyncio.run(service.analyze_codebase_json(
                    repo_path=repo_path,
                    analyze_operators=False,
                    discover_patterns=discover_patterns
                ))
                console.print_json(data=payload)
            else:
                # For CLI display, pass discover_patterns to implementation
                from neuron_scanner.application.service import AnalyzeCodebaseService
                service = AnalyzeCodebaseService(NeuronCodeScanner)
                result = asyncio.run(service.analyze_codebase(
                    repo_path=repo_path,
                    analyze_operators=False,
                    discover_patterns=discover_patterns
                ))
                console.print(Markdown(result))
            return

        if mode == "--validate-k8s":
            # Parse query string using ArgumentParser (DRY - no duplication!)
            manifest_path, options = ArgumentParser.parse_encoded_query(query)
            parsed_options = ArgumentParser.get_k8s_options_with_defaults(options)

            if output_json:
                # Use shared service instance (has ML model manager)
                payload = asyncio.run(
                    k8s_service.validate_manifests_json(
                        manifest_path=manifest_path,
                        **parsed_options
                    )
                )
                console.print_json(data=payload)
            else:
                # Use shared service instance (has ML model manager)
                result = asyncio.run(
                    k8s_service.validate_manifests(
                        manifest_path=manifest_path,
                        **parsed_options
                    )
                )
                console.print(Markdown(result))
            return
    
    server_runner = ServerRunner(FastMCP, create_mcp_server)
    server_runner.run()


if __name__ == "__main__":
    main()
