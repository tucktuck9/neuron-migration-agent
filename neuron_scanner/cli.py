"""
Command-line interface for the Neuron Scanner.

Co-located with `mcp_server.py` so entrypoints live together at the package root.
"""

import os
import sys
from pathlib import Path
import logging

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Allow running via `python neuron_scanner/cli.py ...` without installing the package.
# When installed, the console script already has the correct import path.
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


console = Console()
stderr_console = Console(stderr=True)

def _parse_input_shape(input_shape: str | None) -> tuple[int, ...] | None:
    """
    Parse CLI --input-shape argument into a tuple of ints.
    """
    if not input_shape:
        return None
    return tuple(int(x.strip()) for x in input_shape.split(","))


def _apply_validate_model_env_overrides(
    bucket: str | None,
    role_arn: str | None,
    region: str | None,
) -> None:
    """
    Apply validate-model CLI flags as env vars for the SageMaker validator.

    ARCHITECTURE:
        The recommendations validator uses Pydantic BaseSettings that reads env vars.
        CLI flags override env vars so users don't need to export variables manually.
    """
    if bucket is not None:
        os.environ["NEURON_VALIDATION_BUCKET"] = bucket
    if role_arn is not None:
        os.environ["NEURON_VALIDATION_ROLE_ARN"] = role_arn
    if region is not None:
        os.environ["AWS_REGION"] = region



class StripeStyleHelpFormatter(click.HelpFormatter):
    """Click help formatter with Stripe-like headings."""

    def write_heading(self, heading: str) -> None:
        if heading == "Options":
            heading = "Flags"
        # Bold the heading
        self.write(f"**{heading}**:\n")



class StripeStyleGroup(click.Group):
    """Click group that uses StripeStyleHelpFormatter."""

    def make_formatter(self, ctx: click.Context) -> click.HelpFormatter:
        return StripeStyleHelpFormatter()


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
    invoke_without_command=True,
    epilog='Use "neuron-scanner [command] --help" for more information about a command.',
    cls=StripeStyleGroup,
)
@click.version_option(version="0.1.0", prog_name="neuron-scanner")
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress non-essential output",
)
# Root-level CLI flags (preferred interface).
# These are intentionally supported in addition to subcommands.
@click.option(
    "--analyze-codebase",
    "analyze_codebase_path",
    type=click.Path(path_type=Path, exists=True),
    help="Analyze application code for migration readiness",
)
@click.option(
    "--validate-k8s",
    "validate_k8s_path",
    type=click.Path(path_type=Path, exists=True),
    help="Validate Kubernetes manifests for deployment",
)
@click.option(
    "--analyze-operators",
    "analyze_operators",
    is_flag=True,
    default=False,
    help="Run torch_neuronx operator compatibility analysis",
)
@click.option(
    "--discover-patterns",
    "discover_patterns",
    is_flag=True,
    default=False,
    help="Use ML to discover custom CUDA wrappers (experimental)",
)
@click.pass_context
def main(
    ctx: click.Context,
    quiet: bool,
    analyze_codebase_path: Path | None,
    validate_k8s_path: Path | None,
    analyze_operators: bool,
    discover_patterns: bool,
):
    """
    Accelerate your migration to AWS Neuron.
    """
    try:
        if quiet:
            logging.getLogger().setLevel(logging.ERROR)

        if ctx.invoked_subcommand is not None:
            return

        # Support root-level flags when no subcommand is provided.
        selected_root_modes = [
            mode for mode in [analyze_codebase_path is not None, validate_k8s_path is not None] if mode
        ]
        if selected_root_modes:
            if len(selected_root_modes) != 1:
                raise click.UsageError("Specify only one of --analyze-codebase or --validate-k8s.")
            if analyze_codebase_path is not None:
                ctx.invoke(
                    analyze_codebase,
                    repo_path=analyze_codebase_path,
                    quiet=quiet,
                    analyze_operators=analyze_operators,
                    discover_patterns=discover_patterns,
                )
                return
            ctx.invoke(validate_k8s, manifest_path=validate_k8s_path, quiet=quiet)
            return

        # No subcommand + no root flags: show help (Stripe-like behavior).
        click.echo(ctx.get_help())
    except Exception as e:
        stderr_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command("analyze-codebase")
@click.argument("repo_path", type=click.Path(path_type=Path, exists=True))
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output.")
@click.option(
    "--analyze-operators",
    is_flag=True,
    default=False,
    help="Run torch_neuronx operator compatibility analysis",
)
@click.option(
    "--discover-patterns",
    is_flag=True,
    default=False,
    help="Use ML to discover custom CUDA wrappers (experimental)",
)
def analyze_codebase(repo_path: Path, quiet: bool, analyze_operators: bool, discover_patterns: bool):
    """Analyze application code for CUDA patterns and migration readiness."""
    try:
        if quiet:
            logging.getLogger().setLevel(logging.ERROR)
        import asyncio

        from neuron_scanner.application.scanner import NeuronCodeScanner
        from neuron_scanner.application.service import AnalyzeCodebaseService

        service = AnalyzeCodebaseService(NeuronCodeScanner)
        markdown_text = asyncio.run(
            service.analyze_codebase(
                repo_path=str(repo_path),
                analyze_operators=analyze_operators,
                discover_patterns=discover_patterns,
            )
        )
        console.print(Markdown(markdown_text))
    except Exception as e:
        stderr_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command("validate-k8s")
@click.argument("manifest_path", type=click.Path(path_type=Path, exists=True))
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.option(
    "--strict-mode",
    is_flag=True,
    help="Enable strict Kubernetes schema validation",
)
@click.option(
    "--instance-type",
    type=str,
    default=None,
    help="Target instance type for recommendations (e.g. trn2.24xlarge)",
)
@click.option(
    "--k8s-version",
    type=str,
    default=None,
    help="Kubernetes version to validate against (e.g. 1.28, 1.29, 1.33)",
)
def validate_k8s(manifest_path: Path, quiet: bool, strict_mode: bool, instance_type: str | None, k8s_version: str | None):
    """Validate Kubernetes manifests for Neuron deployment compatibility."""
    try:
        if quiet:
            logging.getLogger().setLevel(logging.ERROR)
        import asyncio
        from neuron_scanner.kubernetes.service import KubernetesValidatorService
        from neuron_scanner.core.model_manager import MLModelManager

        # Initialize with ML model manager for intelligent workload classification
        model_manager = MLModelManager()
        service = KubernetesValidatorService(model_manager)
        
        markdown_text = asyncio.run(
            service.validate_manifests(
                manifest_path=str(manifest_path),
                strict_mode=strict_mode,
                instance_type=instance_type,
                k8s_version=k8s_version,
            )
        )
        console.print(Markdown(markdown_text))
    except Exception as e:
        stderr_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command("validate-model")
@click.argument("s3_uri", type=str)
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.option(
    "--bucket",
    type=str,
    default=None,
    help="S3 bucket used for compile artifacts (overrides NEURON_VALIDATION_BUCKET)",
)
@click.option(
    "--role-arn",
    type=str,
    default=None,
    help="IAM role ARN for SageMaker (overrides NEURON_VALIDATION_ROLE_ARN)",
)
@click.option(
    "--region",
    type=str,
    default=None,
    help="AWS region (overrides AWS_REGION; default is us-east-1 if unset)",
)
@click.option(
    "--input-shape",
    type=str,
    default=None,
    help="Override input tensor shape (auto-detected from config.json)",
)
@click.option(
    "--instance-type",
    type=str,
    default=None,
    help="Override SageMaker instance type for compilation",
)
def validate_model(
    s3_uri: str,
    quiet: bool,
    bucket: str | None,
    role_arn: str | None,
    region: str | None,
    input_shape: str | None,
    instance_type: str | None,
):
    """Compile and validate model on AWS Neuron hardware using SageMaker."""
    try:
        if quiet:
            logging.getLogger().setLevel(logging.ERROR)
        
        from neuron_scanner.recommendations.service import ModelValidationService
        service = ModelValidationService()
        
        # Enforce S3 URI
        if not s3_uri.startswith("s3://"):
            raise click.UsageError(
                "SageMaker validation requires an S3 URI. "
                "Example: neuron-scanner validate-model s3://bucket/model.tar.gz"
            )

        # Optional env var overrides via CLI flags.
        # These must be applied before the service creates the validator.
        if (bucket is None) ^ (role_arn is None):
            raise click.UsageError("Provide both --bucket and --role-arn (or neither).")
        _apply_validate_model_env_overrides(bucket=bucket, role_arn=role_arn, region=region)
        
        # Parse input shape
        try:
            parsed_shape = _parse_input_shape(input_shape)
        except ValueError:
            raise click.UsageError(
                f"Invalid --input-shape format: {input_shape}. "
                "Use comma-separated integers, e.g., 1,3,224,224"
            )
        
        output = service.validate_s3_uri(
            s3_uri=s3_uri,
            input_shape=parsed_shape,
            instance_type=instance_type,
        )
        
        console.print(Markdown(output))
    except Exception as e:
        stderr_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure help/usage matches the installed console script name.
    main(prog_name="neuron-scanner")

