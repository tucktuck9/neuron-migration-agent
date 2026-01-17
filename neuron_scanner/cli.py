"""
Command-line interface for the Neuron Scanner.

Co-located with `mcp_server.py` so entrypoints live together at the package root.
"""

import os
import sys
import logging

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Allow running via `python neuron_scanner/cli.py ...` without installing the package.
# When installed, the console script already has the correct import path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        The model validation service uses Pydantic BaseSettings that reads env vars.
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
@click.pass_context
def main(
    ctx: click.Context,
    quiet: bool,
):
    """
    Accelerate your migration to AWS Neuron.
    """
    try:
        if quiet:
            logging.getLogger().setLevel(logging.ERROR)

        if ctx.invoked_subcommand is None:
            # No subcommand: show help (Stripe-like behavior).
            click.echo(ctx.get_help())
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
        
        from neuron_scanner.model_validation.service import ModelValidationService
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

