# neuron-migration-agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)]()

> **Phase 1: `validate-model` — Neuron Compatibility Validator**
> Validate model compatibility with AWS Neuron (Inferentia/Trainium) by running real compilation checks on a temporary SageMaker notebook instance.

Neuron Migrator is a **CLI tool** and an **MCP server**:
- **CLI (`neuron-scanner`)**: Run Neuron migration tools from your terminal (today: `validate-model`, more coming).
- **MCP server (`neuron_scanner/mcp_server.py`)**: Runs locally as a migration agent that exposes the same tools to MCP clients like Claude Desktop/Cursor.

## 📖 Table of Contents
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Tools](#-tools)
- [CLI Usage](#-cli-usage)
- [Setup Claude Code CLI](#-setup-claude-code-cli)
- [Setup MCP Server](#-setup-mcp-server)
- [Troubleshooting](#troubleshooting-common-issues)


## 📋 Prerequisites

Before you begin, complete these tasks:

- [Install Python 3.10 or higher](https://www.python.org/downloads/release/python-3100/)
- [Download Claude Desktop](https://claude.com/download)


## 🚀 Quickstart

```bash
# 1. Clone and setup virtual environment
git clone https://github.com/aws-samples/neuron-migrator.git
cd neuron-migrator
python3.11 -m venv neuron-env
source neuron-env/bin/activate

# 2. Install dependencies
python3 -m pip install --upgrade pip setuptools wheel
pip3 install -e ".[sagemaker]"

# 3. Verify installation
neuron-scanner validate-model --help
```

## ✨ Tools

### `validate_model`

Verify model compatibility with AWS Neuron hardware by running actual compilation jobs on SageMaker Notebook instances. This tool manages the entire lifecycle: it automatically provisions a temporary SageMaker notebook instance, deploys your model, runs `torch_neuronx.trace()` to verify compilation, and then immediately terminates the instance upon successful compilation to minimize costs. It supports auto-detection of input shapes from configuration files (`config.json`) inside your S3 tarball and provides detailed logs for debugging compilation issues, building example inputs based on model type/architecture (e.g., vision, decoder-only, encoder-decoder, encoder-only). This "ground truth" validation is essential for confirming that your specific model architecture and custom layers can be successfully compiled for Inferentia or Trainium before you commit to a full migration. To use this tool, you need either a TorchScript model (model.pt, traced_model.pt, model.pth) or a HuggingFace model (via transformers, as model.tar.gz) in your Amazon S3 bucket. For setup instructions, see [Advanced Tutorials](ADVANCED-TUTORIAL.md).

Parameters:
- `s3_uri` (required): S3 URI to model.tar.gz file (e.g., s3://my-bucket/models/model.tar.gz)
- `instance_type` (optional): Override SageMaker instance type (e.g., ml.inf2.xlarge, ml.trn1.2xlarge)
- `input_shape` (optional): Override input tensor shape (auto-detected from config.json, format: comma-separated integers)

## Setup

### 1. Create an IAM Role

Create an IAM Role named `NeuronValidationRole` with SageMaker as a trusted entity. Replace `my-neuron-testing-bucket` in the following steps with your actual S3 bucket name.

1. Open the [IAM Console](https://console.aws.amazon.com/iam/)
2. Go to **Roles → Create role**
3. Select **AWS service** → **SageMaker** → **SageMaker - Execution**
4. Click **Next** (this automatically sets up the trust relationship)
5. Click **Create policy**, paste the JSON below, name it `NeuronValidationPolicy`
6. Attach the policy to the role and name the role `NeuronValidationRole`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerNotebookInstancePermissions",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateNotebookInstance",
        "sagemaker:DeleteNotebookInstance",
        "sagemaker:DescribeNotebookInstance",
        "sagemaker:StartNotebookInstance",
        "sagemaker:StopNotebookInstance",
        "sagemaker:CreateNotebookInstanceLifecycleConfig",
        "sagemaker:DeleteNotebookInstanceLifecycleConfig",
        "sagemaker:DescribeNotebookInstanceLifecycleConfig",
        "sagemaker:UpdateNotebookInstanceLifecycleConfig"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ECRLoginAndPull",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3ModelAndScriptAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::my-neuron-testing-bucket/*"
    },
    {
      "Sid": "S3BucketListAccess",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::my-neuron-testing-bucket"
    },
    {
      "Sid": "CloudWatchLogsAccess",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    },
    {
      "Sid": "IAMPassRoleForNotebook",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::*:role/NeuronValidationRole",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "sagemaker.amazonaws.com"
        }
      }
    }
  ]
}
```

### 2. Add a Trust Policy
1. Open the [IAM Console](https://console.aws.amazon.com/iam/)
2. Go to **Roles**
3. Search for and select your role: `NeuronValidationRole`.
4. Go to the **Trust relationships** tab (it's separate from the Permissions tab).
5. Click **Edit trust policy**.
6. Add the Sagemaker trust policy JSON. Replace the sample value for the AWS account(`01234567890`):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::01234567890:root"
            },
            "Action": "sts:AssumeRole",
            "Condition": {}
        },
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
    }
  ]
}
```

### 3. Set Environment Variables

Replace the sample values for the S3 bucket (`my-neuron-testing-bucket`) and AWS account (`123456789012`) with actual values:

```bash
# Required
export NEURON_VALIDATION_BUCKET=my-neuron-testing-bucket         # S3 bucket for compiled model output
export NEURON_VALIDATION_ROLE_ARN=arn:aws:iam::123456789012:role/NeuronValidationRole
   
# Specify the region of your S3 bucket
export AWS_REGION=us-east-1
```

## 💻 CLI Usage

This project ships with a command-line interface (`neuron-scanner`) for the Neuron Migrator toolset. Today, it includes **`validate-model`**; additional tools will be added over time. 

### Commands
For a high-level overview of commands and options, run `neuron-scanner validate-model --help`.

#### `validate-model`

Validate Neuron compatibility by compiling your model on a temporary SageMaker Notebook instance (e.g., uses Inferentia by default). 

```bash
neuron-scanner validate-model s3://your-bucket/models/model.tar.gz
```

To validate model compilation on specific hardware, use `--instance-type`.

```bash
neuron-scanner validate-model s3://your-bucket/model.tar.gz --instance-type ml.trn1.2xlarge
```

If auto-detection fails or you need to test a specific sequence length (e.g., batch size 1, sequence length 256), use `--input-shape` to override it manually.

```bash
neuron-scanner validate-model s3://your-bucket/model.tar.gz --input-shape 1,256
```

## 💻 Setup Claude Code CLI

If using the Claude Desktop app, Claude Code CLI is required to view and access code on your local machine.

```bash
# Install and configure
curl -fsSL https://claude.ai/install.sh | bash
source ~/.bashrc  # or ~/.zshrc
claude --version
```

## 🤖 Setup MCP Server

When you setup the MCP server (neuron-scanner), it runs on your local machine with access to your local filesystem. This allows MCP-compatible tools like Claude Desktop to call MCP server tools (like analyze_codebase), control your local machine to run shell commands, and enable read access to any files on your local machine. This integration transforms your IDE into an intelligent migration assistant that can read your repository, run analysis tools, and generate context-aware code patches without manual copy-pasting.

**Configure Claude Desktop**: Create `claude_desktop_config.json` (at `~/Library/Application\ Support/Claude/claude_desktop_config.json` on macOS) or add directly in **Cursor** (Settings > Cursor Settings > Tools & MCP). Ensure the paths match your local installation.

```json
{
  "mcpServers": {
    "neuron-scanner": {
      "command": "/absolute/path/to/neuron-migrator/neuron-env/bin/python3",
      "args": ["/absolute/path/to/neuron-migrator/neuron_scanner/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/neuron-migrator"
      }
    }
  }
}
```

## 🆘 Troubleshooting

Resolve common issues related to environment setup and tool execution. 
 
- **Python dependency issues**: For Python dependency issues, verify you are using Python 3.10+ and have activated your virtual environment (e.g., `source neuron-env/bin/activate`). 
- **MCP server connection issues**: If the MCP server fails to connect, double-check the absolute paths to your configuration file, as these must point to the valid virtual environment python executable and the server script (e.g., on macOS, try `/Users/<mac-user>/<your-path>`).

## 📄 License

This project is licensed under the Apache 2.0 License. See [`LICENSE`](LICENSE).
