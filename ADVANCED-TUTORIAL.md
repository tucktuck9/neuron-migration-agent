# Advanced Tutorials

The `validate-model` tool uses SageMaker Notebook Instances running on Neuron-enabled hardware (ml.inf2.xlarge, ml.trn1.2xlarge) to compile your model with `torch_neuronx.trace()`. This approach supports modern PyTorch versions (2.5+) and automatically detects input shapes from your model's `config.json`.

**How It Works:**
1. Creates a temporary SageMaker Notebook Instance with lifecycle configuration
2. Instance boots and automatically runs the compilation script
3. Script downloads your model, runs `torch_neuronx.trace()`, uploads results
4. Dynamically configures inputs based on the config.json file in your `.tar.gz` file
5. Tool polls S3 for results, then cleans up the instance

**Supported Targets:**
- [AWS Inferentia EC2 Instances](https://aws.amazon.com/ai/machine-learning/inferentia/)
- [AWS Trainium EC2 Instances](https://aws.amazon.com/ai/machine-learning/trainium/)

## 📖 Table of Contents

- [1. (Optional) Increase SageMaker Service Quotas](#1-optional-increase-service-quotas)
- [2. (Optional) Download Hugging Face Model](#2-optional-download-hugging-face-model)
- [3. (Optional) Upload the Model to Amazon S3 (Python)](#3-optional-upload-the-model-to-amazon-s3-python)
- [4. (Required) Create an IAM Role](#required-create-an-iam-role)
- [5. (Required) Add Trust Policy](#required-add-trust-policy)
- [6. (Required) Set Environment Variables](#required-set-environment-variables)


## 1. (Optional) Increase SageMaker Service Quotas
If you haven't already used SageMaker Notebooks, you may need to increase service quotas to use Sagemaker notebook instances.

1. Go to the [Sagemaker Quotas Console](https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas)
2. Search for the instance type for notebook instance usage, for example: "ml.inf2.xlarge for notebook instance usage"
3. Click "Request quota increase" or "Request increase at account level" and request at least 1 instance

![Claude Code Menu](images/console-service-quotas.png)

## 2. (Optional) Download Hugging Face Model

### Prerequisites
- Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli).

### Setup Hugging Face CLI 

```bash
# 1. Add Hugging Face token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# 2. Login
hf auth login --token hf_xxxxxxxxxxxxxxxxxxxxx

# 3. Verify authentication by getting private or public model info
hf models info <organization>/<model-name>
```

### 3. (Optional) Upload the Model to Amazon S3 (Python)

If you don't already have model artifacts in an Amazon S3 bucket, you can prepare a model directly from Hugging Face (by providing the Model ID) or from a local directory. The provided script (experimental) will automatically download, configure, and package it (as `.tar.gz` with a `config.json` containing input parameters) for Neuron validation.

In your terminal, run:

```bash
# 1. Export AWS access key - Example: 
export AWS_ACCESS_KEY_ID=EXAMPLE123456789

# 2. Export AWS secret key - Example:
export AWS_SECRET_ACCESS_KEY=EXAMPLE123456789

# 3. Export model ID (Use Hugging Face ID to download automatically) - Example:
export MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# 4. Export S3 bucket name - Example:
export S3_BUCKET="my-neuron-testing-bucket"

# 5. Verify by listing S3 buckets
aws s3 ls

# 6. Create a /models directory in your bucket
aws s3api put-object --bucket $S3_BUCKET --key models/ --content-length 0

# 7. Package the model (Downloads and packages automatically)
python3 prepare_model.py

# 8. Upload the compressed tarball model.tar.gz file to S3
aws s3 cp model.tar.gz s3://$S3_BUCKET/models/model.tar.gz --acl bucket-owner-full-control

# 9. Verify the upload
aws s3 ls s3://$S3_BUCKET/models/
```

## 4. (Required) Create an IAM Role

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

## 5. (Required) Add Trust Policy
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

## 6. (Required) Set Environment Variables

Replace the sample values for the S3 bucket (`my-neuron-testing-bucket`) and AWS account (`123456789012`) with actual values:

```bash
# Required
export NEURON_VALIDATION_BUCKET=my-neuron-testing-bucket         # S3 bucket for compiled model output
export NEURON_VALIDATION_ROLE_ARN=arn:aws:iam::123456789012:role/NeuronValidationRole
   
# Specify the region of your S3 bucket
export AWS_REGION=us-east-1
```


