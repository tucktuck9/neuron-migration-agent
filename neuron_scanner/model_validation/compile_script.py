#!/usr/bin/env python3
"""
Neuron Compilation Script for SageMaker Processing Jobs.

This script runs inside a SageMaker Processing container on ml.inf2.xlarge
and compiles PyTorch models to Neuron format using torch_neuronx.trace().

ARCHITECTURE:
    1. Extract model from model.tar.gz
    2. Auto-detect input shape from config.json (or use provided shape)
    3. Load model (TorchScript or HuggingFace)
    4. Create appropriate example inputs based on model type
    5. Compile with torch_neuronx.trace()
    6. Save compiled .neff artifact
    7. Write result.json with success/failure details

USAGE (inside container):
    python compile_script.py --input-dir /opt/ml/processing/input \
                             --output-dir /opt/ml/processing/output

OPTIONAL:
    --input-shape 1,512  # Override auto-detected shape
"""

import argparse
import json
import logging
import os
import sys
import tarfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Input Shape Auto-Detection
# =============================================================================

def auto_detect_input_shape(model_dir: str) -> Tuple[Tuple[int, ...], str, Dict[str, Any]]:
    """
    Read config.json and infer input shape automatically.
    
    HuggingFace models include config.json with:
    - max_position_embeddings → sequence length for text models
    - image_size → resolution for vision models
    - model_type → architecture hint (bert, gpt2, vit, etc.)
    
    Args:
        model_dir: Directory containing extracted model files
    
    Returns:
        Tuple of (input_shape, model_type, config_dict)
    """
    config_path = os.path.join(model_dir, "config.json")
    config = {}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Found config.json with model_type: {config.get('model_type', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to parse config.json: {e}")
    else:
        logger.info("No config.json found, using defaults")
    
    model_type = config.get("model_type", "unknown")
    
    # Vision models (ViT, CLIP, etc.)
    if "image_size" in config:
        size = config["image_size"]
        if isinstance(size, dict):
            size = size.get("height", 224)
        elif isinstance(size, list):
            size = size[0]
        logger.info(f"Detected vision model with image_size: {size}")
        return (1, 3, size, size), "vision", config
    
    # Text models (BERT, GPT, LLaMA, etc.)
    if "max_position_embeddings" in config:
        max_seq = config["max_position_embeddings"]
        # Cap at 512 for faster compilation during validation
        seq_len = min(max_seq, 512)
        logger.info(f"Detected text model with max_position_embeddings: {max_seq}, using: {seq_len}")
        return (1, seq_len), "text", config
    
    # Sentence Transformers often have max_seq_length
    if "max_seq_length" in config:
        seq_len = min(config["max_seq_length"], 512)
        logger.info(f"Detected sentence transformer with max_seq_length: {seq_len}")
        return (1, seq_len), "text", config
    
    # Default: assume text model with seq_len=512
    logger.info("Using default text model shape: (1, 512)")
    return (1, 512), "text", config


def create_example_inputs(
    input_shape: Tuple[int, ...],
    model_type: str,
    config: Dict[str, Any]
) -> Tuple:
    """
    Create appropriate example inputs based on model type.
    
    Args:
        input_shape: Detected or provided input shape
        model_type: Model type from config.json (bert, gpt2, vit, etc.)
        config: Full config dictionary
    
    Returns:
        Tuple of example input tensors for tracing
    """
    import torch
    
    architecture = config.get("model_type", "").lower()
    
    # Vision models
    if model_type == "vision":
        logger.info(f"Creating vision inputs with shape: {input_shape}")
        pixel_values = torch.randn(input_shape, dtype=torch.float32)
        return (pixel_values,)
    
    # Decoder-only models (GPT-2, LLaMA, Mistral, etc.)
    decoder_only_types = ["gpt2", "gpt_neo", "gpt_neox", "llama", "mistral", "mixtral", "falcon", "phi"]
    if any(t in architecture for t in decoder_only_types):
        logger.info(f"Creating decoder-only inputs (input_ids only) with shape: {input_shape}")
        input_ids = torch.randint(0, 10000, input_shape, dtype=torch.long)
        return (input_ids,)
    
    # Encoder-decoder models (T5, BART, etc.)
    encoder_decoder_types = ["t5", "bart", "mbart", "pegasus"]
    if any(t in architecture for t in encoder_decoder_types):
        logger.info(f"Creating encoder-decoder inputs with shape: {input_shape}")
        input_ids = torch.randint(0, 10000, input_shape, dtype=torch.long)
        attention_mask = torch.ones(input_shape, dtype=torch.long)
        decoder_input_ids = torch.randint(0, 10000, input_shape, dtype=torch.long)
        return (input_ids, attention_mask, decoder_input_ids)
    
    # Default: encoder-only models (BERT, RoBERTa, DistilBERT, sentence-transformers, etc.)
    logger.info(f"Creating encoder inputs (input_ids, attention_mask) with shape: {input_shape}")
    input_ids = torch.randint(0, 10000, input_shape, dtype=torch.long)
    attention_mask = torch.ones(input_shape, dtype=torch.long)
    return (input_ids, attention_mask)


# =============================================================================
# Model Loading
# =============================================================================

def extract_model(input_dir: str) -> str:
    """
    Extract model.tar.gz if present.
    
    Args:
        input_dir: Directory containing model files
    
    Returns:
        Path to extracted model directory
    """
    # Find tar.gz files
    tar_files = list(Path(input_dir).glob("*.tar.gz"))
    
    if tar_files:
        tar_path = tar_files[0]
        logger.info(f"Extracting {tar_path}")
        
        extract_dir = os.path.join(input_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        
        return extract_dir
    
    # No tar.gz found, assume files are already extracted
    logger.info("No tar.gz found, using input directory directly")
    return input_dir


def load_model(model_dir: str):
    """
    Load model from directory.
    
    Supports:
    - TorchScript models (model.pt)
    - HuggingFace models (with config.json)
    
    Args:
        model_dir: Directory containing model files
    
    Returns:
        Loaded PyTorch model
    """
    import torch
    
    # Check for TorchScript model
    torchscript_paths = [
        os.path.join(model_dir, "model.pt"),
        os.path.join(model_dir, "traced_model.pt"),
        os.path.join(model_dir, "model.pth"),
    ]
    
    for pt_path in torchscript_paths:
        if os.path.exists(pt_path):
            logger.info(f"Loading TorchScript model from: {pt_path}")
            model = torch.jit.load(pt_path)
            model.eval()
            return model
    
    # Check for HuggingFace model
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        logger.info("Loading HuggingFace model from directory")
        try:
            from transformers import AutoModel, AutoModelForSequenceClassification
            
            # Try AutoModel first (general purpose)
            try:
                model = AutoModel.from_pretrained(model_dir)
                model.eval()
                logger.info("Loaded with AutoModel")
                return model
            except Exception:
                pass
            
            # Try classification model
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                model.eval()
                logger.info("Loaded with AutoModelForSequenceClassification")
                return model
            except Exception:
                pass
            
        except ImportError:
            logger.warning("transformers not available, cannot load HuggingFace model")
    
    # List directory contents for debugging
    contents = os.listdir(model_dir)
    raise FileNotFoundError(
        f"Could not find loadable model in {model_dir}. "
        f"Expected model.pt (TorchScript) or HuggingFace model with config.json. "
        f"Found: {contents}"
    )


# =============================================================================
# Neuron Compilation
# =============================================================================

def compile_model(model, example_inputs: Tuple, output_dir: str) -> Dict[str, Any]:
    """
    Compile model to Neuron format using torch_neuronx.trace().
    
    Args:
        model: PyTorch model to compile
        example_inputs: Tuple of example input tensors
        output_dir: Directory to save compiled model
    
    Returns:
        Result dict with status and details
    """
    import torch
    
    try:
        import torch_neuronx
    except ImportError as e:
        return {
            "status": "ERROR",
            "error": "torch_neuronx not available in this environment",
            "details": str(e),
        }
    
    try:
        logger.info("Starting Neuron compilation with torch_neuronx.trace()...")
        logger.info(f"  Input shapes: {[tuple(t.shape) for t in example_inputs]}")
        
        # Compile the model
        compiled_model = torch_neuronx.trace(model, example_inputs)
        
        # Save the compiled model
        output_path = os.path.join(output_dir, "compiled_model.pt")
        compiled_model.save(output_path)
        
        logger.info(f"✅ Compilation successful! Saved to: {output_path}")
        
        return {
            "status": "COMPATIBLE",
            "output_path": output_path,
            "message": "Model compiled successfully for Neuron",
        }
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        
        logger.error(f"❌ Compilation failed: {error_msg}")
        logger.debug(f"Traceback:\n{tb}")
        
        return {
            "status": "INCOMPATIBLE",
            "error": error_msg,
            "traceback": tb,
            "message": f"Neuron compilation failed: {error_msg}",
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for SageMaker Processing job."""
    parser = argparse.ArgumentParser(description="Compile PyTorch model for Neuron")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/opt/ml/processing/input",
        help="Directory containing model files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/ml/processing/output",
        help="Directory to save compiled model and results"
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Override input shape (e.g., '1,512' for text or '1,3,224,224' for vision)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    result = {
        "status": "ERROR",
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
    }
    
    try:
        # Step 1: Extract model
        logger.info("=" * 60)
        logger.info("Step 1: Extracting model...")
        model_dir = extract_model(args.input_dir)
        result["model_dir"] = model_dir
        
        # Step 2: Auto-detect or parse input shape
        logger.info("=" * 60)
        logger.info("Step 2: Detecting input shape...")
        
        if args.input_shape:
            # Parse user-provided shape
            shape_parts = [int(x.strip()) for x in args.input_shape.split(",")]
            input_shape = tuple(shape_parts)
            logger.info(f"Using user-provided input shape: {input_shape}")
            
            # Still need to detect model type for example inputs
            _, model_type, config = auto_detect_input_shape(model_dir)
        else:
            # Auto-detect from config.json
            input_shape, model_type, config = auto_detect_input_shape(model_dir)
            logger.info(f"Auto-detected input shape: {input_shape}")
        
        result["input_shape"] = input_shape
        result["model_type"] = model_type
        result["detected_architecture"] = config.get("model_type", "unknown")
        
        # Step 3: Load model
        logger.info("=" * 60)
        logger.info("Step 3: Loading model...")
        model = load_model(model_dir)
        
        # Step 4: Create example inputs
        logger.info("=" * 60)
        logger.info("Step 4: Creating example inputs...")
        example_inputs = create_example_inputs(input_shape, model_type, config)
        
        # Step 5: Compile with Neuron
        logger.info("=" * 60)
        logger.info("Step 5: Compiling with Neuron SDK...")
        compile_result = compile_model(model, example_inputs, args.output_dir)
        result.update(compile_result)
        
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"Error during compilation: {e}")
        logger.debug(traceback.format_exc())
    
    # Write result.json
    result_path = os.path.join(args.output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Result written to: {result_path}")
    
    # Exit with appropriate code
    if result["status"] == "COMPATIBLE":
        logger.info("=" * 60)
        logger.info("✅ COMPILATION SUCCESSFUL")
        sys.exit(0)
    else:
        logger.info("=" * 60)
        logger.info(f"❌ COMPILATION FAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
