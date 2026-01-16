#!/usr/bin/env python3
"""
Model Preparation Script for Neuron Validation.

This script prepares any HuggingFace model for Neuron compilation by:
1. Loading the model from HuggingFace Hub or local path
2. Auto-detecting model type and input requirements
3. Saving in HuggingFace format (includes config.json for auto-detection)
4. Packaging into model.tar.gz for upload to S3

USAGE:
    # From HuggingFace Hub (uses model's full sequence length)
    MODEL_ID=bert-base-uncased python prepare_model.py
    
    # From local directory
    MODEL_ID=./my_finetuned_model python prepare_model.py
    
    # With custom output name
    MODEL_ID=distilbert-base-uncased OUTPUT_NAME=my-model python prepare_model.py
    
    # With custom sequence length (for faster validation)
    MODEL_ID=meta-llama/Llama-2-7b-hf MAX_SEQ_LENGTH=512 python prepare_model.py

ENVIRONMENT VARIABLES:
    MODEL_ID        (required) HuggingFace model ID or local path
    OUTPUT_NAME     (optional) Output filename without extension (default: model)
    MAX_SEQ_LENGTH  (optional) Override sequence length (default: from model config)
    KEEP_ARTIFACTS  (optional) Set to "true" to keep model_artifacts/ folder

INPUT SHAPE AUTO-DETECTION:
    The script reads the model's config.json to determine input shapes:
    
    | Config Key              | Model Type          | Example Value |
    |-------------------------|---------------------|---------------|
    | max_position_embeddings | BERT, RoBERTa, etc. | 512, 1024     |
    | n_positions             | GPT-2               | 1024          |
    | max_seq_length          | Sentence Transformers | 256, 512    |
    | sliding_window          | Mistral             | 4096          |
    | image_size              | ViT, CLIP           | 224, 384      |
    
    If no config key is found, defaults to 512 for text models.

SUPPORTED MODEL TYPES:
    - Text (BERT, RoBERTa, DistilBERT, etc.)
    - Decoder-only (GPT-2, LLaMA, Mistral, etc.)
    - Encoder-decoder (T5, BART, etc.)
    - Vision (ViT, CLIP, etc.)
    - Sentence Transformers
    - Sequence Classification
    - Token Classification
    - Question Answering
"""

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# =============================================================================
# Model Type Detection
# =============================================================================

# Model architecture categories
DECODER_ONLY_TYPES = [
    "gpt2", "gpt_neo", "gpt_neox", "gptj", "llama", "mistral", 
    "mixtral", "falcon", "phi", "qwen", "gemma", "codegen", "bloom"
]

ENCODER_DECODER_TYPES = [
    "t5", "bart", "mbart", "pegasus", "marian", "mt5", "longt5"
]

VISION_TYPES = [
    "vit", "clip", "deit", "beit", "swin", "convnext", "resnet", 
    "dinov2", "siglip"
]

MULTIMODAL_TYPES = [
    "clip", "blip", "llava", "flamingo", "idefics"
]


def detect_model_category(config: Dict[str, Any]) -> str:
    """
    Detect model category from config.json.
    
    Returns:
        One of: 'vision', 'decoder_only', 'encoder_decoder', 'encoder', 'multimodal'
    """
    model_type = config.get("model_type", "").lower()
    architectures = config.get("architectures", [])
    arch_str = " ".join(architectures).lower() if architectures else ""
    
    # Check for vision models
    if "image_size" in config or any(v in model_type for v in VISION_TYPES):
        if any(m in model_type for m in MULTIMODAL_TYPES):
            return "multimodal"
        return "vision"
    
    # Check for decoder-only (autoregressive) models
    if any(d in model_type for d in DECODER_ONLY_TYPES):
        return "decoder_only"
    if "causallm" in arch_str or "forgenereration" in arch_str.replace(" ", ""):
        return "decoder_only"
    
    # Check for encoder-decoder models
    if any(e in model_type for e in ENCODER_DECODER_TYPES):
        return "encoder_decoder"
    if "seq2seq" in arch_str or "conditionalgeneration" in arch_str.replace(" ", ""):
        return "encoder_decoder"
    
    # Default: encoder-only (BERT, RoBERTa, etc.)
    return "encoder"


def get_input_shape_from_config(
    config: Dict[str, Any], 
    model_category: str,
    max_seq_cap: Optional[int] = None
) -> Dict[str, Any]:
    """
    Determine input shapes from model config.
    
    Args:
        config: Model config dictionary
        model_category: Detected model category
        max_seq_cap: Optional cap on sequence length (None = use full model capacity)
    
    Returns:
        Dict with input_shape, input_names, and other metadata
    """
    result = {
        "model_category": model_category,
        "batch_size": 1,
    }
    
    # Vision models
    if model_category == "vision":
        image_size = config.get("image_size", 224)
        if isinstance(image_size, dict):
            image_size = image_size.get("height", 224)
        elif isinstance(image_size, list):
            image_size = image_size[0]
        
        num_channels = config.get("num_channels", 3)
        
        result.update({
            "input_shape": (1, num_channels, image_size, image_size),
            "input_names": ["pixel_values"],
            "input_dtype": "float32",
            "image_size": image_size,
        })
        return result
    
    # Text models - determine sequence length from config (100% dynamic!)
    max_seq_length = None
    source = "default"
    
    # Try various config keys for sequence length (in order of preference)
    if "max_position_embeddings" in config:
        max_seq_length = config["max_position_embeddings"]
        source = "max_position_embeddings"
    elif "n_positions" in config:  # GPT-2 style
        max_seq_length = config["n_positions"]
        source = "n_positions"
    elif "max_seq_length" in config:  # Sentence transformers
        max_seq_length = config["max_seq_length"]
        source = "max_seq_length"
    elif "max_sequence_length" in config:
        max_seq_length = config["max_sequence_length"]
        source = "max_sequence_length"
    elif "sliding_window" in config:  # Mistral-style
        max_seq_length = config["sliding_window"]
        source = "sliding_window"
    
    # If still None, use a reasonable default
    if max_seq_length is None:
        max_seq_length = 512
        source = "default (no config key found)"
    
    # Store original before any capping
    original_seq_length = max_seq_length
    
    # Apply optional cap (for faster validation)
    if max_seq_cap is not None and max_seq_length > max_seq_cap:
        max_seq_length = max_seq_cap
        print(f"  Note: Capped sequence length from {original_seq_length} to {max_seq_cap} (set MAX_SEQ_LENGTH to override)")
    
    print(f"  Sequence length: {max_seq_length} (from {source})")
    
    result["sequence_length"] = max_seq_length
    result["vocab_size"] = config.get("vocab_size", 30522)
    
    # Set input names based on model category
    if model_category == "decoder_only":
        result.update({
            "input_shape": (1, max_seq_length),
            "input_names": ["input_ids"],
            "input_dtype": "long",
        })
    elif model_category == "encoder_decoder":
        result.update({
            "input_shape": (1, max_seq_length),
            "input_names": ["input_ids", "attention_mask", "decoder_input_ids"],
            "input_dtype": "long",
        })
    else:  # encoder (BERT-like)
        result.update({
            "input_shape": (1, max_seq_length),
            "input_names": ["input_ids", "attention_mask"],
            "input_dtype": "long",
        })
    
    return result


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_processor(model_id: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load model, processor/tokenizer, and config from HuggingFace.
    
    Returns:
        Tuple of (model, processor_or_tokenizer, config_dict)
    """
    from transformers import AutoConfig
    
    print(f"Loading model: {model_id}")
    
    # Load config first to determine model type
    config = AutoConfig.from_pretrained(model_id)
    config_dict = config.to_dict()
    
    model_category = detect_model_category(config_dict)
    print(f"  Detected category: {model_category}")
    print(f"  Model type: {config_dict.get('model_type', 'unknown')}")
    
    # Load model based on architecture
    architectures = config_dict.get("architectures", [])
    arch_str = " ".join(architectures).lower() if architectures else ""
    
    model = None
    processor = None
    
    try:
        if model_category == "vision":
            from transformers import AutoImageProcessor, AutoModel
            print("  Loading as vision model...")
            model = AutoModel.from_pretrained(model_id)
            processor = AutoImageProcessor.from_pretrained(model_id)
            
        elif "sequenceclassification" in arch_str:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            print("  Loading as sequence classification model...")
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            processor = AutoTokenizer.from_pretrained(model_id)
            
        elif "tokenclassification" in arch_str:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            print("  Loading as token classification model...")
            model = AutoModelForTokenClassification.from_pretrained(model_id)
            processor = AutoTokenizer.from_pretrained(model_id)
            
        elif "questionanswering" in arch_str:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer
            print("  Loading as question answering model...")
            model = AutoModelForQuestionAnswering.from_pretrained(model_id)
            processor = AutoTokenizer.from_pretrained(model_id)
            
        elif model_category == "decoder_only":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("  Loading as causal LM (decoder-only)...")
            model = AutoModelForCausalLM.from_pretrained(model_id)
            processor = AutoTokenizer.from_pretrained(model_id)
            
        elif model_category == "encoder_decoder":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print("  Loading as seq2seq model...")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            processor = AutoTokenizer.from_pretrained(model_id)
            
        else:
            # Default: try AutoModel (works for most encoders)
            from transformers import AutoModel, AutoTokenizer
            print("  Loading as generic model...")
            model = AutoModel.from_pretrained(model_id)
            processor = AutoTokenizer.from_pretrained(model_id)
            
    except Exception as e:
        print(f"  Warning: Failed with specific loader: {e}")
        print("  Falling back to AutoModel...")
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(model_id)
        try:
            processor = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            processor = None
    
    model.eval()
    print(f"  Model loaded successfully!")
    
    return model, processor, config_dict


# =============================================================================
# Model Saving
# =============================================================================

def save_model_for_neuron(
    model: Any,
    processor: Any,
    config_dict: Dict[str, Any],
    output_dir: str,
    max_seq_cap: Optional[int] = None
) -> Dict[str, Any]:
    """
    Save model in HuggingFace format with Neuron metadata.
    
    Args:
        model: The loaded model
        processor: Tokenizer or image processor
        config_dict: Model config dictionary
        output_dir: Directory to save artifacts
        max_seq_cap: Optional cap on sequence length (None = use full model capacity)
    
    Returns:
        Input shape metadata for reference
    """
    print(f"\nSaving model to {output_dir}/")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(output_dir)
    print("  ✅ Saved model weights")
    
    if processor is not None:
        processor.save_pretrained(output_dir)
        print("  ✅ Saved tokenizer/processor")
    
    # Detect input requirements
    model_category = detect_model_category(config_dict)
    input_info = get_input_shape_from_config(config_dict, model_category, max_seq_cap)
    
    # Add Neuron-specific metadata to config
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        # Add Neuron metadata
        saved_config["_neuron_input_shape"] = list(input_info["input_shape"])
        saved_config["_neuron_input_names"] = input_info["input_names"]
        saved_config["_neuron_model_category"] = input_info["model_category"]
        
        with open(config_path, "w") as f:
            json.dump(saved_config, f, indent=2)
        
        print("  ✅ Added Neuron metadata to config.json")
    
    return input_info


def create_tarball(output_dir: str, tarball_name: str) -> str:
    """
    Create tar.gz archive from output directory.
    
    Returns:
        Path to created tarball
    """
    print(f"\nCreating {tarball_name}...")
    
    with tarfile.open(tarball_name, "w:gz") as tar:
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            tar.add(filepath, arcname=filename)
            print(f"  + {filename}")
    
    # Get file size
    size_mb = os.path.getsize(tarball_name) / (1024 * 1024)
    print(f"\n✅ Created {tarball_name} ({size_mb:.1f} MB)")
    
    return tarball_name


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    # Get configuration from environment
model_id = os.getenv("MODEL_ID")
    if not model_id:
        print("ERROR: MODEL_ID environment variable required")
        print("\nUsage:")
        print("  MODEL_ID=bert-base-uncased python prepare_model.py")
        print("  MODEL_ID=./my_local_model python prepare_model.py")
        print("\nOptional environment variables:")
        print("  OUTPUT_NAME=my-model      # Output filename (default: model)")
        print("  MAX_SEQ_LENGTH=2048       # Sequence length (default: from config)")
        print("  KEEP_ARTIFACTS=true       # Keep model_artifacts/ folder")
        sys.exit(1)
    
    output_name = os.getenv("OUTPUT_NAME", "model")
    output_dir = os.getenv("OUTPUT_DIR", "model_artifacts")
    
    # Parse MAX_SEQ_LENGTH - None means use full model capacity (100% dynamic)
    max_seq_str = os.getenv("MAX_SEQ_LENGTH")
    max_seq_cap = int(max_seq_str) if max_seq_str else None
    
    print("=" * 60)
    print("Neuron Model Preparation Script")
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Output: {output_name}.tar.gz")
    if max_seq_cap:
        print(f"Max sequence length: {max_seq_cap} (from MAX_SEQ_LENGTH)")
    else:
        print(f"Max sequence length: auto (from model config)")
    print("=" * 60)
    
    try:
        # Load model
        model, processor, config_dict = load_model_and_processor(model_id)
        
        # Save in HuggingFace format
        input_info = save_model_for_neuron(model, processor, config_dict, output_dir, max_seq_cap)
        
        # Create tarball
        tarball_path = create_tarball(output_dir, f"{output_name}.tar.gz")

        # Print summary
        print("\n" + "=" * 60)
        print("SUCCESS! Model prepared for Neuron validation")
        print("=" * 60)
        print(f"\nModel details:")
        print(f"  Type: {config_dict.get('model_type', 'unknown')}")
        print(f"  Category: {input_info['model_category']}")
        print(f"  Input shape: {input_info['input_shape']}")
        print(f"  Input names: {input_info['input_names']}")
        
        print(f"\nNext steps:")
        print(f"  1. Upload to S3:")
        print(f"     aws s3 cp {tarball_path} s3://YOUR_BUCKET/models/")
        print(f"\n  2. Run validation:")
        print(f"     neuron-scanner validate-model s3://YOUR_BUCKET/models/{output_name}.tar.gz")
        
        # Cleanup
        if os.getenv("KEEP_ARTIFACTS", "").lower() != "true":
            shutil.rmtree(output_dir)
            print(f"\n(Cleaned up {output_dir}/, set KEEP_ARTIFACTS=true to keep)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
