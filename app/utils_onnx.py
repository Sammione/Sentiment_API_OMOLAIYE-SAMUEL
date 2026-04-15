from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import settings

def export_to_onnx(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_path: Path):
    """
    Export a Hugging Face model to ONNX format.
    
    Args:
        model: The HuggingFace pretrained model to export.
        tokenizer: The tokenizer associated with the model.
        output_path: Path where the ONNX model will be saved.
    
    Raises:
        RuntimeError: If the ONNX export fails.
    """
    model.eval()
    
    # Create dummy input with consistent max_length from config
    dummy_text = "This is a sample sentence for ONNX export."
    inputs = tokenizer(
        dummy_text, 
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=settings.transformer_max_length
    )
    
    # Define dynamic axes for variable length inputs
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare input tuple for export
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    try:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True
        )
        print(f"Model exported to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {e}") from e
