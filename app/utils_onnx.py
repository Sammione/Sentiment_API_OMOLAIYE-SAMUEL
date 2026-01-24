from __future__ import annotations
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

def export_to_onnx(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_path: Path):
    """
    Export a Hugging Face model to ONNX format.
    """
    model.eval()
    
    # Create dummy input
    dummy_text = "This is a sample sentence for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    # Define dynamic axes for variable length inputs
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14
    )
    print(f"Model exported to {output_path}")
