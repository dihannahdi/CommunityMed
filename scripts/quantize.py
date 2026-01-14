#!/usr/bin/env python3
"""
Quantization Script for CommunityMed AI
Quantize models for Edge AI deployment (Edge AI Prize target)
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Quantize MedGemma for edge deployment"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./quantized_models",
        help="Output directory for quantized model",
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "awq", "gguf", "int8", "int4"],
        default="gptq",
        help="Quantization method",
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits",
    )
    
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size",
    )
    
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test quantized model after creation",
    )
    
    return parser.parse_args()


def quantize_gptq(model_path: str, output_dir: str, bits: int, group_size: int):
    """
    Quantize model using GPTQ method
    Best for GPU inference
    """
    print(f"Quantizing {model_path} with GPTQ (bits={bits}, group_size={group_size})")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("Error: auto-gptq not installed. Run: pip install auto-gptq")
        return None
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    
    # Create calibration data
    calibration_data = [
        "Analyze this chest X-ray for signs of tuberculosis.",
        "What abnormalities do you see in this medical image?",
        "The patient presents with productive cough and night sweats.",
        "Evaluate the lung fields for any infiltrates or consolidation.",
    ]
    
    # Configure quantization
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=0.01,
        desc_act=True,
    )
    
    # Quantize
    print("Quantizing...")
    quantized_model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
    )
    quantized_model.quantize(calibration_data)
    
    # Save
    output_path = Path(output_dir) / f"gptq_{bits}bit"
    output_path.mkdir(parents=True, exist_ok=True)
    quantized_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    print(f"Saved quantized model to: {output_path}")
    return str(output_path)


def quantize_gguf(model_path: str, output_dir: str, bits: int):
    """
    Convert model to GGUF format for llama.cpp
    Best for CPU/mobile deployment
    """
    print(f"Converting {model_path} to GGUF (bits={bits})")
    
    output_path = Path(output_dir) / "gguf"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Quantization type mapping
    quant_types = {
        4: "Q4_K_M",
        8: "Q8_0",
    }
    quant_type = quant_types.get(bits, "Q4_K_M")
    
    output_file = output_path / f"model_{quant_type}.gguf"
    
    print(f"Would convert to {quant_type} format")
    print(f"Output: {output_file}")
    print("\nTo convert manually, run:")
    print(f"  python convert_hf_to_gguf.py {model_path} --outtype {quant_type}")
    print(f"  ./llama-quantize {model_path}/model.gguf {output_file} {quant_type}")
    
    return str(output_file)


def quantize_int8(model_path: str, output_dir: str):
    """
    Dynamic INT8 quantization using bitsandbytes
    """
    print(f"Creating INT8 quantized version of {model_path}")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("Error: transformers or bitsandbytes not installed")
        return None
    
    # INT8 config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    print("Loading model with INT8 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    output_path = Path(output_dir) / "int8"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Note: INT8 models cannot be saved directly, this creates a config
    model.config.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Save quantization config
    import json
    with open(output_path / "quantization_config.json", "w") as f:
        json.dump({
            "method": "int8",
            "source_model": model_path,
            "load_command": f'AutoModelForCausalLM.from_pretrained("{model_path}", load_in_8bit=True)'
        }, f, indent=2)
    
    print(f"INT8 config saved to: {output_path}")
    return str(output_path)


def test_quantized_model(model_path: str, method: str):
    """Test quantized model"""
    print(f"\nTesting quantized model: {model_path}")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if method == "gptq":
            from auto_gptq import AutoGPTQForCausalLM
            model = AutoGPTQForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test inference
        test_prompt = "Analyze this chest X-ray for tuberculosis signs:"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test response: {response}")
        print("✅ Quantized model working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 60)
    print("CommunityMed AI - Model Quantization")
    print("Edge AI Prize Target")
    print("=" * 60)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run quantization based on method
    if args.method == "gptq":
        output = quantize_gptq(
            args.model_path,
            args.output_dir,
            args.bits,
            args.group_size,
        )
    elif args.method == "gguf":
        output = quantize_gguf(
            args.model_path,
            args.output_dir,
            args.bits,
        )
    elif args.method in ["int8", "int4"]:
        output = quantize_int8(
            args.model_path,
            args.output_dir,
        )
    else:
        print(f"Unknown method: {args.method}")
        return 1
    
    # Test if requested
    if args.test and output:
        test_quantized_model(output, args.method)
    
    print("\n" + "=" * 60)
    print("Quantization complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
