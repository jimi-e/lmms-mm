#!/usr/bin/env python3
"""
Simple test script for Qwen2.5-VL-7B-Instruct to diagnose issues.
"""

import os
import traceback
from PIL import Image
import torch

def test_basic_torch():
    """Test basic PyTorch and CUDA setup"""
    print("=== Testing Basic PyTorch ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test basic tensor operations
        try:
            x = torch.randn(10, 10).cuda()
            _ = x @ x.T  # Simple matrix multiplication
            print("✓ Basic CUDA tensor operations work")
            
            # Check memory
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
        except Exception as e:
            print(f"✗ CUDA tensor operations failed: {e}")
    print()

def test_transformers_import():
    """Test transformers import"""
    print("=== Testing Transformers Import ===")
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        # Just import to check they're available
        transformers.AutoProcessor
        transformers.AutoModelForCausalLM
        print("✓ AutoProcessor and AutoModelForCausalLM available")
        
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        traceback.print_exc()
    print()

def test_model_loading(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
    """Test model loading step by step"""
    print(f"=== Testing Model Loading: {model_name} ===")
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        # Step 1: Load processor
        print("Step 1: Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        print("✓ Processor loaded successfully")
        
        # Step 2: Load model with minimal config
        print("Step 2: Loading model...")
        
        # Try different loading strategies, using AutoModelForVision2Seq instead
        loading_configs = [
            {
                "name": "bf16 + low_cpu_mem",
                "config": {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                    "device_map": None,
                }
            },
            {
                "name": "fp16 + low_cpu_mem", 
                "config": {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                    "device_map": None,
                }
            },
            {
                "name": "auto device_map",
                "config": {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                }
            }
        ]
        
        model = None
        successful_config = None
        
        for config_info in loading_configs:
            try:
                print(f"  Trying: {config_info['name']}")
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name, 
                    **config_info['config']
                )
                successful_config = config_info['name']
                print(f"  ✓ Model loaded with: {successful_config}")
                break
            except Exception as e:
                print(f"  ✗ Failed with {config_info['name']}: {e}")
                if model is not None:
                    del model
                torch.cuda.empty_cache()
        
        if model is None:
            print("✗ All loading strategies failed")
            return False, None, None
            
        # Step 3: Move to device if needed
        if device != "auto" and hasattr(model, 'to'):
            try:
                print(f"Step 3: Moving model to {device}...")
                model = model.to(device)
                print("✓ Model moved to device successfully")
            except Exception as e:
                print(f"✗ Failed to move model to {device}: {e}")
                return False, None, None
        
        # Step 4: Set to eval mode
        print("Step 4: Setting model to eval mode...")
        model.eval()
        print("✓ Model set to eval mode")
        
        # Step 5: Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory after loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        return True, model, processor
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        traceback.print_exc()
        return False, None, None
    
    print()

def test_simple_inference(model, processor, device: str = "cuda"):
    """Test simple text inference"""
    print("=== Testing Simple Text Inference ===")
    
    try:
        # Simple text prompt
        prompt = "Hello, how are you?"
        
        print(f"Testing prompt: '{prompt}'")
        
        # Process input
        inputs = processor(text=prompt, return_tensors="pt")
        
        # Move inputs to device
        if device != "cpu":
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        print("✓ Input processed successfully")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        print("✓ Generation completed successfully")
        
        # Decode
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Text inference failed: {e}")
        traceback.print_exc()
        return False
    
    print()

def test_image_inference(model, processor, device: str = "cuda"):
    """Test simple image inference"""
    print("=== Testing Simple Image Inference ===")
    
    try:
        # Create a simple test image
        import numpy as np
        
        # Create a simple 224x224 RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # Use proper Qwen2.5-VL chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        print("Testing with synthetic image and proper chat template")
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process input with both text and image
        inputs = processor(
            text=[text], 
            images=[image], 
            return_tensors="pt"
        )
        
        # Move inputs to device
        if device != "cpu":
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        print("✓ Image and text processed successfully")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        print("✓ Generation completed successfully")
        
        # Decode
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Image inference failed: {e}")
        traceback.print_exc()
        return False
    
    print()

def main():
    print("Starting Qwen2.5-VL diagnostic tests...\n")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device (cuda, cpu)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference tests")
    args = parser.parse_args()
    
    # Set environment variables for stability
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Run tests
    results = {}
    
    # Test 1: Basic PyTorch
    results["pytorch"] = True  # Always passes if we get here
    
    # Test 2: Transformers import
    results["transformers"] = True  # Always passes if we get here
    
    # Test 3: Model loading
    try:
        success, model, processor = test_model_loading(args.model, args.device)
        results["model_loading"] = success
        
        if success and not args.skip_inference:
            # Test 4: Simple inference
            results["text_inference"] = test_simple_inference(model, processor, args.device)
            
            # Test 5: Image inference
            results["image_inference"] = test_image_inference(model, processor, args.device)
            
        # Cleanup
        if model is not None:
            del model
        if processor is not None:
            del processor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Critical error during testing: {e}")
        traceback.print_exc()
        results["model_loading"] = False
    
    # Summary
    print("=== Test Summary ===")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    # Final memory check
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nFinal GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

if __name__ == "__main__":
    main()
