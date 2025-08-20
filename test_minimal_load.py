#!/usr/bin/env python3
"""
Minimal test to isolate the segfault issue
"""

import os
import sys

def test_imports():
    """Test basic imports"""
    print("=== Testing Imports ===")
    
    try:
        print("1. Importing torch...")
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   Device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    try:
        print("2. Importing transformers...")
        import transformers
        print(f"   ✓ Transformers {transformers.__version__}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    try:
        print("3. Setting environment variables...")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("   ✓ Environment variables set")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    return True

def test_processor_load():
    """Test loading just the processor"""
    print("\n=== Testing Processor Load ===")
    
    try:
        from transformers import AutoProcessor
        
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading processor for {model_name}...")
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        print("✓ Processor loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Processor loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_load_cpu():
    """Test loading model on CPU first"""
    print("\n=== Testing Model Load (CPU) ===")
    
    try:
        import torch
        from transformers import AutoModelForVision2Seq
        
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading model {model_name} on CPU...")
        
        # Try the simplest possible load
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for stability
            device_map=None,  # No device mapping
            low_cpu_mem_usage=False,  # Don't use low memory mode initially
        )
        
        print("✓ Model loaded on CPU successfully")
        
        # Check model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {param_count:,}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_to_cuda():
    """Test moving model to CUDA"""
    print("\n=== Testing Model to CUDA ===")
    
    try:
        import torch
        from transformers import AutoModelForVision2Seq
        
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping")
            return False
        
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # Clear cache first
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"Loading model {model_name}...")
        
        # Load with bfloat16 to save memory
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        
        print("Model loaded on CPU, now moving to CUDA...")
        
        # Try moving to CUDA
        model = model.to('cuda')
        
        print("✓ Model moved to CUDA successfully")
        
        # Check memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting minimal load test...\n")
    
    results = {}
    
    # Test 1: Basic imports
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\n⚠ Basic imports failed, cannot continue")
        return
    
    # Test 2: Processor load
    results["processor"] = test_processor_load()
    
    # Test 3: Model load on CPU
    results["model_cpu"] = test_model_load_cpu()
    
    # Test 4: Model to CUDA
    if "--skip-cuda" not in sys.argv:
        results["model_cuda"] = test_model_to_cuda()
    
    # Summary
    print("\n=== Test Summary ===")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    main()