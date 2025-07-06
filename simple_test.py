#!/usr/bin/env python3
"""
Simple test without heavy dependencies
"""

def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Testing basic functionality...")
    
    # Test imports
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available")
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ Transformers imported successfully")
        print(f"   Version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    # Test basic model info
    try:
        from transformers import AutoTokenizer
        print("⏳ Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Smaller model for test
        
        test_text = "Hello world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        decoded = tokenizer.decode(tokens['input_ids'][0])
        
        print(f"✅ Tokenizer test passed")
        print(f"   Input: {test_text}")
        print(f"   Tokens: {tokens['input_ids'].tolist()}")
        print(f"   Decoded: {decoded}")
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1) 