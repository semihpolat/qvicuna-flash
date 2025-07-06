#!/usr/bin/env python3
"""
qVicuna Flash Local Test
Test the model locally before deploying
"""

import torch
from predict import Predictor
import time

def test_qvicuna_flash():
    """Test qVicuna Flash locally"""
    print("🧪 Testing qVicuna Flash...")
    
    # Initialize predictor
    predictor = Predictor()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This model requires GPU.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    # Setup model
    print("⏳ Setting up model...")
    predictor.setup()
    
    # Test messages
    test_messages = [
        "Merhaba!",
        "Python nedir?",
        "Bugün hava nasıl?",
        "1+1 kaç eder?",
        "Türkiye'nin başkenti neresi?"
    ]
    
    print("\n" + "="*50)
    print("🚀 FLASH MODE TESTS")
    print("="*50)
    
    for message in test_messages:
        print(f"\n💬 Input: {message}")
        
        start_time = time.time()
        result = predictor.predict(
            message=message,
            max_tokens=50,
            flash_mode=True
        )
        total_time = time.time() - start_time
        
        print(f"⚡ Response: {result['response']}")
        print(f"⏱️  Time: {result['inference_time_ms']}ms (Total: {total_time*1000:.2f}ms)")
        print(f"🎯 Mode: {result['mode']}")
        
        if result['response'] == message:
            print("⚠️  WARNING: Response equals input!")
    
    print("\n" + "="*50)
    print("🎯 STANDARD MODE TESTS")
    print("="*50)
    
    # Test one message in standard mode
    message = "Python programlama dili hakkında kısa bilgi ver"
    print(f"\n💬 Input: {message}")
    
    start_time = time.time()
    result = predictor.predict(
        message=message,
        max_tokens=100,
        flash_mode=False
    )
    total_time = time.time() - start_time
    
    print(f"🎯 Response: {result['response']}")
    print(f"⏱️  Time: {result['inference_time_ms']}ms (Total: {total_time*1000:.2f}ms)")
    print(f"🎯 Mode: {result['mode']}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_qvicuna_flash() 