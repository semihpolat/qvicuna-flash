import os
import time

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    raise

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ Transformers loaded successfully")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")
    raise

try:
    from cog import BasePredictor, Input
    print("✅ Cog loaded successfully")
except ImportError as e:
    print(f"❌ Cog import failed: {e}")
    raise

class Predictor(BasePredictor):
    def setup(self) -> None:
        """qVicuna Flash setup"""
        print("🚀 Loading qVicuna Flash...")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            device = "cuda"
        else:
            print("⚠️  CUDA not available, using CPU")
            device = "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Tokenizer loading failed: {e}")
            raise
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "lmsys/vicuna-7b-v1.5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self.device = device
        print("✅ qVicuna Flash ready!")

    def predict(
        self,
        message: str = Input(description="Your message", default="Hello!"),
        max_tokens: int = Input(description="Max tokens", default=50, ge=1, le=200),
        flash_mode: bool = Input(description="Flash Mode", default=True)
    ) -> dict:
        """Generate response"""
        
        start_time = time.time()
        
        try:
            if flash_mode:
                # Flash mode - optimized for speed
                # Use a simple prompt format
                prompt = f"Human: {message}\nAssistant:"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_tokens, 50),
                        do_sample=False,
                        temperature=0.1,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens (response part)
                input_length = inputs['input_ids'].shape[1]
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                
                # Clean up the response
                if response.startswith("Human:") or response.startswith("Assistant:"):
                    response = response.split(":", 1)[1].strip()
                
            else:
                # Standard mode - better quality
                prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {message}\nASSISTANT:"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                input_length = inputs['input_ids'].shape[1]
                response_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            # Additional cleanup
            if not response or response == message:
                response = "I understand your message. How can I help you?"
            
            # Remove any remaining prompt artifacts
            response = response.replace("Human:", "").replace("Assistant:", "").replace("USER:", "").replace("ASSISTANT:", "").strip()
            
            inference_time = (time.time() - start_time) * 1000
            
            # Simple output - only response and inference time
            return {
                "response": response,
                "inference_time_ms": round(inference_time, 2)
            }
            
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "inference_time_ms": 0
            }