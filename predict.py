import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cog import BasePredictor, Input
import time

class Predictor(BasePredictor):
    def setup(self) -> None:
        """qVicuna Flash setup"""
        print("🚀 Loading qVicuna Flash...")
        
        self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.model = AutoModelForCausalLM.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        print("✅ qVicuna Flash ready!")

    def predict(
        self,
        message: str = Input(description="Your message", default="Hello!"),
        max_tokens: int = Input(description="Max tokens", default=50, ge=1, le=200),
        flash_mode: bool = Input(description="Flash Mode", default=True)
    ) -> dict:
        """Generate response"""
        
        start_time = time.time()
        
        if flash_mode:
            # Flash mode - minimal prompt
            inputs = self.tokenizer(message, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 50),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
        else:
            # Standard mode
            prompt = f"USER: {message}\nASSISTANT:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("ASSISTANT:")[-1].strip()
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "response": response,
            "inference_time_ms": round(inference_time, 2),
            "mode": "⚡ Flash" if flash_mode else "🎯 Standard",
            "model": "qVicuna Flash 7B"
        }