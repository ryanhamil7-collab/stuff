import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os
from src.utils import log

class ModelCompressor:
    """
    Model compression using pruning and quantization.
    
    Target: <8GB VRAM on T4, <1.5s inference
    Techniques:
    - Magnitude pruning (40% sparsity)
    - 4-bit GPTQ quantization
    - Post-training optimization
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        log.info(f"ModelCompressor initialized for {model_name}")
    
    def apply_magnitude_pruning(self, model: nn.Module, sparsity: float = 0.4) -> nn.Module:
        """
        Apply magnitude-based pruning to model.
        
        Args:
            model: PyTorch model
            sparsity: Target sparsity (0.4 = 40% of weights pruned)
        
        Returns:
            Pruned model
        """
        log.info(f"Applying magnitude pruning with {sparsity*100}% sparsity")
        
        try:
            import torch.nn.utils.prune as prune
            
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, 'weight'))
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
            
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            log.info(f"Pruning complete: removed {sparsity*100}% of weights")
            
            return model
            
        except Exception as e:
            log.error(f"Pruning failed: {str(e)}")
            return model
    
    def apply_4bit_quantization(self, model_name: str, output_dir: str) -> Optional[str]:
        """
        Apply 4-bit GPTQ quantization.
        
        Args:
            model_name: Model name or path
            output_dir: Output directory for quantized model
        
        Returns:
            Path to quantized model or None
        """
        log.info("Applying 4-bit GPTQ quantization")
        
        try:
            from transformers import GPTQConfig, AutoModelForCausalLM
            
            quantization_config = GPTQConfig(
                bits=4,
                dataset="c4",
                tokenizer=self.tokenizer,
                group_size=128,
                desc_act=False
            )
            
            log.info("Loading model for quantization (this may take several minutes)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
            
            log.info(f"Quantized model saved to {output_dir}")
            
            return output_dir
            
        except ImportError:
            log.error("auto-gptq not installed. Install with: pip install auto-gptq")
            return None
        except Exception as e:
            log.error(f"Quantization failed: {str(e)}")
            return None
    
    def compress_model(
        self,
        model_path: str,
        output_dir: str,
        prune: bool = True,
        quantize: bool = True,
        sparsity: float = 0.4
    ) -> Optional[str]:
        """
        Full compression pipeline.
        
        Args:
            model_path: Path to model
            output_dir: Output directory
            prune: Whether to apply pruning
            quantize: Whether to apply quantization
            sparsity: Pruning sparsity
        
        Returns:
            Path to compressed model
        """
        log.info(f"Starting compression pipeline for {model_path}")
        log.info(f"Prune: {prune}, Quantize: {quantize}, Sparsity: {sparsity}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if quantize:
                compressed_path = self.apply_4bit_quantization(model_path, output_dir)
                if compressed_path:
                    return compressed_path
            
            if prune:
                log.info("Loading model for pruning...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
                model = self.apply_magnitude_pruning(model, sparsity)
                
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                
                log.info(f"Pruned model saved to {output_dir}")
                return output_dir
            
            return None
            
        except Exception as e:
            log.error(f"Compression pipeline failed: {str(e)}")
            return None
    
    def benchmark_model(self, model_path: str, num_samples: int = 10) -> dict:
        """
        Benchmark model inference speed and memory usage.
        
        Args:
            model_path: Path to model
            num_samples: Number of samples to test
        
        Returns:
            Benchmark results
        """
        log.info(f"Benchmarking model: {model_path}")
        
        try:
            import time
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            test_prompt = "Analyze the following stock: AAPL. Current price: $150. RSI: 65. MACD: 0.5."
            
            times = []
            for _ in range(num_samples):
                inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
                memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
            else:
                memory_allocated = 0
                memory_reserved = 0
            
            results = {
                'avg_inference_time': avg_time,
                'min_inference_time': min(times),
                'max_inference_time': max(times),
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'target_met': avg_time < 1.5 and memory_allocated < 8.0
            }
            
            log.info(f"Benchmark results:")
            log.info(f"  Avg inference time: {avg_time:.3f}s (target: <1.5s)")
            log.info(f"  Memory allocated: {memory_allocated:.2f}GB (target: <8GB)")
            log.info(f"  Target met: {results['target_met']}")
            
            return results
            
        except Exception as e:
            log.error(f"Benchmarking failed: {str(e)}")
            return {}

def compress_llm_cli(model_name: str, output_dir: str, prune: bool = True, quantize_4bit: bool = True):
    """
    CLI function for model compression.
    
    Usage:
        python -m src.models.model_compression --model mistralai/Mistral-7B-Instruct-v0.2 --output models/compressed --prune --quantize-4bit
    
    Args:
        model_name: Model name or path
        output_dir: Output directory
        prune: Apply pruning
        quantize_4bit: Apply 4-bit quantization
    """
    compressor = ModelCompressor(model_name)
    
    compressed_path = compressor.compress_model(
        model_name,
        output_dir,
        prune=prune,
        quantize=quantize_4bit
    )
    
    if compressed_path:
        log.info(f"Compression successful: {compressed_path}")
        
        log.info("Running benchmark...")
        results = compressor.benchmark_model(compressed_path)
        
        if results.get('target_met'):
            log.info("✓ Model meets performance targets (<8GB VRAM, <1.5s inference)")
        else:
            log.warning("⚠ Model does not meet performance targets")
    else:
        log.error("Compression failed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compress LLM models")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prune", action="store_true", help="Apply pruning")
    parser.add_argument("--quantize-4bit", action="store_true", help="Apply 4-bit quantization")
    parser.add_argument("--sparsity", type=float, default=0.4, help="Pruning sparsity (default: 0.4)")
    
    args = parser.parse_args()
    
    compress_llm_cli(args.model, args.output, args.prune, args.quantize_4bit)
