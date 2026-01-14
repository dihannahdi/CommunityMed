"""
CommunityMed AI Benchmark Suite
Validates model performance for MedGemma Impact Challenge

This script measures:
1. Model loading time and memory usage
2. Inference latency (P50, P95, P99)
3. Accuracy on TB detection benchmarks
4. Edge deployment metrics

Run: python scripts/benchmark.py --model medgemma-1.5-4b-it
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import statistics

import torch
import numpy as np
from PIL import Image
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.models.medgemma_loader import MedGemmaLoader, get_model_info
    from src.models.hear_loader import MockHeARLoader
    from src.models.medsiglip_loader import MockMedSigLIPLoader
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.info("Run from project root: python scripts/benchmark.py")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_name: str
    timestamp: str
    device: str
    gpu_name: Optional[str]
    gpu_memory_total_gb: Optional[float]
    
    # Loading metrics
    load_time_seconds: float
    model_memory_gb: float
    
    # Inference metrics
    inference_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    throughput_samples_per_sec: float
    
    # Accuracy metrics (if test data available)
    accuracy: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Edge deployment
    quantization: str = "none"
    model_size_gb: Optional[float] = None


class MedGemmaBenchmark:
    """Benchmark suite for MedGemma models"""
    
    def __init__(self, model_name: str = "medgemma-4b-it", use_quantization: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.model = None
        self.processor = None
        self.loader = None
        
    def setup(self) -> float:
        """
        Load model and return load time
        
        Returns:
            Load time in seconds
        """
        logger.info(f"Loading {self.model_name}...")
        start_time = time.time()
        
        self.loader = MedGemmaLoader()
        
        # Check model type
        if self.model_name in self.loader.MULTIMODAL_MODELS:
            self.model, self.processor = self.loader.load_multimodal_model(
                model_name=self.model_name,
                use_quantization=self.use_quantization,
            )
        else:
            self.model, self.processor = self.loader.load_text_model(
                model_name=self.model_name,
                use_quantization=self.use_quantization,
            )
        
        load_time = time.time() - start_time
        logger.success(f"Model loaded in {load_time:.2f}s")
        
        return load_time
    
    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage in GB
        
        Returns:
            Memory usage in GB
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def run_inference_benchmark(
        self,
        num_samples: int = 50,
        warmup: int = 5,
    ) -> List[float]:
        """
        Run inference benchmark with synthetic data
        
        Args:
            num_samples: Number of inference samples
            warmup: Warmup iterations
            
        Returns:
            List of latencies in milliseconds
        """
        if self.model is None:
            raise RuntimeError("Call setup() first")
        
        # Create synthetic test image (chest X-ray-like)
        test_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        test_prompt = "Analyze this chest X-ray for signs of tuberculosis. Describe any abnormalities."
        
        latencies = []
        
        # Warmup
        logger.info(f"Warming up with {warmup} iterations...")
        for _ in range(warmup):
            _ = self.loader.generate_with_image(
                self.model,
                self.processor,
                test_image,
                test_prompt,
                max_new_tokens=64,
            )
        
        # Benchmark
        logger.info(f"Running {num_samples} inference samples...")
        for i in range(num_samples):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            _ = self.loader.generate_with_image(
                self.model,
                self.processor,
                test_image,
                test_prompt,
                max_new_tokens=64,
            )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{num_samples} completed (last: {latency_ms:.1f}ms)")
        
        return latencies
    
    def calculate_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics
        
        Args:
            latencies: List of latencies in ms
            
        Returns:
            Dictionary of statistics
        """
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
            "mean": statistics.mean(latencies),
            "std": statistics.stdev(latencies) if n > 1 else 0,
            "min": min(latencies),
            "max": max(latencies),
        }
    
    def run_full_benchmark(self, num_samples: int = 50) -> BenchmarkResult:
        """
        Run complete benchmark suite
        
        Args:
            num_samples: Number of inference samples
            
        Returns:
            BenchmarkResult with all metrics
        """
        # Device info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
        
        # Load model
        load_time = self.setup()
        model_memory = self.get_memory_usage()
        
        # Run inference benchmark
        latencies = self.run_inference_benchmark(num_samples=num_samples)
        stats = self.calculate_statistics(latencies)
        
        # Calculate throughput
        total_time_sec = sum(latencies) / 1000
        throughput = num_samples / total_time_sec
        
        result = BenchmarkResult(
            model_name=self.model_name,
            timestamp=datetime.now().isoformat(),
            device=device,
            gpu_name=gpu_name,
            gpu_memory_total_gb=gpu_memory,
            load_time_seconds=load_time,
            model_memory_gb=model_memory,
            inference_count=num_samples,
            latency_p50_ms=stats["p50"],
            latency_p95_ms=stats["p95"],
            latency_p99_ms=stats["p99"],
            latency_mean_ms=stats["mean"],
            throughput_samples_per_sec=throughput,
            quantization="4-bit NF4" if self.use_quantization else "none",
        )
        
        return result


class HeARBenchmark:
    """Benchmark suite for HeAR audio model"""
    
    def __init__(self):
        self.loader = MockHeARLoader()
        
    def run_benchmark(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark HeAR embeddings extraction
        
        Args:
            num_samples: Number of samples
            
        Returns:
            Benchmark results
        """
        # Generate synthetic audio (4 seconds at 16kHz)
        audio = np.random.randn(64000).astype(np.float32)
        
        latencies = []
        
        for _ in range(num_samples):
            start = time.perf_counter()
            _ = self.loader.extract_embeddings(audio)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        return {
            "model": "HeAR (mock)",
            "samples": num_samples,
            "latency_mean_ms": statistics.mean(latencies),
            "latency_p95_ms": sorted(latencies)[int(num_samples * 0.95)],
            "embedding_dim": 768,
        }


class MedSigLIPBenchmark:
    """Benchmark suite for MedSigLIP image model"""
    
    def __init__(self):
        self.loader = MockMedSigLIPLoader()
        
    def run_benchmark(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark MedSigLIP embeddings extraction
        
        Args:
            num_samples: Number of samples
            
        Returns:
            Benchmark results
        """
        # Generate synthetic image
        image = Image.new("RGB", (384, 384), color=(128, 128, 128))
        
        latencies = []
        
        for _ in range(num_samples):
            start = time.perf_counter()
            _ = self.loader.extract_embeddings(image)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        return {
            "model": "MedSigLIP (mock)",
            "samples": num_samples,
            "latency_mean_ms": statistics.mean(latencies),
            "latency_p95_ms": sorted(latencies)[int(num_samples * 0.95)],
            "embedding_dim": 1152,
        }


def print_results(result: BenchmarkResult):
    """Pretty print benchmark results"""
    print("\n" + "="*60)
    print("🏥 CommunityMed AI Benchmark Results")
    print("="*60)
    print(f"\n📊 Model: {result.model_name}")
    print(f"⏰ Timestamp: {result.timestamp}")
    print(f"💻 Device: {result.device}")
    if result.gpu_name:
        print(f"🎮 GPU: {result.gpu_name} ({result.gpu_memory_total_gb:.1f} GB)")
    
    print(f"\n📦 Loading:")
    print(f"  • Load time: {result.load_time_seconds:.2f}s")
    print(f"  • Model memory: {result.model_memory_gb:.2f} GB")
    print(f"  • Quantization: {result.quantization}")
    
    print(f"\n⚡ Inference ({result.inference_count} samples):")
    print(f"  • P50 latency: {result.latency_p50_ms:.1f} ms")
    print(f"  • P95 latency: {result.latency_p95_ms:.1f} ms")
    print(f"  • P99 latency: {result.latency_p99_ms:.1f} ms")
    print(f"  • Mean latency: {result.latency_mean_ms:.1f} ms")
    print(f"  • Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
    
    if result.accuracy:
        print(f"\n🎯 Accuracy:")
        print(f"  • Accuracy: {result.accuracy:.1%}")
        print(f"  • Sensitivity: {result.sensitivity:.1%}")
        print(f"  • Specificity: {result.specificity:.1%}")
        print(f"  • F1 Score: {result.f1_score:.3f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="CommunityMed AI Benchmark Suite")
    parser.add_argument(
        "--model",
        type=str,
        default="medgemma-4b-it",
        choices=["medgemma-4b-it", "medgemma-1.5-4b-it", "medgemma-27b-text-it", "hear", "medsiglip", "all"],
        help="Model to benchmark"
    )
    parser.add_argument("--samples", type=int, default=50, help="Number of inference samples")
    parser.add_argument("--no-quantization", action="store_true", help="Disable quantization")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    results = {}
    
    if args.model in ["medgemma-4b-it", "medgemma-1.5-4b-it", "medgemma-27b-text-it"]:
        benchmark = MedGemmaBenchmark(
            model_name=args.model,
            use_quantization=not args.no_quantization,
        )
        result = benchmark.run_full_benchmark(num_samples=args.samples)
        print_results(result)
        results[args.model] = asdict(result)
        
    elif args.model == "hear":
        benchmark = HeARBenchmark()
        result = benchmark.run_benchmark(num_samples=args.samples)
        print(f"\n🎤 HeAR Benchmark Results:")
        print(json.dumps(result, indent=2))
        results["hear"] = result
        
    elif args.model == "medsiglip":
        benchmark = MedSigLIPBenchmark()
        result = benchmark.run_benchmark(num_samples=args.samples)
        print(f"\n🖼️ MedSigLIP Benchmark Results:")
        print(json.dumps(result, indent=2))
        results["medsiglip"] = result
        
    elif args.model == "all":
        # Benchmark all HAI-DEF models
        print("🚀 Running full HAI-DEF benchmark suite...\n")
        
        # HeAR
        hear_bench = HeARBenchmark()
        results["hear"] = hear_bench.run_benchmark()
        
        # MedSigLIP
        siglip_bench = MedSigLIPBenchmark()
        results["medsiglip"] = siglip_bench.run_benchmark()
        
        # MedGemma (only if GPU available)
        if torch.cuda.is_available():
            for model in ["medgemma-4b-it"]:
                try:
                    benchmark = MedGemmaBenchmark(model_name=model)
                    result = benchmark.run_full_benchmark(num_samples=min(args.samples, 20))
                    print_results(result)
                    results[model] = asdict(result)
                except Exception as e:
                    logger.error(f"Failed to benchmark {model}: {e}")
        else:
            logger.warning("GPU not available, skipping MedGemma benchmarks")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.success(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
