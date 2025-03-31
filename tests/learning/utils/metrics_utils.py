"""
Metrics utilities for learning tests.

This module provides utilities for collecting, analyzing, and exporting metrics
from oscillator-based learning tests, with a focus on performance timing,
memory usage, and quality metrics.
"""

import time
import csv
import os
import functools
import numpy as np
import psutil
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field

# Ensure plots directory exists
os.makedirs('metrics', exist_ok=True)

@dataclass
class ProcessMetrics:
    """Container for metrics collected during processing."""
    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    total_time_ms: float = 0.0
    per_iteration_times_ms: List[float] = field(default_factory=list)
    
    # Convergence metrics
    iterations: int = 0
    converged: bool = False
    convergence_iteration: int = 0
    convergence_time_ms: float = 0.0
    
    # Quality metrics
    mean_coherence: float = 0.0
    max_coherence: float = 0.0
    final_coherence: float = 0.0
    coherence_history: List[float] = field(default_factory=list)
    
    # Pattern metrics
    pattern_count: int = 0
    pattern_distinctiveness: float = 0.0
    
    # Memory metrics
    peak_memory_kb: float = 0.0
    avg_memory_kb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    # Model-specific metrics
    model_specific: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for CSV export."""
        return {
            "total_time_ms": self.total_time_ms,
            "iterations": self.iterations,
            "converged": int(self.converged),
            "convergence_iteration": self.convergence_iteration,
            "convergence_time_ms": self.convergence_time_ms,
            "mean_coherence": self.mean_coherence,
            "max_coherence": self.max_coherence,
            "final_coherence": self.final_coherence,
            "pattern_count": self.pattern_count,
            "pattern_distinctiveness": self.pattern_distinctiveness,
            "peak_memory_kb": self.peak_memory_kb,
            "avg_memory_kb": self.avg_memory_kb,
            **{f"model_{k}": v for k, v in self.model_specific.items()}
        }

class MetricsCollector:
    """
    Collects and manages metrics during model processing.
    
    This class wraps model operators to inject timing and metrics collection
    code at critical points in the processing pipeline.
    """
    
    def __init__(self, 
                 convergence_threshold: float = 1e-4,
                 stability_window: int = 10,
                 track_memory: bool = True):
        """
        Initialize the metrics collector.
        
        Args:
            convergence_threshold: Threshold for detecting convergence
            stability_window: Number of iterations to check for stability
            track_memory: Whether to track memory usage
        """
        self.convergence_threshold = convergence_threshold
        self.stability_window = stability_window
        self.track_memory = track_memory
        self.metrics = ProcessMetrics()
        self.process = None  # Current process for memory tracking
        
        if self.track_memory:
            self.process = psutil.Process()
    
    def start_timing(self):
        """Start timing the process."""
        self.metrics.start_time = time.time()
        
        if self.track_memory:
            self.metrics.memory_samples.append(self._get_memory_usage())
    
    def end_timing(self):
        """End timing the process."""
        self.metrics.end_time = time.time()
        self.metrics.total_time_ms = (self.metrics.end_time - self.metrics.start_time) * 1000
        
        if self.track_memory:
            self.metrics.memory_samples.append(self._get_memory_usage())
            self.metrics.peak_memory_kb = max(self.metrics.memory_samples)
            self.metrics.avg_memory_kb = sum(self.metrics.memory_samples) / len(self.metrics.memory_samples)
    
    def record_iteration(self, iteration: int, delta: Dict[str, Any]):
        """
        Record metrics for a single iteration.
        
        Args:
            iteration: The current iteration number
            delta: The delta dictionary from the model operator
        """
        # Record iteration time
        current_time = time.time()
        if iteration > 0:  # Skip first iteration for per-iteration timing
            iteration_time_ms = (current_time - self._last_iteration_time) * 1000
            self.metrics.per_iteration_times_ms.append(iteration_time_ms)
        self._last_iteration_time = current_time
        
        # Record coherence
        if "mean_coherence" in delta:
            coherence = delta["mean_coherence"]
            self.metrics.coherence_history.append(coherence)
            self.metrics.final_coherence = coherence
            
            if coherence > self.metrics.max_coherence:
                self.metrics.max_coherence = coherence
        
        # Check for convergence
        if not self.metrics.converged and self._check_convergence():
            self.metrics.converged = True
            self.metrics.convergence_iteration = iteration
            self.metrics.convergence_time_ms = (time.time() - self.metrics.start_time) * 1000
        
        # Record memory usage
        if self.track_memory:
            self.metrics.memory_samples.append(self._get_memory_usage())
        
        # Record pattern count if available
        if "num_patterns" in delta:
            self.metrics.pattern_count = delta["num_patterns"]
        
        # Record model-specific metrics
        for key, value in delta.items():
            if key not in ["type", "mean_coherence", "coherence", "num_patterns"]:
                if isinstance(value, (int, float, bool, str)):
                    self.metrics.model_specific[key] = value
    
    def finalize_metrics(self):
        """Finalize metrics after processing is complete."""
        # Calculate mean coherence
        if self.metrics.coherence_history:
            self.metrics.mean_coherence = sum(self.metrics.coherence_history) / len(self.metrics.coherence_history)
        
        # Record total iterations
        self.metrics.iterations = len(self.metrics.coherence_history)
        
        # If never converged, set convergence iteration to max
        if not self.metrics.converged:
            self.metrics.convergence_iteration = self.metrics.iterations
            self.metrics.convergence_time_ms = self.metrics.total_time_ms
    
    def _check_convergence(self) -> bool:
        """
        Check if the system has converged based on coherence history.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.metrics.coherence_history) < self.stability_window:
            return False
        
        # Get the last n values
        recent_values = self.metrics.coherence_history[-self.stability_window:]
        
        # Check if the change is below threshold
        for i in range(1, len(recent_values)):
            if abs(recent_values[i] - recent_values[i-1]) > self.convergence_threshold:
                return False
        
        return True
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in KB.
        
        Returns:
            Memory usage in KB
        """
        if self.process:
            try:
                return self.process.memory_info().rss / 1024  # Convert bytes to KB
            except:
                return 0.0
        return 0.0

class BenchmarkRunner:
    """
    Runs benchmarks on oscillator-based models.
    
    This class manages the execution of standardized tests across character sets
    and collects metrics for analysis and comparison.
    """
    
    def __init__(self, 
                 output_dir: str = "metrics",
                 max_iterations: int = 1000,
                 trials: int = 3):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save metrics files
            max_iterations: Maximum iterations per test
            trials: Number of trials per test for statistical significance
        """
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.trials = trials
        os.makedirs(output_dir, exist_ok=True)
    
    def run_character_benchmark(self, 
                               model_name: str,
                               process_func: Callable,
                               characters: List[str],
                               noise_levels: List[float] = None,
                               occlusion_levels: List[float] = None) -> str:
        """
        Run a benchmark on a set of characters.
        
        Args:
            model_name: Name of the model for reporting
            process_func: Function that processes a character and returns metrics
            characters: List of characters to test
            noise_levels: List of noise levels to test (optional)
            occlusion_levels: List of occlusion levels to test (optional)
            
        Returns:
            Path to the CSV file with results
        """
        results = []
        
        # Process each character
        for char in characters:
            # Run multiple trials
            for trial in range(1, self.trials + 1):
                print(f"Processing character '{char}', trial {trial}...")
                
                # Process clean character
                metrics = process_func(char, noise_level=0.0, occlusion_level=0.0)
                
                # Record results
                result = {
                    "character": char,
                    "trial": trial,
                    "noise_level": 0.0,
                    "occlusion_level": 0.0,
                    **metrics.to_dict()
                }
                results.append(result)
                
                # Process with noise if specified
                if noise_levels:
                    for noise_level in noise_levels:
                        metrics = process_func(char, noise_level=noise_level, occlusion_level=0.0)
                        result = {
                            "character": char,
                            "trial": trial,
                            "noise_level": noise_level,
                            "occlusion_level": 0.0,
                            **metrics.to_dict()
                        }
                        results.append(result)
                
                # Process with occlusion if specified
                if occlusion_levels:
                    for occlusion_level in occlusion_levels:
                        metrics = process_func(char, noise_level=0.0, occlusion_level=occlusion_level)
                        result = {
                            "character": char,
                            "trial": trial,
                            "noise_level": 0.0,
                            "occlusion_level": occlusion_level,
                            **metrics.to_dict()
                        }
                        results.append(result)
        
        # Export results to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(self.output_dir, f"{model_name}_benchmark_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        
        print(f"Benchmark results saved to {csv_path}")
        return csv_path
    
    def run_scaling_benchmark(self,
                             model_name: str,
                             scaling_func: Callable,
                             grid_sizes: List[Tuple[int, int]],
                             oscillator_dims: List[int] = None) -> str:
        """
        Run a benchmark to test scaling properties.
        
        Args:
            model_name: Name of the model for reporting
            scaling_func: Function that processes with different sizes and returns metrics
            grid_sizes: List of grid sizes to test
            oscillator_dims: List of oscillator dimensions to test (optional)
            
        Returns:
            Path to the CSV file with results
        """
        results = []
        
        # Test each grid size
        for grid_size in grid_sizes:
            # Run multiple trials
            for trial in range(1, self.trials + 1):
                print(f"Testing grid size {grid_size}, trial {trial}...")
                
                # Process with default oscillator dimension
                metrics = scaling_func(grid_size=grid_size)
                
                # Record results
                result = {
                    "grid_size_x": grid_size[0],
                    "grid_size_y": grid_size[1],
                    "oscillator_dim": metrics.model_specific.get("oscillator_dim", 0),
                    "trial": trial,
                    **metrics.to_dict()
                }
                results.append(result)
                
                # Process with different oscillator dimensions if specified
                if oscillator_dims:
                    for dim in oscillator_dims:
                        metrics = scaling_func(grid_size=grid_size, oscillator_dim=dim)
                        result = {
                            "grid_size_x": grid_size[0],
                            "grid_size_y": grid_size[1],
                            "oscillator_dim": dim,
                            "trial": trial,
                            **metrics.to_dict()
                        }
                        results.append(result)
        
        # Export results to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(self.output_dir, f"{model_name}_scaling_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        
        print(f"Scaling benchmark results saved to {csv_path}")
        return csv_path

def time_process(func):
    """
    Decorator to time a function execution.
    
    Args:
        func: The function to time
        
    Returns:
        Wrapped function that times execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"{func.__name__} executed in {elapsed_ms:.2f} ms")
        return result
    return wrapper

def track_memory(func):
    """
    Decorator to track memory usage during function execution.
    
    Args:
        func: The function to track
        
    Returns:
        Wrapped function that tracks memory usage
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024  # KB
        result = func(*args, **kwargs)
