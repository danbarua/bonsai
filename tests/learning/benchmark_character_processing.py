"""
Benchmark tests for character processing with oscillator-based models.

This script runs benchmarks on the Hebbian Kuramoto and Predictive Hebbian models
to collect performance metrics and compare their capabilities.
"""

import os
import time
import numpy as np
from typing import Tuple, Dict, Any, List

from models.hebbian import HebbianKuramotoOperator
from models.predictive import PredictiveHebbianOperator
from dynamics.oscillators import LayeredOscillatorState

from tests.learning.utils.character_utils import (
    get_character_matrix,
    add_noise_to_character,
    create_occluded_character,
    create_single_layer_state,
    create_hierarchical_state
)

from tests.learning.utils.metrics_utils import (
    ProcessMetrics,
    MetricsCollector,
    BenchmarkRunner,
    time_process,
    track_memory,
    measure_character_processing,
    compare_models
)

# Ensure metrics directory exists
os.makedirs('metrics', exist_ok=True)

# Define standard test characters
STANDARD_CHARACTERS = ['A', 'B', 'C', '1', '2', '+', '-']

# Define standard test conditions
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]
OCCLUSION_LEVELS = [0.0, 0.1, 0.2, 0.3]
OCCLUSION_TYPES = ['horizontal', 'vertical', 'random']

# Define grid sizes for scaling tests
GRID_SIZES = [(8, 12), (16, 24), (32, 48)]
OSCILLATOR_DIMS = [2, 4, 8]

@time_process
def process_hebbian_character(char: str, noise_level: float = 0.0, 
                             occlusion_level: float = 0.0, occlusion_type: str = 'random',
                             initialize: bool = False, max_iterations: int = 1000) -> Tuple[LayeredOscillatorState, HebbianKuramotoOperator]:
    """
    Process a character with the Hebbian Kuramoto model.
    
    Args:
        char: Character to process
        noise_level: Noise level to apply
        occlusion_level: Occlusion level to apply
        occlusion_type: Type of occlusion to apply
        initialize: Whether to initialize and return state and operator
        max_iterations: Maximum iterations to run
        
    Returns:
        Tuple of (final_state, operator)
    """
    # Get character matrix
    char_matrix = get_character_matrix(char)
    
    # Apply noise if specified
    if noise_level > 0:
        char_matrix = add_noise_to_character(char_matrix, noise_level)
    
    # Apply occlusion if specified
    if occlusion_level > 0:
        char_matrix = create_occluded_character(char, occlusion_type, occlusion_level)
    
    # Create state
    state = create_single_layer_state(char_matrix, perturbation_strength=2.0)
    
    # Initialize operator
    operator = HebbianKuramotoOperator(dt=0.01, mu=0.1, alpha=0.01)
    
    # If only initializing, return state and operator
    if initialize:
        return state, operator
    
    # Process until convergence or max iterations
    collector = MetricsCollector(convergence_threshold=1e-4, stability_window=10)
    collector.start_timing()
    
    for i in range(max_iterations):
        # Apply one iteration
        state = operator.apply(state)
        
        # Record metrics
        collector.record_iteration(i, operator.last_delta)
        
        # Check for convergence
        if collector.metrics.converged:
            print(f"Hebbian converged after {i+1} iterations")
            break
    
    collector.end_timing()
    collector.finalize_metrics()
    
    return state, operator, collector.metrics

@time_process
def process_predictive_character(char: str, noise_level: float = 0.0, 
                                occlusion_level: float = 0.0, occlusion_type: str = 'random',
                                initialize: bool = False, max_iterations: int = 1000) -> Tuple[LayeredOscillatorState, PredictiveHebbianOperator]:
    """
    Process a character with the Predictive Hebbian model.
    
    Args:
        char: Character to process
        noise_level: Noise level to apply
        occlusion_level: Occlusion level to apply
        occlusion_type: Type of occlusion to apply
        initialize: Whether to initialize and return state and operator
        max_iterations: Maximum iterations to run
        
    Returns:
        Tuple of (final_state, operator)
    """
    # Get character matrix
    char_matrix = get_character_matrix(char)
    
    # Apply noise if specified
    if noise_level > 0:
        char_matrix = add_noise_to_character(char_matrix, noise_level)
    
    # Apply occlusion if specified
    if occlusion_level > 0:
        char_matrix = create_occluded_character(char, occlusion_type, occlusion_level)
    
    # Create hierarchical state
    state = create_hierarchical_state(char_matrix, perturbation_strength=2.0)
    
    # Initialize operator
    operator = PredictiveHebbianOperator(
        dt=0.01,
        pc_learning_rate=0.05,
        hebb_learning_rate=0.05,
        pc_error_scaling=0.5,
        pc_precision=1.0,
        hebb_decay_rate=0.1
    )
    
    # If only initializing, return state and operator
    if initialize:
        return state, operator
    
    # Process until convergence or max iterations
    collector = MetricsCollector(convergence_threshold=1e-4, stability_window=10)
    collector.start_timing()
    
    for i in range(max_iterations):
        # Apply one iteration
        state = operator.apply(state)
        
        # Record metrics
        collector.record_iteration(i, operator.last_delta)
        
        # Check for convergence
        if collector.metrics.converged:
            print(f"Predictive converged after {i+1} iterations")
            break
    
    collector.end_timing()
    collector.finalize_metrics()
    
    return state, operator, collector.metrics

def measure_hebbian_processing(char: str, noise_level: float = 0.0, occlusion_level: float = 0.0) -> ProcessMetrics:
    """
    Measure metrics for Hebbian character processing.
    
    Args:
        char: Character to process
        noise_level: Noise level to apply
        occlusion_level: Occlusion level to apply
        
    Returns:
        ProcessMetrics object with collected metrics
    """
    # Process character and get metrics
    _, _, metrics = process_hebbian_character(
        char, 
        noise_level=noise_level, 
        occlusion_level=occlusion_level,
        max_iterations=1000
    )
    
    return metrics

def measure_predictive_processing(char: str, noise_level: float = 0.0, occlusion_level: float = 0.0) -> ProcessMetrics:
    """
    Measure metrics for Predictive character processing.
    
    Args:
        char: Character to process
        noise_level: Noise level to apply
        occlusion_level: Occlusion level to apply
        
    Returns:
        ProcessMetrics object with collected metrics
    """
    # Process character and get metrics
    _, _, metrics = process_predictive_character(
        char, 
        noise_level=noise_level, 
        occlusion_level=occlusion_level,
        max_iterations=1000
    )
    
    return metrics

def measure_hebbian_scaling(grid_size: Tuple[int, int], oscillator_dim: int = None) -> ProcessMetrics:
    """
    Measure scaling metrics for Hebbian model.
    
    Args:
        grid_size: Size of the grid
        oscillator_dim: Dimensionality of oscillators (not used for Hebbian)
        
    Returns:
        ProcessMetrics object with collected metrics
    """
    # Create a random character matrix of the specified size
    char_matrix = np.random.rand(*grid_size) > 0.5
    
    # Create state
    state = create_single_layer_state(char_matrix, perturbation_strength=2.0)
    
    # Initialize operator
    operator = HebbianKuramotoOperator(dt=0.01, mu=0.1, alpha=0.01)
    
    # Process for a fixed number of iterations
    collector = MetricsCollector(track_memory=True)
    collector.start_timing()
    
    for i in range(100):  # Fixed number of iterations for scaling tests
        # Apply one iteration
        state = operator.apply(state)
        
        # Record metrics
        collector.record_iteration(i, operator.last_delta)
    
    collector.end_timing()
    collector.finalize_metrics()
    
    # Add grid size to model-specific metrics
    collector.metrics.model_specific["grid_size_x"] = grid_size[0]
    collector.metrics.model_specific["grid_size_y"] = grid_size[1]
    collector.metrics.model_specific["oscillator_dim"] = 1  # Hebbian uses 1D oscillators
    
    return collector.metrics

def measure_predictive_scaling(grid_size: Tuple[int, int], oscillator_dim: int = 4) -> ProcessMetrics:
    """
    Measure scaling metrics for Predictive model.
    
    Args:
        grid_size: Size of the grid
        oscillator_dim: Dimensionality of oscillators
        
    Returns:
        ProcessMetrics object with collected metrics
    """
    # Create a random character matrix of the specified size
    char_matrix = np.random.rand(*grid_size) > 0.5
    
    # Create hierarchical state with specified oscillator dimension
    # For simplicity, we'll use a 2-layer hierarchy with the second layer half the size
    layer_shapes = [
        grid_size,
        (grid_size[0]//2, grid_size[1]//2)
    ]
    
    # Create state
    state = create_hierarchical_state(char_matrix, layer_shapes=layer_shapes, perturbation_strength=2.0)
    
    # Initialize operator
    operator = PredictiveHebbianOperator(
        dt=0.01,
        pc_learning_rate=0.05,
        hebb_learning_rate=0.05,
        pc_error_scaling=0.5,
        pc_precision=1.0,
        hebb_decay_rate=0.1
    )
    
    # Process for a fixed number of iterations
    collector = MetricsCollector(track_memory=True)
    collector.start_timing()
    
    for i in range(100):  # Fixed number of iterations for scaling tests
        # Apply one iteration
        state = operator.apply(state)
        
        # Record metrics
        collector.record_iteration(i, operator.last_delta)
    
    collector.end_timing()
    collector.finalize_metrics()
    
    # Add grid size and oscillator dimension to model-specific metrics
    collector.metrics.model_specific["grid_size_x"] = grid_size[0]
    collector.metrics.model_specific["grid_size_y"] = grid_size[1]
    collector.metrics.model_specific["oscillator_dim"] = oscillator_dim
    
    return collector.metrics

def run_character_benchmarks(characters: List[str] = None, trials: int = 3):
    """
    Run character processing benchmarks for both models.
    
    Args:
        characters: List of characters to test
        trials: Number of trials per test
    """
    if characters is None:
        characters = STANDARD_CHARACTERS
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        output_dir="metrics",
        max_iterations=1000,
        trials=trials
    )
    
    # Run Hebbian benchmark
    print("\n=== Running Hebbian Kuramoto Benchmark ===")
    hebbian_csv = runner.run_character_benchmark(
        model_name="hebbian_kuramoto",
        process_func=measure_hebbian_processing,
        characters=characters,
        noise_levels=[0.0, 0.1, 0.2],
        occlusion_levels=[0.0, 0.1, 0.2]
    )
    
    # Run Predictive benchmark
    print("\n=== Running Predictive Hebbian Benchmark ===")
    predictive_csv = runner.run_character_benchmark(
        model_name="predictive_hebbian",
        process_func=measure_predictive_processing,
        characters=characters,
        noise_levels=[0.0, 0.1, 0.2],
        occlusion_levels=[0.0, 0.1, 0.2]
    )
    
    # Compare models
    print("\n=== Comparing Models ===")
    comparison_csv = compare_models(
        model_names=["hebbian_kuramoto", "predictive_hebbian"],
        csv_paths=[hebbian_csv, predictive_csv],
        output_dir="metrics"
    )
    
    print(f"\nBenchmark results saved to:")
    print(f"  Hebbian: {hebbian_csv}")
    print(f"  Predictive: {predictive_csv}")
    print(f"  Comparison: {comparison_csv}")

def run_scaling_benchmarks(grid_sizes: List[Tuple[int, int]] = None, oscillator_dims: List[int] = None, trials: int = 3):
    """
    Run scaling benchmarks for both models.
    
    Args:
        grid_sizes: List of grid sizes to test
        oscillator_dims: List of oscillator dimensions to test
        trials: Number of trials per test
    """
    if grid_sizes is None:
        grid_sizes = GRID_SIZES
    
    if oscillator_dims is None:
        oscillator_dims = OSCILLATOR_DIMS
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        output_dir="metrics",
        max_iterations=100,  # Fewer iterations for scaling tests
        trials=trials
    )
    
    # Run Hebbian scaling benchmark
    print("\n=== Running Hebbian Kuramoto Scaling Benchmark ===")
    hebbian_csv = runner.run_scaling_benchmark(
        model_name="hebbian_kuramoto",
        scaling_func=measure_hebbian_scaling,
        grid_sizes=grid_sizes
    )
    
    # Run Predictive scaling benchmark
    print("\n=== Running Predictive Hebbian Scaling Benchmark ===")
    predictive_csv = runner.run_scaling_benchmark(
        model_name="predictive_hebbian",
        scaling_func=measure_predictive_scaling,
        grid_sizes=grid_sizes,
        oscillator_dims=oscillator_dims
    )
    
    print(f"\nScaling benchmark results saved to:")
    print(f"  Hebbian: {hebbian_csv}")
    print(f"  Predictive: {predictive_csv}")

def run_single_character_test(char: str = 'A', noise_level: float = 0.0, occlusion_level: float = 0.0):
    """
    Run a single character test for both models and print detailed metrics.
    
    Args:
        char: Character to test
        noise_level: Noise level to apply
        occlusion_level: Occlusion level to apply
    """
    print(f"\n=== Testing Character '{char}' (Noise: {noise_level}, Occlusion: {occlusion_level}) ===")
    
    # Process with Hebbian model
    print("\nHebbian Kuramoto Model:")
    _, _, hebbian_metrics = process_hebbian_character(
        char, 
        noise_level=noise_level, 
        occlusion_level=occlusion_level,
        max_iterations=1000
    )
    
    # Process with Predictive model
    print("\nPredictive Hebbian Model:")
    _, _, predictive_metrics = process_predictive_character(
        char, 
        noise_level=noise_level, 
        occlusion_level=occlusion_level,
        max_iterations=1000
    )
    
    # Print comparison
    print("\n=== Metrics Comparison ===")
    print(f"{'Metric':<25} {'Hebbian':<15} {'Predictive':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15}")
    print(f"{'Total Time (ms)':<25} {hebbian_metrics.total_time_ms:<15.2f} {predictive_metrics.total_time_ms:<15.2f}")
    print(f"{'Iterations':<25} {hebbian_metrics.iterations:<15d} {predictive_metrics.iterations:<15d}")
    print(f"{'Converged':<25} {hebbian_metrics.converged:<15} {predictive_metrics.converged:<15}")
    print(f"{'Convergence Iteration':<25} {hebbian_metrics.convergence_iteration:<15d} {predictive_metrics.convergence_iteration:<15d}")
    print(f"{'Convergence Time (ms)':<25} {hebbian_metrics.convergence_time_ms:<15.2f} {predictive_metrics.convergence_time_ms:<15.2f}")
    print(f"{'Mean Coherence':<25} {hebbian_metrics.mean_coherence:<15.4f} {predictive_metrics.mean_coherence:<15.4f}")
    print(f"{'Max Coherence':<25} {hebbian_metrics.max_coherence:<15.4f} {predictive_metrics.max_coherence:<15.4f}")
    print(f"{'Final Coherence':<25} {hebbian_metrics.final_coherence:<15.4f} {predictive_metrics.final_coherence:<15.4f}")
    print(f"{'Peak Memory (KB)':<25} {hebbian_metrics.peak_memory_kb:<15.2f} {predictive_metrics.peak_memory_kb:<15.2f}")
    print(f"{'Avg Memory (KB)':<25} {hebbian_metrics.avg_memory_kb:<15.2f} {predictive_metrics.avg_memory_kb:<15.2f}")
    
    # Calculate performance ratio
    time_ratio = predictive_metrics.total_time_ms / hebbian_metrics.total_time_ms if hebbian_metrics.total_time_ms > 0 else float('inf')
    memory_ratio = predictive_metrics.peak_memory_kb / hebbian_metrics.peak_memory_kb if hebbian_metrics.peak_memory_kb > 0 else float('inf')
    
    print(f"\nPerformance Ratios (Predictive / Hebbian):")
    print(f"Time Ratio: {time_ratio:.2f}x")
    print(f"Memory Ratio: {memory_ratio:.2f}x")

if __name__ == "__main__":
    # Run a single character test for quick demonstration
    run_single_character_test(char='A')
    
    # Uncomment to run full benchmarks
    # run_character_benchmarks(characters=['A', 'B', 'C'], trials=1)
    # run_scaling_benchmarks(grid_sizes=[(8, 12), (16, 24)], oscillator_dims=[2, 4], trials=1)
