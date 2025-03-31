"""
Analyze benchmark results for oscillator-based models.

This script analyzes the benchmark results from the CSV files and generates
a summary report with key findings.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Ensure metrics directory exists
os.makedirs('metrics/reports', exist_ok=True)

def load_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries with the CSV data
    """
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numeric values
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except (ValueError, TypeError):
                    pass
            data.append(row)
    return data

def analyze_character_benchmarks(hebbian_csv: str, predictive_csv: str) -> Dict[str, Any]:
    """
    Analyze character benchmark results.
    
    Args:
        hebbian_csv: Path to the Hebbian benchmark CSV
        predictive_csv: Path to the Predictive benchmark CSV
        
    Returns:
        Dictionary with analysis results
    """
    hebbian_data = load_csv(hebbian_csv)
    predictive_data = load_csv(predictive_csv)
    
    # Group by character and condition
    hebbian_by_char = {}
    predictive_by_char = {}
    
    for row in hebbian_data:
        key = (row['character'], row['noise_level'], row['occlusion_level'])
        if key not in hebbian_by_char:
            hebbian_by_char[key] = []
        hebbian_by_char[key].append(row)
    
    for row in predictive_data:
        key = (row['character'], row['noise_level'], row['occlusion_level'])
        if key not in predictive_by_char:
            predictive_by_char[key] = []
        predictive_by_char[key].append(row)
    
    # Compute average metrics for each group
    results = {
        'clean_characters': {
            'hebbian': {'time_ms': [], 'coherence': [], 'memory_kb': []},
            'predictive': {'time_ms': [], 'coherence': [], 'memory_kb': []}
        },
        'noisy_characters': {
            'hebbian': {'time_ms': [], 'coherence': [], 'memory_kb': []},
            'predictive': {'time_ms': [], 'coherence': [], 'memory_kb': []}
        },
        'occluded_characters': {
            'hebbian': {'time_ms': [], 'coherence': [], 'memory_kb': []},
            'predictive': {'time_ms': [], 'coherence': [], 'memory_kb': []}
        },
        'performance_ratios': {
            'time_ratio': [],
            'memory_ratio': [],
            'coherence_ratio': []
        }
    }
    
    # Process each character and condition
    for key in set(hebbian_by_char.keys()) | set(predictive_by_char.keys()):
        char, noise, occlusion = key
        
        # Skip if either model doesn't have data for this condition
        if key not in hebbian_by_char or key not in predictive_by_char:
            continue
        
        # Get average metrics for each model
        hebbian_time = np.mean([row['total_time_ms'] for row in hebbian_by_char[key]])
        hebbian_coherence = np.mean([row['final_coherence'] for row in hebbian_by_char[key]])
        hebbian_memory = np.mean([row['peak_memory_kb'] for row in hebbian_by_char[key]])
        
        predictive_time = np.mean([row['total_time_ms'] for row in predictive_by_char[key]])
        predictive_coherence = np.mean([row['final_coherence'] for row in predictive_by_char[key]])
        predictive_memory = np.mean([row['peak_memory_kb'] for row in predictive_by_char[key]])
        
        # Calculate performance ratios
        time_ratio = predictive_time / hebbian_time if hebbian_time > 0 else float('inf')
        memory_ratio = predictive_memory / hebbian_memory if hebbian_memory > 0 else float('inf')
        coherence_ratio = predictive_coherence / hebbian_coherence if hebbian_coherence > 0 else float('inf')
        
        results['performance_ratios']['time_ratio'].append(time_ratio)
        results['performance_ratios']['memory_ratio'].append(memory_ratio)
        results['performance_ratios']['coherence_ratio'].append(coherence_ratio)
        
        # Categorize by condition
        if noise == 0.0 and occlusion == 0.0:
            # Clean character
            results['clean_characters']['hebbian']['time_ms'].append(hebbian_time)
            results['clean_characters']['hebbian']['coherence'].append(hebbian_coherence)
            results['clean_characters']['hebbian']['memory_kb'].append(hebbian_memory)
            
            results['clean_characters']['predictive']['time_ms'].append(predictive_time)
            results['clean_characters']['predictive']['coherence'].append(predictive_coherence)
            results['clean_characters']['predictive']['memory_kb'].append(predictive_memory)
        elif noise > 0.0:
            # Noisy character
            results['noisy_characters']['hebbian']['time_ms'].append(hebbian_time)
            results['noisy_characters']['hebbian']['coherence'].append(hebbian_coherence)
            results['noisy_characters']['hebbian']['memory_kb'].append(hebbian_memory)
            
            results['noisy_characters']['predictive']['time_ms'].append(predictive_time)
            results['noisy_characters']['predictive']['coherence'].append(predictive_coherence)
            results['noisy_characters']['predictive']['memory_kb'].append(predictive_memory)
        elif occlusion > 0.0:
            # Occluded character
            results['occluded_characters']['hebbian']['time_ms'].append(hebbian_time)
            results['occluded_characters']['hebbian']['coherence'].append(hebbian_coherence)
            results['occluded_characters']['hebbian']['memory_kb'].append(hebbian_memory)
            
            results['occluded_characters']['predictive']['time_ms'].append(predictive_time)
            results['occluded_characters']['predictive']['coherence'].append(predictive_coherence)
            results['occluded_characters']['predictive']['memory_kb'].append(predictive_memory)
    
    # Compute overall averages
    for category in ['clean_characters', 'noisy_characters', 'occluded_characters']:
        for model in ['hebbian', 'predictive']:
            for metric in ['time_ms', 'coherence', 'memory_kb']:
                if results[category][model][metric]:
                    results[category][model][f'avg_{metric}'] = np.mean(results[category][model][metric])
                else:
                    results[category][model][f'avg_{metric}'] = 0.0
    
    # Compute overall performance ratios
    for metric in ['time_ratio', 'memory_ratio', 'coherence_ratio']:
        if results['performance_ratios'][metric]:
            results['performance_ratios'][f'avg_{metric}'] = np.mean(results['performance_ratios'][metric])
        else:
            results['performance_ratios'][f'avg_{metric}'] = 0.0
    
    return results

def analyze_scaling_benchmarks(hebbian_csv: str, predictive_csv: str) -> Dict[str, Any]:
    """
    Analyze scaling benchmark results.
    
    Args:
        hebbian_csv: Path to the Hebbian scaling benchmark CSV
        predictive_csv: Path to the Predictive scaling benchmark CSV
        
    Returns:
        Dictionary with analysis results
    """
    hebbian_data = load_csv(hebbian_csv)
    predictive_data = load_csv(predictive_csv)
    
    # Group by grid size
    hebbian_by_size = {}
    predictive_by_size = {}
    
    for row in hebbian_data:
        key = (int(row['grid_size_x']), int(row['grid_size_y']))
        if key not in hebbian_by_size:
            hebbian_by_size[key] = []
        hebbian_by_size[key].append(row)
    
    for row in predictive_data:
        key = (int(row['grid_size_x']), int(row['grid_size_y']), int(row['oscillator_dim']))
        if key not in predictive_by_size:
            predictive_by_size[key] = []
        predictive_by_size[key].append(row)
    
    # Compute average metrics for each grid size
    results = {
        'hebbian_by_size': {},
        'predictive_by_size': {},
        'scaling_factors': {
            'hebbian_time_scaling': [],
            'predictive_time_scaling': [],
            'hebbian_memory_scaling': [],
            'predictive_memory_scaling': []
        }
    }
    
    # Process Hebbian data
    grid_sizes = sorted(hebbian_by_size.keys())
    for size in grid_sizes:
        avg_time = np.mean([row['total_time_ms'] for row in hebbian_by_size[size]])
        avg_memory = np.mean([row['peak_memory_kb'] for row in hebbian_by_size[size]])
        
        results['hebbian_by_size'][size] = {
            'avg_time_ms': avg_time,
            'avg_memory_kb': avg_memory,
            'grid_size': size[0] * size[1]  # Total number of oscillators
        }
    
    # Process Predictive data
    for size in sorted(predictive_by_size.keys()):
        grid_size, _, oscillator_dim = size
        avg_time = np.mean([row['total_time_ms'] for row in predictive_by_size[size]])
        avg_memory = np.mean([row['peak_memory_kb'] for row in predictive_by_size[size]])
        
        results['predictive_by_size'][size] = {
            'avg_time_ms': avg_time,
            'avg_memory_kb': avg_memory,
            'grid_size': size[0] * size[1],  # Total number of oscillators
            'oscillator_dim': oscillator_dim
        }
    
    # Calculate scaling factors
    if len(grid_sizes) >= 2:
        for i in range(1, len(grid_sizes)):
            size1 = grid_sizes[i-1]
            size2 = grid_sizes[i]
            
            # Grid size ratio
            grid_ratio = (size2[0] * size2[1]) / (size1[0] * size1[1])
            
            # Time scaling
            time_ratio = results['hebbian_by_size'][size2]['avg_time_ms'] / results['hebbian_by_size'][size1]['avg_time_ms']
            results['scaling_factors']['hebbian_time_scaling'].append(time_ratio / grid_ratio)
            
            # Memory scaling
            memory_ratio = results['hebbian_by_size'][size2]['avg_memory_kb'] / results['hebbian_by_size'][size1]['avg_memory_kb']
            results['scaling_factors']['hebbian_memory_scaling'].append(memory_ratio / grid_ratio)
    
    # Calculate average scaling factors
    for metric in ['hebbian_time_scaling', 'predictive_time_scaling', 'hebbian_memory_scaling', 'predictive_memory_scaling']:
        if results['scaling_factors'][metric]:
            results['scaling_factors'][f'avg_{metric}'] = np.mean(results['scaling_factors'][metric])
        else:
            results['scaling_factors'][f'avg_{metric}'] = 0.0
    
    return results

def generate_report(character_results: Dict[str, Any], scaling_results: Dict[str, Any]) -> str:
    """
    Generate a summary report with key findings.
    
    Args:
        character_results: Results from character benchmark analysis
        scaling_results: Results from scaling benchmark analysis
        
    Returns:
        Path to the report file
    """
    report_path = 'metrics/reports/benchmark_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("=== Oscillator-Based Learning Benchmark Summary ===\n\n")
        
        # Character processing results
        f.write("== Character Processing Performance ==\n\n")
        
        # Clean characters
        f.write("Clean Characters:\n")
        f.write(f"  Hebbian Kuramoto:\n")
        f.write(f"    Average Time: {character_results['clean_characters']['hebbian'].get('avg_time_ms', 0):.2f} ms\n")
        f.write(f"    Average Coherence: {character_results['clean_characters']['hebbian'].get('avg_coherence', 0):.4f}\n")
        f.write(f"    Average Memory: {character_results['clean_characters']['hebbian'].get('avg_memory_kb', 0):.2f} KB\n\n")
        
        f.write(f"  Predictive Hebbian:\n")
        f.write(f"    Average Time: {character_results['clean_characters']['predictive'].get('avg_time_ms', 0):.2f} ms\n")
        f.write(f"    Average Coherence: {character_results['clean_characters']['predictive'].get('avg_coherence', 0):.4f}\n")
        f.write(f"    Average Memory: {character_results['clean_characters']['predictive'].get('avg_memory_kb', 0):.2f} KB\n\n")
        
        # Noisy characters
        f.write("Noisy Characters:\n")
        f.write(f"  Hebbian Kuramoto:\n")
        f.write(f"    Average Time: {character_results['noisy_characters']['hebbian'].get('avg_time_ms', 0):.2f} ms\n")
        f.write(f"    Average Coherence: {character_results['noisy_characters']['hebbian'].get('avg_coherence', 0):.4f}\n")
        f.write(f"    Average Memory: {character_results['noisy_characters']['hebbian'].get('avg_memory_kb', 0):.2f} KB\n\n")
        
        f.write(f"  Predictive Hebbian:\n")
        f.write(f"    Average Time: {character_results['noisy_characters']['predictive'].get('avg_time_ms', 0):.2f} ms\n")
        f.write(f"    Average Coherence: {character_results['noisy_characters']['predictive'].get('avg_coherence', 0):.4f}\n")
        f.write(f"    Average Memory: {character_results['noisy_characters']['predictive'].get('avg_memory_kb', 0):.2f} KB\n\n")
        
        # Performance ratios
        f.write("Performance Ratios (Predictive / Hebbian):\n")
        f.write(f"  Time Ratio: {character_results['performance_ratios'].get('avg_time_ratio', 0):.2f}x\n")
        f.write(f"  Memory Ratio: {character_results['performance_ratios'].get('avg_memory_ratio', 0):.2f}x\n")
        f.write(f"  Coherence Ratio: {character_results['performance_ratios'].get('avg_coherence_ratio', 0):.2f}x\n\n")
        
        # Scaling results
        f.write("== Scaling Performance ==\n\n")
        
        # Hebbian scaling
        f.write("Hebbian Kuramoto Scaling:\n")
        for size, metrics in sorted(scaling_results['hebbian_by_size'].items()):
            f.write(f"  Grid Size {size[0]}x{size[1]} ({metrics['grid_size']} oscillators):\n")
            f.write(f"    Time: {metrics['avg_time_ms']:.2f} ms\n")
            f.write(f"    Memory: {metrics['avg_memory_kb']:.2f} KB\n\n")
        
        # Predictive scaling
        f.write("Predictive Hebbian Scaling:\n")
        for size, metrics in sorted(scaling_results['predictive_by_size'].items()):
            f.write(f"  Grid Size {size[0]}x{size[1]} ({metrics['grid_size']} oscillators), Dim {size[2]}:\n")
            f.write(f"    Time: {metrics['avg_time_ms']:.2f} ms\n")
            f.write(f"    Memory: {metrics['avg_memory_kb']:.2f} KB\n\n")
        
        # Scaling factors
        f.write("Scaling Factors:\n")
        f.write(f"  Hebbian Time Scaling: {scaling_results['scaling_factors'].get('avg_hebbian_time_scaling', 0):.2f}x\n")
        f.write(f"  Hebbian Memory Scaling: {scaling_results['scaling_factors'].get('avg_hebbian_memory_scaling', 0):.2f}x\n\n")
        
        # Key findings
        f.write("== Key Findings ==\n\n")
        
        # Time comparison
        time_ratio = character_results['performance_ratios'].get('avg_time_ratio', 0)
        if time_ratio > 1.5:
            f.write(f"1. Predictive Hebbian is {time_ratio:.1f}x slower than Hebbian Kuramoto\n")
        else:
            f.write(f"1. Predictive Hebbian and Hebbian Kuramoto have comparable processing times\n")
        
        # Memory comparison
        memory_ratio = character_results['performance_ratios'].get('avg_memory_ratio', 0)
        if memory_ratio > 1.2:
            f.write(f"2. Predictive Hebbian uses {memory_ratio:.1f}x more memory than Hebbian Kuramoto\n")
        else:
            f.write(f"2. Predictive Hebbian and Hebbian Kuramoto have comparable memory usage\n")
        
        # Coherence comparison
        coherence_ratio = character_results['performance_ratios'].get('avg_coherence_ratio', 0)
        if coherence_ratio > 1.2:
            f.write(f"3. Predictive Hebbian achieves {coherence_ratio:.1f}x higher coherence than Hebbian Kuramoto\n")
        else:
            f.write(f"3. Predictive Hebbian and Hebbian Kuramoto have comparable coherence\n")
        
        # Scaling comparison
        hebbian_time_scaling = scaling_results['scaling_factors'].get('avg_hebbian_time_scaling', 0)
        if hebbian_time_scaling > 0:
            f.write(f"4. Hebbian Kuramoto processing time scales with factor {hebbian_time_scaling:.2f} relative to grid size\n")
        
        # Noise robustness
        hebb_clean_coherence = character_results['clean_characters']['hebbian'].get('avg_coherence', 0)
        hebb_noisy_coherence = character_results['noisy_characters']['hebbian'].get('avg_coherence', 0)
        pred_clean_coherence = character_results['clean_characters']['predictive'].get('avg_coherence', 0)
        pred_noisy_coherence = character_results['noisy_characters']['predictive'].get('avg_coherence', 0)
        
        if hebb_clean_coherence > 0 and pred_clean_coherence > 0:
            hebb_noise_ratio = hebb_noisy_coherence / hebb_clean_coherence
            pred_noise_ratio = pred_noisy_coherence / pred_clean_coherence
            
            if pred_noise_ratio > hebb_noise_ratio:
                f.write(f"5. Predictive Hebbian shows better noise robustness than Hebbian Kuramoto\n")
            else:
                f.write(f"5. Hebbian Kuramoto shows better noise robustness than Predictive Hebbian\n")
    
    return report_path

def plot_scaling_results(scaling_results: Dict[str, Any]) -> str:
    """
    Generate plots for scaling results.
    
    Args:
        scaling_results: Results from scaling benchmark analysis
        
    Returns:
        Path to the plot file
    """
    plot_path = 'metrics/reports/scaling_plot.png'
    
    # Extract data for plotting
    hebbian_sizes = []
    hebbian_times = []
    hebbian_memories = []
    
    for size, metrics in sorted(scaling_results['hebbian_by_size'].items()):
        hebbian_sizes.append(metrics['grid_size'])
        hebbian_times.append(metrics['avg_time_ms'])
        hebbian_memories.append(metrics['avg_memory_kb'])
    
    # Group predictive data by oscillator dimension
    predictive_by_dim = {}
    
    for size, metrics in scaling_results['predictive_by_size'].items():
        dim = size[2]
        if dim not in predictive_by_dim:
            predictive_by_dim[dim] = {'sizes': [], 'times': [], 'memories': []}
        
        predictive_by_dim[dim]['sizes'].append(metrics['grid_size'])
        predictive_by_dim[dim]['times'].append(metrics['avg_time_ms'])
        predictive_by_dim[dim]['memories'].append(metrics['avg_memory_kb'])
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Time scaling plot
    ax1.plot(hebbian_sizes, hebbian_times, 'o-', label='Hebbian Kuramoto')
    
    for dim, data in predictive_by_dim.items():
        ax1.plot(data['sizes'], data['times'], 'o-', label=f'Predictive Hebbian (Dim {dim})')
    
    ax1.set_xlabel('Grid Size (Number of Oscillators)')
    ax1.set_ylabel('Processing Time (ms)')
    ax1.set_title('Processing Time vs. Grid Size')
    ax1.legend()
    ax1.grid(True)
    
    # Memory scaling plot
    ax2.plot(hebbian_sizes, hebbian_memories, 'o-', label='Hebbian Kuramoto')
    
    for dim, data in predictive_by_dim.items():
        ax2.plot(data['sizes'], data['memories'], 'o-', label=f'Predictive Hebbian (Dim {dim})')
    
    ax2.set_xlabel('Grid Size (Number of Oscillators)')
    ax2.set_ylabel('Memory Usage (KB)')
    ax2.set_title('Memory Usage vs. Grid Size')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def main():
    """Main function to analyze benchmarks and generate reports."""
    # Find the most recent benchmark files
    metrics_dir = 'metrics'
    hebbian_benchmark = None
    predictive_benchmark = None
    hebbian_scaling = None
    predictive_scaling = None
    
    for file in os.listdir(metrics_dir):
        if file.startswith('hebbian_kuramoto_benchmark_') and file.endswith('.csv'):
            if hebbian_benchmark is None or file > hebbian_benchmark:
                hebbian_benchmark = file
        elif file.startswith('predictive_hebbian_benchmark_') and file.endswith('.csv'):
            if predictive_benchmark is None or file > predictive_benchmark:
                predictive_benchmark = file
        elif file.startswith('hebbian_kuramoto_scaling_') and file.endswith('.csv'):
            if hebbian_scaling is None or file > hebbian_scaling:
                hebbian_scaling = file
        elif file.startswith('predictive_hebbian_scaling_') and file.endswith('.csv'):
            if predictive_scaling is None or file > predictive_scaling:
                predictive_scaling = file
    
    if hebbian_benchmark and predictive_benchmark:
        print(f"Analyzing character benchmarks:")
        print(f"  Hebbian: {hebbian_benchmark}")
        print(f"  Predictive: {predictive_benchmark}")
        
        character_results = analyze_character_benchmarks(
            os.path.join(metrics_dir, hebbian_benchmark),
            os.path.join(metrics_dir, predictive_benchmark)
        )
    else:
        print("Character benchmark files not found")
        character_results = {
            'clean_characters': {
                'hebbian': {}, 'predictive': {}
            },
            'noisy_characters': {
                'hebbian': {}, 'predictive': {}
            },
            'occluded_characters': {
                'hebbian': {}, 'predictive': {}
            },
            'performance_ratios': {}
        }
    
    if hebbian_scaling and predictive_scaling:
        print(f"Analyzing scaling benchmarks:")
        print(f"  Hebbian: {hebbian_scaling}")
        print(f"  Predictive: {predictive_scaling}")
        
        scaling_results = analyze_scaling_benchmarks(
            os.path.join(metrics_dir, hebbian_scaling),
            os.path.join(metrics_dir, predictive_scaling)
        )
        
        # Generate scaling plot
        plot_path = plot_scaling_results(scaling_results)
        print(f"Scaling plot saved to {plot_path}")
    else:
        print("Scaling benchmark files not found")
        scaling_results = {
            'hebbian_by_size': {},
            'predictive_by_size': {},
            'scaling_factors': {}
        }
    
    # Generate report
    report_path = generate_report(character_results, scaling_results)
    print(f"Benchmark summary report saved to {report_path}")

if __name__ == "__main__":
    main()
