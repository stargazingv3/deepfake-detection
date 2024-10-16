import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

def measure_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def benchmark_configuration(data_path, batch_size, num_threads, min_batches=5):
    """Benchmark a specific batch size and thread count combination
    
    Args:
        data_path: Path to the dataset
        batch_size: Size of each batch
        num_threads: Number of threads to use
        min_batches: Minimum number of batches to process for accurate benchmarking
    """
    loaded_data = load_from_disk(data_path)
    
    # Calculate number of samples to process
    # Process at least min_batches batches to get meaningful metrics
    num_samples = batch_size * min_batches
    
    start_time = time.time()
    start_memory = measure_memory_usage()
    
    processed_samples = 0
    batch_times = []
    memory_usage = []
    
    # Process batches
    for batch in loaded_data.iter(batch_size=batch_size):
        batch_start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(len(batch['id'])):
                futures.append(executor.submit(lambda: time.sleep(0.001)))
            
            for future in futures:
                future.result()
        
        batch_end = time.time()
        
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        memory_usage.append(measure_memory_usage() - start_memory)
        
        processed_samples += len(batch['id'])
        if processed_samples >= num_samples:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'throughput': processed_samples / total_time,
        'avg_batch_time': np.mean(batch_times),
        'avg_memory_mb': np.mean(memory_usage),
        'total_time': total_time,
        'samples_processed': processed_samples,
        'num_batches': len(batch_times)
    }

def benchmark_combinations(data_path, batch_sizes, thread_counts, min_batches=5):
    results = []
    total_combinations = len(batch_sizes) * len(thread_counts)
    
    with tqdm(total=total_combinations, desc="Testing configurations") as pbar:
        for num_threads in thread_counts:
            for batch_size in batch_sizes:
                metrics = benchmark_configuration(data_path, batch_size, num_threads, min_batches)
                
                result = {
                    'batch_size': batch_size,
                    'num_threads': num_threads,
                    **metrics
                }
                results.append(result)

                results_df = pd.DataFrame(results)
                results_df.to_csv('batch_thread_benchmark_results.csv', mode='a', index=False)
                
                print(f"\nBatch Size: {batch_size}, Threads: {num_threads}")
                print(f"Samples processed: {metrics['samples_processed']}")
                print(f"Number of batches: {metrics['num_batches']}")
                print(f"Throughput: {metrics['throughput']:.2f} samples/second")
                print(f"Average batch time: {metrics['avg_batch_time']:.3f} seconds")
                print(f"Average memory usage: {metrics['avg_memory_mb']:.1f} MB")
                print(f"Total time: {metrics['total_time']:.2f} seconds")
                print("-" * 50)
                
                pbar.update(1)
    
    return pd.DataFrame(results)

def plot_heatmap(results):
    # Create pivot table for heatmap
    pivot = results.pivot(index='batch_size', columns='num_threads', values='throughput')
    
    plt.figure(figsize=(15, 10))
    plt.imshow(pivot, cmap='viridis', aspect='auto')
    plt.colorbar(label='Throughput (samples/second)')
    
    # Set x and y axis labels
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Batch Size')
    plt.title('Throughput Heatmap: Batch Size vs Thread Count')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f'{pivot.iloc[i, j]:.0f}',
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('throughput_heatmap.png')
    plt.show()

def plot_detailed_results(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Group results by batch size and plot for different thread counts
    batch_sizes = sorted(results['batch_size'].unique())
    thread_counts = sorted(results['num_threads'].unique())
    
    for thread_count in thread_counts:
        thread_data = results[results['num_threads'] == thread_count]
        
        # Throughput plot
        ax1.plot(thread_data['batch_size'], thread_data['throughput'], 
                marker='o', label=f'{thread_count} threads')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (samples/second)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.grid(True)
        ax1.legend()
        
        # Memory usage plot
        ax2.plot(thread_data['batch_size'], thread_data['avg_memory_mb'],
                marker='o', label=f'{thread_count} threads')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Average Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Batch Size')
        ax2.grid(True)
        ax2.legend()
        
        # Batch time plot
        ax3.plot(thread_data['batch_size'], thread_data['avg_batch_time'],
                marker='o', label=f'{thread_count} threads')
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Average Batch Time (seconds)')
        ax3.set_title('Batch Processing Time vs Batch Size')
        ax3.grid(True)
        ax3.legend()
        
        # Total time plot
        ax4.plot(thread_data['batch_size'], thread_data['total_time'],
                marker='o', label=f'{thread_count} threads')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Total Time (seconds)')
        ax4.set_title('Total Processing Time vs Batch Size')
        ax4.grid(True)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('detailed_results.png')
    plt.show()

def find_optimal_configuration(data_path="/data/deepfakes"):
    # Test ranges
    #batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    #thread_counts = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    batch_sizes = [4096]
    thread_counts = [1024, 2048, 4096]
    
    print("Starting batch size and thread count benchmark...")
    print("This will test various combinations to measure performance metrics.")
    
    results = benchmark_combinations(data_path, batch_sizes, thread_counts, min_batches=3)
    
    # Plot results
    plot_heatmap(results)
    plot_detailed_results(results)
    
    # Find optimal configuration based on throughput
    optimal_idx = results['throughput'].idxmax()
    optimal_config = results.iloc[optimal_idx]
    
    print("\nOptimal Configuration:")
    print(f"Batch Size: {optimal_config['batch_size']}")
    print(f"Number of Threads: {optimal_config['num_threads']}")
    print(f"Throughput: {optimal_config['throughput']:.2f} samples/second")
    print(f"Memory Usage: {optimal_config['avg_memory_mb']:.1f} MB")
    print(f"Average Batch Time: {optimal_config['avg_batch_time']:.3f} seconds")
    
    return optimal_config['batch_size'], optimal_config['num_threads']

if __name__ == "__main__":
    optimal_batch_size, optimal_threads = find_optimal_configuration()
