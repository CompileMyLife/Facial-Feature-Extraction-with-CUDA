#!/usr/bin/env python3

import os
import re
import argparse
import matplotlib.pyplot as plt

# Pattern for extracting log information
patterns_dict = {
    'TIME_IN_SECONDS': r'Time = \d+ nanoseconds\s*\((\d+\.\d+) sec\)',
    'CPU_CANDIDATES': r'CPU detection detected (\d+) candidates',
    'GPU_CANDIDATES': r'CUDA detection detected (\d+) candidates',
    'IMAGE_NAME': r'Reading image file: .*?photo(\d+)_(\d+)_people\.pgm'
}

# Function to parse a log file and extract relevant data
def parse(log_file_path, is_cpu):
    results_dict = {}
    with open(log_file_path, 'r') as log_file:
        content = log_file.read()
        times = re.findall(patterns_dict['TIME_IN_SECONDS'], content)
        candidates = re.findall(patterns_dict['CPU_CANDIDATES'] if is_cpu else patterns_dict['GPU_CANDIDATES'], content)
        images = re.findall(patterns_dict['IMAGE_NAME'], content)

        for (image_id, actual_faces), time, candidate in zip(images, times, candidates):
            time = float(time)
            candidate = int(candidate)
            actual_faces = int(actual_faces)
            image = f'photo{image_id}_{actual_faces}_people.pgm'
            if image not in results_dict:
                results_dict[image] = {'cpu_time': 0, 'gpu_time': 0, 'cpu_candidates': 0, 'gpu_candidates': 0, 'actual_faces': actual_faces}
            if is_cpu:
                results_dict[image]['cpu_time'] += time
                results_dict[image]['cpu_candidates'] += candidate
            else:
                results_dict[image]['gpu_time'] += time
                results_dict[image]['gpu_candidates'] += candidate
    return results_dict

# Function to calculate speedup and accuracy
def calc_metrics(cpu_results_dict, gpu_results_dict):
    metrics_dict = {}
    for image in cpu_results_dict:
        if image in gpu_results_dict:
            cpu_time = cpu_results_dict[image]['cpu_time']
            gpu_time = gpu_results_dict[image]['gpu_time']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            actual_faces = cpu_results_dict[image]['actual_faces']
            cpu_acc = cpu_results_dict[image]['cpu_candidates'] / actual_faces
            gpu_acc = gpu_results_dict[image]['gpu_candidates'] / actual_faces
            metrics_dict[image] = {'cpu_time': cpu_time, 'gpu_time': gpu_time, 'speedup': speedup, 'cpu_acc': cpu_acc, 'gpu_acc': gpu_acc}
    return metrics_dict

# Function to generate and save plots
def gen_plots(metrics_dict, logs_dir):
    images = list(metrics_dict.keys())
    cpu_times = [metrics_dict[image]['cpu_time'] for image in images]
    gpu_times = [metrics_dict[image]['gpu_time'] for image in images]
    speedups = [metrics_dict[image]['speedup'] for image in images]
    cpu_acc = [metrics_dict[image]['cpu_acc'] for image in images]
    gpu_acc = [metrics_dict[image]['gpu_acc'] for image in images]

    plots_dir = os.path.join(logs_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Time comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(images, cpu_times, label='CPU Time', alpha=0.6)
    ax.bar(images, gpu_times, label='GPU Time', alpha=0.6)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('CPU vs GPU Time for Each Image')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_comparison.png'))
    plt.close()

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(images, cpu_acc, label='CPU Accuracy', alpha=0.6)
    ax.bar(images, gpu_acc, label='GPU Accuracy', alpha=0.6)
    ax.set_ylabel('Accuracy')
    ax.set_title('CPU vs GPU Accuracy for Each Image')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_comparison.png'))
    plt.close()

    # Print speedup values
    for image in images:
        print(f"{image} - Speedup: {metrics_dict[image]['speedup']:.2f}")

# Main function
def main(cpu_logs, gpu_logs, logs_dir):
    cpu_results_dict = {}
    gpu_results_dict = {}

    for log_file in os.listdir(cpu_logs):
        if log_file.endswith('.log'):
            cpu_results_dict.update(parse(os.path.join(cpu_logs, log_file), is_cpu=True))

    for log_file in os.listdir(gpu_logs):
        if log_file.endswith('.log'):
            gpu_results_dict.update(parse(os.path.join(gpu_logs, log_file), is_cpu=False))

    metrics_dict = calc_metrics(cpu_results_dict, gpu_results_dict)
    gen_plots(metrics_dict, logs_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse and analyze CPU/GPU log files.')
    parser.add_argument('cpu_logs', type=str, help='Path to CPU logs directory')
    parser.add_argument('gpu_logs', type=str, help='Path to GPU logs directory')
    parser.add_argument('logs_dir', type=str, help='Path to save plots')
    args = parser.parse_args()
    main(args.cpu_logs, args.gpu_logs, args.logs_dir)
