#!/usr/bin/env python3

import os
import re
import argparse
import matplotlib.pyplot as plt

# Pattern for extracting log information
patterns_dict = {
    'TIME_IN_SECONDS': r'TIME_IN_SECONDS:\s*(\d+\.\d+)',
    'HOW_MANY_CANDIDATES_DETECTED': r'HOW_MANY_CANDIDATES_DETECTED:\s*(\d+)',
    'IMAGE_NAME': r'IMAGE_NAME:\s*(\S+)',
}

# Function to parse a log file and extract relevant data
def parse(log_file_path):
    results_dict = {}
    with open(log_file_path, 'r') as log_file:
        content = log_file.read()
        times = re.findall(patterns_dict['TIME_IN_SECONDS'], content)
        candidates = re.findall(patterns_dict['HOW_MANY_CANDIDATES_DETECTED'], content)
        images = re.findall(patterns_dict['IMAGE_NAME'], content)

        for time, candidate, image in zip(times, candidates, images):
            time = float(time)
            candidate = int(candidate)
            if image not in results_dict:
                results_dict[image] = {'cpu_time': 0, 'gpu_time': 0, 'cpu_candidates': 0, 'gpu_candidates': 0}
            if 'cpu' in log_file_path:
                results_dict[image]['cpu_time'] += time
                results_dict[image]['cpu_candidates'] += candidate
            else:
                results_dict[image]['gpu_time'] += time
                results_dict[image]['gpu_candidates'] += candidate
    return results_dict

# Function to calculate speedup
def calc_speedup(cpu_results_dict, gpu_results_dict):
    speedups_dict = {}
    for image in cpu_results_dict:
        if image in gpu_results_dict:
            cpu_time = cpu_results_dict[image]['cpu_time']
            gpu_time = gpu_results_dict[image]['gpu_time']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            speedups_dict[image] = {'cpu_time': cpu_time, 'gpu_time': gpu_time, 'speedup': speedup}
    return speedups_dict

# Function to generate a plot for the speedup
def gen_plots(speedups_dict):
    images = list(speedups_dict.keys())
    cpu_times = [speedups_dict[image]['cpu_time'] for image in images]
    gpu_times = [speedups_dict[image]['gpu_time'] for image in images]
    speedups = [speedups_dict[image]['speedup'] for image in images]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].bar(images, cpu_times, label='CPU Time', alpha=0.6)
    ax[0].bar(images, gpu_times, label='GPU Time', alpha=0.6)
    ax[0].set_ylabel('Time (seconds)')
    ax[0].set_title('CPU vs GPU Time for Each Image')
    ax[0].legend()

    ax[1].bar(images, speedups, color='green', alpha=0.6)
    ax[1].set_ylabel('Speedup')
    ax[1].set_title('Speedup (CPU Time / GPU Time) for Each Image')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function
def main(logs_dir):
    cpu_dir = os.path.join(logs_dir, 'cpu')
    gpu_dir = os.path.join(logs_dir, 'gpu')
    cpu_results_dict = {}
    gpu_results_dict = {}

    if not os.path.isdir(cpu_dir) or not os.path.isdir(gpu_dir):
        print("Error: Both 'cpu' and 'gpu' directories must exist inside the logs directory.")
        return

    for log_file in os.listdir(cpu_dir):
        if log_file.endswith('.txt'):
            cpu_results_dict.update(parse(os.path.join(cpu_dir, log_file)))

    for log_file in os.listdir(gpu_dir):
        if log_file.endswith('.txt'):
            gpu_results_dict.update(parse(os.path.join(gpu_dir, log_file)))

    speedups_dict = calc_speedup(cpu_results_dict, gpu_results_dict)
    gen_plots(speedups_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse and analyze CPU/GPU log files.')
    parser.add_argument('logs_dir', type=str, help='Path to the logs directory containing cpu/ and gpu/ subdirectories')
    args = parser.parse_args()
    main(args.logs_dir)
