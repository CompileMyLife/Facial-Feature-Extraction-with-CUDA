import os
import re
import argparse
import matplotlib.pyplot as plt

# Pattern for extracting log information
patterns_dict = {
    'TIME_IN_SECONDS': r'Time = \d+ nanoseconds\s*\((\d+\.\d+) sec\)',
    'HOW_MANY_CANDIDATES_DETECTED_CPU': r'CPU detection detected (\d+) candidates',
    'HOW_MANY_CANDIDATES_DETECTED_GPU': r'CUDA detection detected (\d+) candidates',
    'IMAGE_NAME': r'Reading image file: .*?photo(\d+)_(\d+)_people\.pgm'
}

# Function to parse a log file and extract relevant data
def parse(log_file_path, is_cpu):
    results_dict = {}
    with open(log_file_path, 'r') as log_file:
        content = log_file.read()
        times = re.findall(patterns_dict['TIME_IN_SECONDS'], content)
        candidates = re.findall(patterns_dict['HOW_MANY_CANDIDATES_DETECTED_CPU' if is_cpu else 'HOW_MANY_CANDIDATES_DETECTED_GPU'], content)
        images = re.findall(patterns_dict['IMAGE_NAME'], content)

        for (image_id, actual_faces), time, candidate in zip(images, times, candidates):
            time = float(time)
            candidate = int(candidate)
            actual_faces = int(actual_faces)
            image_name = f'photo{image_id}_{actual_faces}_people.pgm'
            if image_name not in results_dict:
                results_dict[image_name] = {'cpu_time': 0, 'gpu_time': 0, 'cpu_candidates': 0, 'gpu_candidates': 0, 'actual_faces': actual_faces}
            if is_cpu:
                results_dict[image_name]['cpu_time'] += time
                results_dict[image_name]['cpu_candidates'] += candidate
            else:
                results_dict[image_name]['gpu_time'] += time
                results_dict[image_name]['gpu_candidates'] += candidate
    return results_dict

# Function to calculate speedup and accuracy
def calc_metrics(cpu_results_dict, gpu_results_dict):
    metrics_dict = {}
    for image in cpu_results_dict:
        if image in gpu_results_dict:
            cpu_time = cpu_results_dict[image]['cpu_time']
            gpu_time = gpu_results_dict[image]['gpu_time']
            cpu_candidates = cpu_results_dict[image]['cpu_candidates']
            gpu_candidates = gpu_results_dict[image]['gpu_candidates']
            actual_faces = cpu_results_dict[image]['actual_faces']
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            cpu_accuracy = cpu_candidates / actual_faces if actual_faces > 0 else 0
            gpu_accuracy = gpu_candidates / actual_faces if actual_faces > 0 else 0
            metrics_dict[image] = {'cpu_time': cpu_time, 'gpu_time': gpu_time, 'speedup': speedup, 'cpu_accuracy': cpu_accuracy, 'gpu_accuracy': gpu_accuracy}
    return metrics_dict

# Function to generate and save plots
def gen_plots(metrics_dict, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    images = list(metrics_dict.keys())
    speedups = [metrics_dict[image]['speedup'] for image in images]
    cpu_accuracies = [metrics_dict[image]['cpu_accuracy'] for image in images]
    gpu_accuracies = [metrics_dict[image]['gpu_accuracy'] for image in images]

    # Speedup plot
    plt.figure(figsize=(10, 5))
    plt.bar(images, speedups, color='blue', alpha=0.6)
    plt.ylabel('Speedup')
    plt.title('Speedup (CPU Time / GPU Time) for Each Image')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'speedup_plot.png'))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.bar(images, cpu_accuracies, label='CPU Accuracy', alpha=0.6)
    plt.bar(images, gpu_accuracies, label='GPU Accuracy', alpha=0.6)
    plt.ylabel('Accuracy')
    plt.title('CPU vs GPU Accuracy')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_plot.png'))
    plt.close()

# Main function
def main(cpu_log, gpu_log, plots_dir):
    cpu_results_dict = parse(cpu_log, is_cpu=True)
    gpu_results_dict = parse(gpu_log, is_cpu=False)
    metrics_dict = calc_metrics(cpu_results_dict, gpu_results_dict)
    gen_plots(metrics_dict, plots_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse and analyze CPU/GPU log files.')
    parser.add_argument('cpu_log', type=str, help='Path to the CPU log file')
    parser.add_argument('gpu_log', type=str, help='Path to the GPU log file')
    parser.add_argument('plots_dir', type=str, help='Directory to save plots')
    args = parser.parse_args()
    main(args.cpu_log, args.gpu_log, args.plots_dir)
