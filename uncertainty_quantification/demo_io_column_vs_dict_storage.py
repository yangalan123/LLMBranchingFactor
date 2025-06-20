import numpy as np
import psutil
import os
import gc
import time
import torch
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def generate_fixed_indices(shape, vocab_size, top_log_probs):
    total_positions = np.prod(shape[:-1])
    indices = np.linspace(0, vocab_size - 1, top_log_probs, dtype=np.int32)
    indices = np.tile(indices, (total_positions, 1))
    return indices.reshape(shape)


def generate_batch_data(args):
    data_i, num_outputs, seq_length, top_log_probs, vocab_size = args
    outputs = []
    rand_indices = generate_fixed_indices((num_outputs, seq_length, top_log_probs), vocab_size, top_log_probs)
    rand_values = np.random.rand(num_outputs, seq_length, top_log_probs).astype(np.float32)

    for output_i in range(num_outputs):
        _outputs = []
        for i in range(seq_length):
            output_dict_i = dict(zip(rand_indices[output_i, i], rand_values[output_i, i]))
            _outputs.append(output_dict_i)
        outputs.append(_outputs)
    return outputs


def generate_columnar_batch(args):
    data_i, num_outputs, seq_length, top_log_probs, vocab_size = args
    keys = generate_fixed_indices((num_outputs, seq_length, top_log_probs), vocab_size, top_log_probs)
    values = np.random.rand(num_outputs, seq_length, top_log_probs).astype(np.float32)
    return data_i, keys, values


def benchmark_nested_structure(seq_length, top_log_probs, data_num, num_outputs, vocab_size):
    initial_memory = get_memory_usage()

    args_list = [(i, num_outputs, seq_length, top_log_probs, vocab_size) for i in range(data_num)]
    n_processes = min(cpu_count(), 16)

    with Pool(n_processes) as pool:
        outputs = list(tqdm(pool.imap(generate_batch_data, args_list),
                            total=data_num, desc="Generating nested structure"))

    gc.collect()
    return outputs, get_memory_usage() - initial_memory


def benchmark_columnar_structure(seq_length, top_log_probs, data_num, num_outputs, vocab_size):
    initial_memory = get_memory_usage()

    columnar_outputs = {
        'keys': np.zeros((data_num, num_outputs, seq_length, top_log_probs), dtype=np.int32),
        'values': np.zeros((data_num, num_outputs, seq_length, top_log_probs), dtype=np.float32)
    }

    args_list = [(i, num_outputs, seq_length, top_log_probs, vocab_size) for i in range(data_num)]
    n_processes = min(cpu_count(), 16)

    with Pool(n_processes) as pool:
        for data_i, keys, values in tqdm(pool.imap(generate_columnar_batch, args_list),
                                         total=data_num, desc="Generating columnar structure"):
            columnar_outputs['keys'][data_i] = keys
            columnar_outputs['values'][data_i] = values

    gc.collect()
    return columnar_outputs, get_memory_usage() - initial_memory


def convert_columnar_to_dict(columnar_data):
    start_time = time.time()
    data_num, num_outputs, seq_length, top_log_probs = columnar_data['keys'].shape

    outputs = []
    for data_i in tqdm(range(data_num), desc="Converting columnar to dict"):
        outputs.append([])
        for output_i in range(num_outputs):
            _outputs = []
            for i in range(seq_length):
                output_dict_i = dict(zip(
                    columnar_data['keys'][data_i, output_i, i],
                    columnar_data['values'][data_i, output_i, i]
                ))
                _outputs.append(output_dict_i)
            outputs[data_i].append(_outputs)

    conversion_time = time.time() - start_time
    return outputs, conversion_time


def benchmark_serialization(nested_data, columnar_data, temp_dir):
    results = {}

    # Nested structure serialization
    start_time = time.time()
    nested_path = temp_dir / "nested.pt"
    torch.save(nested_data, nested_path)
    results['nested_save_time'] = time.time() - start_time
    results['nested_file_size'] = nested_path.stat().st_size / (1024 * 1024)  # MB

    start_time = time.time()
    torch.load(nested_path)
    results['nested_load_time'] = time.time() - start_time

    # Columnar structure serialization
    start_time = time.time()
    columnar_path = temp_dir / "columnar.pt"
    torch.save(columnar_data, columnar_path)
    results['columnar_save_time'] = time.time() - start_time
    results['columnar_file_size'] = columnar_path.stat().st_size / (1024 * 1024)  # MB

    start_time = time.time()
    torch.load(columnar_path)
    results['columnar_load_time'] = time.time() - start_time

    return results


if __name__ == '__main__':
    # Updated parameters
    seq_length = 128
    top_log_probs = 50
    data_num = 100
    num_outputs = 50
    vocab_size = 200000

    print(f"\nRunning benchmark with {min(cpu_count(), 16)} processes")
    print(f"Parameters: seq_length={seq_length}, top_log_probs={top_log_probs}, "
          f"data_num={data_num}, num_outputs={num_outputs}, vocab_size={vocab_size}\n")

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Benchmark structures
        nested_data, nested_memory = benchmark_nested_structure(
            seq_length, top_log_probs, data_num, num_outputs, vocab_size)
        print(f"Nested structure memory: {nested_memory:.2f} MB")
        # Nested structure memory: 3492.61 MB

        gc.collect()
        time.sleep(1)

        columnar_data, columnar_memory = benchmark_columnar_structure(
            seq_length, top_log_probs, data_num, num_outputs, vocab_size)
        print(f"Columnar structure memory: {columnar_memory:.2f} MB")
        # Columnar structure memory: 467.59 MB
        print(f"Memory ratio (nested/columnar): {nested_memory / columnar_memory:.2f}x\n")

        # Benchmark conversion
        _, conversion_time = convert_columnar_to_dict(columnar_data)
        print(f"Columnar to dict conversion time: {conversion_time:.2f}s\n")
        # Columnar to dict conversion time: 4.50 s

        # Benchmark serialization
        results = benchmark_serialization(nested_data, columnar_data, temp_dir)
        print("Serialization benchmarks:")
        print(f"Nested structure: save={results['nested_save_time']:.2f}s, "
              f"load={results['nested_load_time']:.2f}s, size={results['nested_file_size']:.2f}MB")
        print(f"Columnar structure: save={results['columnar_save_time']:.2f}s, "
              f"load={results['columnar_load_time']:.2f}s, size={results['columnar_file_size']:.2f}MB")
        # Serialization benchmarks: Nested structure: save = 387.74 s, load = 86.98 s, size = 2883.78 MB
        # Columnar structure: save = 1.94 s, load = 0.91 s, size = 315.38 MB

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)