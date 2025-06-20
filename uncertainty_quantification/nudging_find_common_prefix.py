import gc
import glob
import os
from multiprocessing import Pool, Manager
from multiprocessing.managers import DictProxy, ListProxy

from tqdm.auto import tqdm
from loguru import logger
import time
import tempfile
import shutil
import psutil
from typing import Dict, List, Tuple
from uncertainty_quantification.consts import ALL_MODELS
from uncertainty_quantification.io_utils import StoreManager
from collections import Counter
import argparse

# Constants
DEFAULT_MAX_WORKERS = 3
DEFAULT_MEMORY_THRESHOLD = 0.85
DEFAULT_SAVE_INTERVAL = 300  # 5 minutes


def build_prefix_tree(tree: Dict, suffix: List[int], multiplier: int):
    """Build a local prefix tree from a list of tokens up to max_length"""
    if len(suffix) == 0:
        return
    if suffix[0] not in tree:
        tree[suffix[0]] = {"count": multiplier, "children": {}}
    else:
        tree[suffix[0]]["count"] += multiplier
    build_prefix_tree(tree[suffix[0]]["children"], suffix[1:], multiplier)


def create_shared_tree_node(manager):
    """Create a new shared tree node using manager dictionaries"""
    return manager.dict({
        "count": 0,
        "children": manager.dict()
    })


def merge_trees(global_tree: Dict, local_tree: Dict, manager):
    """Recursively merge local tree into global tree using shared dictionaries"""
    for token, node in local_tree.items():
        if token not in global_tree:
            global_tree[token] = create_shared_tree_node(manager)

        # Update count
        global_tree[token]["count"] += node["count"]

        # Merge children
        if node["children"]:
            merge_trees(global_tree[token]["children"], node["children"], manager)


def process_file(process_args: Tuple[str, str, str, argparse.Namespace]) -> Tuple[str, str, Dict]:
    """Process a single file and return its local tree"""
    filename, key, temp_dir, args = process_args
    max_prefix_length = args.max_prefix_length
    memory_threshold = args.memory_threshold
    try:
        # Wait if memory usage is too high
        while psutil.virtual_memory().percent / 100 > memory_threshold:
            time.sleep(5)
            gc.collect()

        # Create a temporary StoreManager for this process
        with StoreManager(temp_dir=temp_dir) as store:
            # Generate a unique name for this file

            # Copy and load file through StoreManager
            logger.info(f"Loading {filename}")
            responses = store.load(filename)
            logger.info(f"Loaded {filename}")

            # Use regular dict for local tree (will be merged later)
            local_tree = {}
            # further accelerate the process by using Counter
            local_counter = Counter()
            for response in responses:
                for output in response.outputs:
                    token_prefix = output.token_ids[:max_prefix_length]
                    # build_prefix_tree(local_tree, token_prefix, max_prefix_length)
                    local_counter[tuple(token_prefix)] += 1
            for token_prefix, count in local_counter.items():
                build_prefix_tree(local_tree, list(token_prefix), count)

            # Cleanup
            del responses
            gc.collect()

            return filename, key, local_tree
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return filename, key, {}


def process_model_constraint(model: str, constraint: int, source_dir: str,
                             manager_dict: Dict,
                             store: StoreManager, state_name: str,
                             args: argparse.Namespace,
                             # save_interval: int, max_prefix_length: int,
                             manager) -> None:
    """Process all files for a given model and constraint"""
    max_prefix_length = args.max_prefix_length
    save_interval = args.save_interval
    max_workers = args.max_workers
    files_to_process = glob.glob(os.path.join(source_dir, f"{model}*pt"))
    files_to_process = [x for x in files_to_process if "patch" not in x and "spectrum" not in x]
    if not files_to_process:
        return

    key = f"{constraint}_{model}"
    logger.info(f"Processing {len(files_to_process)} files for model {model} under {source_dir}")

    if not files_to_process:
        logger.info(f"Skipping {key} as all files have been processed")
        return

    # Initialize structures if needed
    if key not in manager_dict:
        manager_dict[key] = manager.dict({
            "tree": manager.dict(),
            "processed_files": manager.list()
        })

    # Create temporary directory for file processing
    temp_dir = tempfile.mkdtemp(prefix='prefix_tree_')
    try:
        # Process files with limited concurrency
        last_save_time = time.time()
        process_args = [(f, key, temp_dir, args) for f in files_to_process]

        with Pool(processes=max_workers) as pool:
            for filename, key, local_tree in tqdm(
                    pool.imap_unordered(process_file, process_args),
                    total=len(files_to_process),
                    desc=f"Processing {key}"
            ):
                if local_tree:  # Only process if we got valid results
                    logger.info(f"(before merge) Current tree size: {len(manager_dict[key]['tree'])}, local tree size: {len(local_tree)}")
                    merge_trees(manager_dict[key]["tree"], local_tree, manager)
                    logger.info(f"Processed {filename}")
                    manager_dict[key]["processed_files"].append(filename)
                    logger.info(f"(after merge) Current tree size: {len(manager_dict[key]['tree'])}, processed files: {len(manager_dict[key]['processed_files'])}")

                    # Periodic saving
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        # Convert to regular dict for saving
                        state = convert_shared_to_regular_dict(manager_dict)
                        store.save(state, state_name, async_write=True)
                        last_save_time = current_time

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def convert_shared_to_regular_dict(shared_dict):
    """Convert a shared dictionary structure to regular dictionaries for saving"""
    regular_dict = dict(shared_dict)
    for key, value in shared_dict.items():
        if isinstance(value, DictProxy):
            regular_dict[key] = convert_shared_to_regular_dict(value)
        elif isinstance(value, ListProxy):
            regular_dict[key] = list(value)
        else:
            regular_dict[key] = value
    return regular_dict


def convert_to_shared_dict(regular_dict, manager):
    """Convert a regular dictionary to shared dictionary structure"""
    shared_dict = manager.dict()
    for key, value in regular_dict.items():
        if isinstance(value, dict):
            if key == "children":
                shared_dict[key] = convert_to_shared_dict(value, manager)
            else:
                shared_dict[key] = manager.dict()
                for k, v in value.items():
                    if isinstance(v, dict):
                        shared_dict[key][k] = convert_to_shared_dict(v, manager)
                    else:
                        shared_dict[key][k] = v
        else:
            shared_dict[key] = value
    return shared_dict


def search_for_common_prefix(tree: Dict, model: str, max_prefix_length: int=15, freq_threshold: int=50,
                             constraint: int = None):
    """Search for common prefixes in the tree and return a list of common prefixes"""

    def _search_common_prefixes(nodes, prefix, common_prefixes):
        if len(prefix) == max_prefix_length:
            return
        for token in nodes:
            if nodes[token]["count"] >= freq_threshold:
                _prefix_key = ",".join([str(x) for x in prefix + [token]])
                common_prefixes[_prefix_key] += nodes[token]["count"]
                _search_common_prefixes(nodes[token]["children"], prefix + [token], common_prefixes)

    common_prefixes = Counter()
    model_name = os.path.basename(model)
    all_keys = []
    if constraint is not None:
        all_keys.append(f"{constraint}_{model_name}")
    else:
        all_keys = [k for k in tree.keys() if model_name in k]

    for key in all_keys:
        if key not in tree:
            continue
        _search_common_prefixes(tree[key]['tree'], [], common_prefixes)

    items = list(common_prefixes.items())
    # Sort items first by length (number of tokens in prefix), then by frequency
    items.sort(key=lambda x: (-len(x[0].split(',')), -x[1]))
    items = [[[int(y) for y in x[0].split(",")], x[1]] for x in items]

    return items


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="response_mmlu_256", help="Root directory for source files")
    parser.add_argument("--save_interval", type=int, default=DEFAULT_SAVE_INTERVAL, help="Interval for saving state")
    parser.add_argument("--max_prefix_length", type=int, default=20, help="Maximum prefix length for tree")
    parser.add_argument("--output_root_dir", type=str, default="mmlu_nudging_prompts_prefix_tree", help="Output root directory")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Maximum number of workers")
    parser.add_argument("--memory_threshold", type=float, default=DEFAULT_MEMORY_THRESHOLD, help="Memory threshold for waiting")
    parser.add_argument("--constraints", type=str, default="1,2,3,4,5", help="Constraint levels to process")
    args = parser.parse_args()
    output_root_dir = args.output_root_dir
    os.makedirs(output_root_dir, exist_ok=True)
    root_dir = args.root_dir
    max_prefix_length = args.max_prefix_length

    # Setup logging
    logger.add(os.path.join(output_root_dir, f"{root_dir}_{max_prefix_length}.log"))

    # Initialize StoreManager
    store = StoreManager(
        base_dir=output_root_dir,
        temp_dir=os.path.join(output_root_dir, "temp")
    )
    constraints = [int(x) for x in args.constraints.split(",")]

    try:
        models = [os.path.basename(x) for x in ALL_MODELS
                  if "llama" in x.lower() and ("chat" in x.lower() or "instruct" in x.lower())]

        source_root_dir = f"{root_dir}" + "/application_ctrlgen_multi_constraints_{}"
        state_name = f"{root_dir}_prefix_tree_max_prefix_length_{max_prefix_length}.pt"
        already_processed_model_constraints = set()

        with Manager() as manager:
            manager_dict = manager.dict()

            # Load existing data if available
            if store.exists(state_name):
                logger.info(f"Found existing {state_name}, loading...")
                try:
                    existing_data = store.load(state_name)
                    # Convert regular dict to shared dict structure
                    manager_dict.update(convert_to_shared_dict(existing_data, manager))
                    already_processed_model_constraints = set([x for x in manager_dict.keys() if len(manager_dict[x]["processed_files"]) > 0])
                except Exception as e:
                    logger.error(f"Error loading existing data: {str(e)}")
                    logger.info("Starting from scratch...")

            # Process models and constraints
            total_tasks = len(models) * 5
            with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
                for constraint in constraints:
                    source_dir = source_root_dir.format(constraint)
                    for model in models:
                        manager_dict_key = f"{constraint}_{model}"
                        if manager_dict_key in already_processed_model_constraints:
                            logger.info(f"Skipping {manager_dict_key} as it has already been processed")
                            pbar.update(1)
                            continue
                        process_model_constraint(
                            model,
                            constraint,
                            source_dir,
                            manager_dict,
                            store,
                            state_name,
                            args,
                            manager
                        )
                        pbar.update(1)

                        # Save after each model-constraint combination
                        state = convert_shared_to_regular_dict(manager_dict)
                        store.save(state, state_name)

            # Final save
            logger.info("Start Saving the result dict...")
            state = convert_shared_to_regular_dict(manager_dict)
            logger.info("Converted shared dict to regular dict...")
            store.save(state, state_name)
            logger.info("Saved the result dict to {}".format(state_name))

    finally:
        pass


if __name__ == '__main__':
    main()

