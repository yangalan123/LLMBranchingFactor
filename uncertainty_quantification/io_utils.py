import os.path
import threading
import queue
import torch
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
from loguru import logger
from contextlib import contextmanager

from vllm.outputs import RequestOutput, CompletionOutput
try:
    from vllm.sequence import PromptLogprobs, SampleLogprobs
except ImportError:
    from vllm.logprobs import PromptLogprobs, SampleLogprobs

from uncertainty_quantification.loglik_computation import get_logprob_per_token_from_vllm_outputs


def _get_backup_path(filepath: Path) -> Path:
    """Get the backup path for a given file."""
    return filepath.parent / (filepath.name + '.backup')


class StoreManager:
    """
    A thread-safe storage manager for PyTorch data that handles safe reading and writing
    to NFS systems using temporary files and backup mechanisms.
    """

    def __init__(self, temp_dir: str = "./temp", base_dir: str = None, disable_columnar_storage: bool = False):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        if base_dir is not None:
            # in some cases, we might be bored with writing up a long prefix, and we are pretty sure we will only work under some specific directory
            # so we can set the base_dir here, and then we can just use the relative path to save/load data
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(exist_ok=True)
        else:
            self.base_dir = None
        self.disable_columnar_storage = disable_columnar_storage

        # Thread-safe lock for file operations
        self._lock = threading.Lock()
        # Queue for managing concurrent write operations
        self._write_queue = queue.Queue()
        # Thread for processing write operations
        self._write_thread = threading.Thread(target=self._process_write_queue, daemon=True)
        self._write_thread.start()
        # Track ongoing operations
        self._active_operations: Dict[str, threading.Event] = {}

    @contextmanager
    def _temp_file(self) -> Path:
        """Create a temporary file context manager."""
        temp_file = Path(tempfile.mktemp(dir=self.temp_dir, suffix='.pt'))
        try:
            yield temp_file
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def _process_write_queue(self):
        """Process queued write operations in a separate thread."""
        while True:
            try:
                data, filepath, done_event = self._write_queue.get()
                try:
                    self._safe_write(data, filepath)
                    done_event.set()
                except Exception as e:
                    logger.error(f"Error writing to {filepath}: {e}")
                    done_event.set()  # Set event even on failure to prevent hanging
                self._write_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)  # Prevent busy waiting

    def _safe_write(self, data: Any, filepath: Path):
        """Internal method to safely write data to disk."""
        # backup_path = filepath.with_suffix('.pt.backup')
        backup_path = _get_backup_path(filepath)

        with self._temp_file() as temp_file:
            # Save to temporary file first
            torch.save(data, temp_file)

            with self._lock:
                # Create backup if original exists
                if filepath.exists():
                    shutil.copy(filepath, backup_path)

                # Move temporary file to final destination
                shutil.copy(temp_file, filepath)

                # Remove backup if everything succeeded
                if backup_path.exists():
                    backup_path.unlink()

    def save(self, data: Any, save_path: str, async_write: bool = True) -> Optional[threading.Event]:
        """
        Save data to storage with the given name.

        Args:
            data: The data to save
            save_path: The complete relative/absolute path to save the data
            async_write: If True, write asynchronously and return an event that will be set when done

        Returns:
            Optional[threading.Event]: If async_write is True, returns an event that will be set when
                                     the write is complete. None otherwise.
        """
        if self.base_dir is not None:
            filepath = self.base_dir / save_path
        else:
            filepath = Path(save_path)

        if not self.disable_columnar_storage:
            try:
                storage_data = []
                for request_output in data:
                    logit_outputs = LogitOutputs(request_output)
                    storage_data.append(logit_outputs)
            except Exception as e:
                logger.error(f"Error converting data to columnar format: {e}, saving original data as-is")
                storage_data = data
        else:
            storage_data = data

        if async_write:
            done_event = threading.Event()
            self._active_operations[str(filepath)] = done_event
            self._write_queue.put((storage_data, filepath, done_event))
            return done_event
        else:
            self._safe_write(storage_data, filepath)
            return None

    def load(self, load_path: str, wait_for_pending: bool = True) -> Any:
        """
        Load data from storage by name.

        Args:
            load_path: path/name of the data to load
            wait_for_pending: If True, wait for any pending writes to complete before loading

        Returns:
            The loaded data
        """
        if self.base_dir is not None:
            filepath = self.base_dir / f"{load_path}"
        else:
            filepath = Path(load_path)
        backup_path = _get_backup_path(filepath)

        # Wait for any pending writes to this file
        if wait_for_pending:
            pending_event = self._active_operations.get(str(filepath))
            if pending_event:
                pending_event.wait()

        with self._lock:
            try:
                return torch.load(filepath)
            except (OSError, RuntimeError) as e:
                logger.warning(f"Error loading {filepath}, trying backup...")
                if backup_path.exists():
                    return torch.load(backup_path)
                raise e

    def exists(self, filepath: str) -> bool:
        """Check if data exists in storage."""
        if self.base_dir is not None:
            filepath = self.base_dir / f"{filepath}"
        else:
            filepath = Path(filepath)
        return filepath.exists()

    def delete(self, delete_path: str):
        """Delete data from storage."""
        if self.base_dir is not None:
            filepath = self.base_dir / f"{delete_path}"
        else:
            filepath = Path(delete_path)
        backup_path = _get_backup_path(filepath)

        with self._lock:
            if filepath.exists():
                filepath.unlink()
            if backup_path.exists():
                backup_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Wait for all pending operations to complete
        self._write_queue.join()
        # # Clean up temporary directory
        # shutil.rmtree(self.temp_dir)

class LogitOutputs:
    def __init__(self, request_output: RequestOutput):
        self.request_id = request_output.request_id
        self.prompt = request_output.prompt
        self.prompt_token_ids = request_output.prompt_token_ids
        self.storage = dict()
        self.storage['prompt_logprobs'] = None if request_output.prompt_logprobs is None else self.convert_vllm_logprobs_to_storage(request_output.prompt_logprobs)
        self.finished = request_output.finished
        self.metrics = request_output.metrics
        self.lora_request = request_output.lora_request
        self.storage["completion_outputs"] = self.process_vllm_completion_outputs(request_output.outputs)

    def process_vllm_completion_outputs(self, outputs: List[CompletionOutput]):
        ret = []
        for output in outputs:
            if output is None:
                ret.append(None)
            else:
                _new_logprobs = None if output.logprobs is None else self.convert_vllm_logprobs_to_storage(output.logprobs)
                new_completion_output = CompletionOutput(
                    index=output.index, text=output.text, token_ids=output.token_ids, logprobs=_new_logprobs, finish_reason=output.finish_reason, stop_reason=output.stop_reason,
                    lora_request=output.lora_request, cumulative_logprob=output.cumulative_logprob)
                ret.append(new_completion_output)
        return ret

    def convert_vllm_logprobs_to_storage(self, vllm_logprobs: Union[PromptLogprobs, SampleLogprobs]):
        """Convert VLLM log probabilities to columnar format and store in the storage."""
        ret = []
        for item_dict in vllm_logprobs:
            ret.append(self.convert_dict_to_columnar(item_dict) if item_dict is not None else None)
        return ret

    def convert_dict_to_columnar(self, data: Dict[int, Any]):
        """Convert a dictionary of data to columnar format."""
        keys = list(data.keys())
        values = [get_logprob_per_token_from_vllm_outputs(data[key]) for key in keys]
        return {
            'keys': torch.tensor(keys, dtype=torch.long),
            'values': torch.tensor(values, dtype=torch.float)
        }

    def convert_columnar_to_dict(self, columnar_data: Dict[str, torch.Tensor]):
        """Convert columnar data to a dictionary."""
        return dict(zip(columnar_data['keys'].tolist(), columnar_data['values'].tolist()))

    def convert_list_columnar_to_dict(self, columnar_data: List[Dict[str, torch.Tensor]]):
        """Convert list of columnar data to a list of dictionaries."""
        return [self.convert_columnar_to_dict(item) if item is not None else None for item in columnar_data]

    def convert_to_vllm_format(self):
        if self.storage['prompt_logprobs'] is not None:
            self.prompt_logprobs = self.convert_list_columnar_to_dict(self.storage['prompt_logprobs'])
        else:
            self.prompt_logprobs = None

        self.outputs = [
            CompletionOutput(
                index=output.index,
                text=output.text,
                token_ids=output.token_ids,
                logprobs=self.convert_list_columnar_to_dict(output.logprobs) if output.logprobs is not None else None,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
                lora_request=output.lora_request,
                cumulative_logprob=output.cumulative_logprob
            ) if output is not None else None
            for output in self.storage["completion_outputs"]
        ]

    def __getstate__(self):
        # Create a copy of the instance's dictionary
        state = self.__dict__.copy()

        # Remove the VLLM format attributes that we don't want to serialize
        state.pop('prompt_logprobs', None)
        state.pop('outputs', None)

        return state

    def __setstate__(self, state):
        # Validate that this is a LogitOutputs serialized data
        if 'storage' not in state:
            raise ValueError(
                "Invalid serialized data: 'storage' field missing. This might not be a LogitOutputs object.")

        if 'prompt_logprobs' in state or 'outputs' in state:
            raise ValueError("Invalid serialized data: Contains VLLM format fields ('prompt_logprobs' or 'outputs'). "
                             "This appears to be a RequestOutput rather than a LogitOutputs object.")

        # If validation passes, restore the instance's dictionary
        self.__dict__.update(state)

        # Convert storage format back to VLLM format
        self.convert_to_vllm_format()

