import os
import sys
import random
import numpy as np
import torch
import logging
import re
from datetime import datetime
import yaml
from pathlib import Path
from ml_collections import config_dict
from transformers import set_seed, TrainerCallback
from transformers.utils import logging as transformers_logging


# Module-level logger (will be configured in train function)
logger = logging.getLogger(__name__)

# Align HF caching for offline runs
DEFAULT_HF_HOME = "/hpcgpfs01/scratch/xdai/huggingface"
HF_HOME = os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
HF_CACHE_DIR = os.environ.setdefault("HF_CACHE_DIR", os.path.join(HF_HOME, "hub"))

if not os.path.isdir(HF_CACHE_DIR):
    logger.warning(
        "HF_CACHE_DIR '%s' does not exist. Please ensure offline weights are synced.",
        HF_CACHE_DIR,
    )


def _resolve_hf_cache_dir() -> Optional[str]:
    """
    Determine the HuggingFace cache directory to use when running in offline mode.

    Preference order:
        1. HF_CACHE_DIR (if set and exists)
        2. HF_HOME/hub (if HF_HOME is set and the hub subfolder exists)
        3. HF_HOME (if set and exists)
    """
    env_cache_dir = os.environ.get("HF_CACHE_DIR")
    if env_cache_dir:
        env_cache_dir = os.path.expanduser(env_cache_dir)
        if os.path.isdir(env_cache_dir):
            return env_cache_dir

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_home = os.path.expanduser(hf_home)
        hf_home_hub = os.path.join(hf_home, "hub")
        if os.path.isdir(hf_home_hub):
            return hf_home_hub
        if os.path.isdir(hf_home):
            return hf_home

    return None


def _locate_offline_checkpoint(model_id: str) -> Optional[str]:
    """
    Locate a locally cached checkpoint following the HF_HOME/HF_CACHE_DIR layout.

    Args:
        model_id: HuggingFace model identifier (e.g., "Synthyra/Profluent-E1-600M")

    Returns:
        Path to cached checkpoint directory, or None if not found
    """
    direct_path = os.path.join(HF_CACHE_DIR, model_id)
    if os.path.isdir(direct_path):
        return direct_path

    repo_dir = os.path.join(HF_CACHE_DIR, f"models--{model_id.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    refs_main = os.path.join(repo_dir, "refs", "main")

    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as ref_file:
            ref = ref_file.read().strip()
        resolved = os.path.join(snapshots_dir, ref)
        if os.path.isdir(resolved):
            return resolved

    if os.path.isdir(snapshots_dir):
        snapshot_dirs = [
            os.path.join(snapshots_dir, d)
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if snapshot_dirs:
            snapshot_dirs.sort(key=os.path.getmtime, reverse=True)
            return snapshot_dirs[0]

    return None


class ClearCacheCallback(TrainerCallback):
    """Callback to clear GPU cache before and after evaluation to prevent OOM."""

    def __init__(self):
        self.prediction_step_count = 0
        self.logger = logging.getLogger(__name__)

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared before evaluation")
        self.prediction_step_count = 0  # Reset counter

    def on_prediction_step(self, args, state, control, **kwargs):
        # Clear cache periodically during prediction to prevent OOM
        self.prediction_step_count += 1
        if torch.cuda.is_available() and self.prediction_step_count % 100 == 0:
            torch.cuda.empty_cache()

    def on_save(self, args, state, control, **kwargs):
        # Clear cache after checkpoint saving (which happens after evaluation)
        # This ensures memory is freed before training resumes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared after checkpoint save")

    def on_train_begin(self, args, state, control, **kwargs):
        # Clear cache at the start of training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared at training start")


class MetricRenameCallback(TrainerCallback):
    """Callback to rename metrics by removing 'Combined' prefix from source-specific metrics."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Rename eval_Combined_Homologs_* to eval_Homologs_*
            # Rename eval_Combined_SwissProt_* to eval_SwissProt_*
            new_logs = {}
            for key, value in logs.items():
                if key.startswith("eval_Combined_Homologs_"):
                    new_key = key.replace("eval_Combined_Homologs_", "eval_Homologs_")
                    new_logs[new_key] = value
                elif key.startswith("eval_Combined_SwissProt_"):
                    new_key = key.replace("eval_Combined_SwissProt_", "eval_SwissProt_")
                    new_logs[new_key] = value
                else:
                    new_logs[key] = value

            # Update logs in place
            logs.clear()
            logs.update(new_logs)


class CompileFlexAttentionForEvalCallback(TrainerCallback):
    """
    Informational callback for evaluation.

    Note: torch.compile for flex_attention doesn't work within the HuggingFace Trainer
    because PyTorch's internal flex_attention dispatch still falls back to dense O(n²)
    attention. This callback now just logs reminders about proper evaluation settings.

    For accurate evaluation metrics, use evaluate_original_model_hf.py after training.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._logged_once = False

    def on_evaluate(self, args, state, control, **kwargs):
        """Log reminder about evaluation settings."""
        if not self._logged_once:
            self.logger.info(
                "Note: Training evaluation uses reduced MSA parameters to avoid OOM. "
                "For final metrics comparable to standalone evaluation, run "
                "evaluate_original_model_hf.py separately after training."
            )
            self._logged_once = True


class MSADatasetEpochCallback(TrainerCallback):
    """
    Callback to update MSA dataset seeds at the start of each epoch.

    This enables dynamic MSA sampling during training, where each epoch samples
    different context sequences from MSAs, effectively augmenting the training data.
    Works with both E1MSADataset and ConcatE1MSADataset.
    """

    def __init__(self, train_dataset):
        """
        Initialize the callback with the training dataset.

        Args:
            train_dataset: The training dataset (ConcatE1MSADataset or E1MSADataset)
                          that supports set_epoch() method.
        """
        self.train_dataset = train_dataset
        self.logger = logging.getLogger(__name__)
        self._last_epoch = -1

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training epoch."""
        epoch = state.epoch if state.epoch is not None else 0
        epoch_int = int(epoch)

        # Avoid updating twice for the same epoch
        if epoch_int == self._last_epoch:
            return

        self._last_epoch = epoch_int

        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch_int)
            self.logger.info(
                f"Updated MSA dataset for epoch {epoch_int} (new MSA sampling)"
            )


def set_seeds(s: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def setup_logging(output_dir: str):
    """
    Set up logging to both console and file.
    Uses transformers logging API and captures Trainer metrics while filtering progress bars.

    Args:
        output_dir: Directory where log file will be saved
    """
    # Create log file path
    log_file = os.path.join(output_dir, "training.log")

    # Open log file for writing (append mode)
    log_file_handle = open(log_file, "a", encoding="utf-8")

    # Save original stdout/stderr before redirecting
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create file handler for our log file
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Create console handler - use original stdout to avoid circular writes after redirect
    console_handler = logging.StreamHandler(original_stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Use transformers logging API to add file handler to transformers logger
    # This ensures Trainer's logging goes to our file
    transformers_logging.set_verbosity_info()
    transformers_logging.add_handler(file_handler)

    # Redirect stdout and stderr to capture Trainer metrics (but filter progress bars)
    class TeeOutput:
        def __init__(self, original_stream, log_file_handle):
            self.original_stream = original_stream
            self.log_file_handle = log_file_handle
            self.buffer = ""

        def _is_progress_bar(self, line):
            """Check if a line is a progress bar (tqdm output)."""
            line_stripped = line.strip()
            if not line_stripped:
                return True  # Empty lines are likely progress bar updates

            # Check for common progress bar patterns
            progress_patterns = [
                r"^\s*\d+%",  # Starts with percentage
                r"\d+/\d+\s*\[",  # Fraction format like "100/200 ["
                r"<\d+:\d+",  # Time estimate like "<00:10"
                r"\|+",  # Multiple pipe characters
                r"█+",  # Block characters
                r"it/s",  # "iterations per second"
                r"s/it",  # "seconds per iteration"
            ]
            for pattern in progress_patterns:
                if re.search(pattern, line_stripped):
                    return True
            return False

        def write(self, data):
            # Always write to original stream (console) - preserves progress bars
            self.original_stream.write(data)
            self.original_stream.flush()

            # Buffer the data
            self.buffer += data

            # Process complete lines
            while "\n" in self.buffer or "\r" in self.buffer:
                # Handle both \n and \r (progress bars use \r)
                if "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                else:
                    # Handle \r (progress bar update)
                    line, self.buffer = self.buffer.split("\r", 1)

                # Clean the line (remove \r if present)
                line_clean = line.replace("\r", "").strip()

                # Only log non-empty lines that are not progress bars
                if line_clean and not self._is_progress_bar(line_clean):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_file_handle.write(f"{timestamp} - STDOUT - {line_clean}\n")
                    self.log_file_handle.flush()

        def flush(self):
            self.original_stream.flush()
            self.log_file_handle.flush()

    # Replace stdout and stderr
    sys.stdout = TeeOutput(original_stdout, log_file_handle)
    sys.stderr = TeeOutput(original_stderr, log_file_handle)

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info("Capturing Trainer metrics to log file (progress bars filtered)")
    return logger


def save_model(model, filepath: str):
    """Saves only the finetuned parameters (LoRA weights)."""
    non_frozen_params = {}
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            non_frozen_params[param_name] = param
    torch.save(non_frozen_params, filepath)
    logger.info(f"Model saved to {filepath}")


def process_config(conf_path, config_name="train"):
    """
    Load YAML configuration and setup output path.

    Parameters:
    - conf_path: the path to the YAML configuration file.
    - config_name: The base name of the configuration file, used for directory structuring.

    Returns:
    - A tuple of the configuration dictionary and the output path as a Path object (if created).
    """
    with open(conf_path, "r") as f:
        # Load only the first YAML document (config), ignore subsequent documents (notes)
        documents = list(yaml.safe_load_all(f))
        conf = documents[0] if documents else {}

    output_path = None

    # Setup the output directory if not in debug mode
    if not conf.get("general", {}).get("debug", False):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        # Use output_dir from training config if available, otherwise use default structure
        base_output_dir = conf.get("training", {}).get(
            "output_dir", f"results/{config_name}"
        )
        output_path = Path(base_output_dir) / Path(timestamp)
        output_path.mkdir(parents=True, exist_ok=True)
        conf["output_path"] = "./" + str(output_path)
        # Save config as YAML
        with open(str(output_path) + "/config.yaml", "w") as f:
            yaml.dump(conf, f, default_flow_style=False)

    # Wrap the configuration dictionary in a custom dictionary class if used
    conf = config_dict.ConfigDict(conf)

    return conf, output_path
