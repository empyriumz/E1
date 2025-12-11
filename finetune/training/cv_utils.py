import logging
import datetime
import torch.distributed as dist
from pathlib import Path
from training.distributed_utils import is_main_process, is_distributed, get_world_size

logger = logging.getLogger(__name__)


def run_cv(conf, train_fn, base_output_path: str, log_summary_fn=None):
    """
    Run K-fold cross-validation.

    Args:
        conf: Configuration object
        train_fn: Training function that takes (conf, fold_idx, base_output_path) and returns fold_results
        base_output_path: Base output path
        log_summary_fn: Optional function to log the summary of all fold results

    Returns:
        List of fold results
    """
    if is_main_process():
        logging.info(
            f"K-fold cross-validation training begins at {datetime.datetime.now().strftime('%m-%d %H:%M')}"
        )

    num_folds = getattr(conf.training, "num_folds", 5)
    if is_main_process():
        logging.info(f"Running {num_folds}-fold cross-validation")
        if is_distributed():
            logging.info(f"  - Distributed training with {get_world_size()} GPUs")

    Path(base_output_path).mkdir(parents=True, exist_ok=True)

    all_fold_results = []

    for fold_idx in range(1, num_folds + 1):
        if is_main_process():
            logging.info(f"\n{'='*120}")
            logging.info(f"Starting Fold {fold_idx}/{num_folds}")
            logging.info(f"{'='*120}")

        fold_results = train_fn(conf, fold_idx, base_output_path)
        all_fold_results.append(fold_results)

        # Synchronize across processes before next fold
        if is_distributed():
            dist.barrier()

    # Log summary (only on main process)
    if is_main_process() and log_summary_fn is not None:
        log_summary_fn(all_fold_results, conf)

        logging.info(
            f"\nK-fold cross-validation completed at {datetime.datetime.now().strftime('%m-%d %H:%M')}"
        )

    return all_fold_results
