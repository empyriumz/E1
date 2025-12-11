import os
import torch
import torch.distributed as dist


def is_distributed():
    """Check if running in distributed mode."""
    return os.environ.get("RANK") is not None


def get_rank():
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return int(os.environ.get("RANK", 0))
    return 0


def get_local_rank():
    """Get local rank for device assignment."""
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def get_world_size():
    """Get total number of processes."""
    if is_distributed():
        return int(os.environ.get("WORLD_SIZE", 1))
    return 1


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


from datetime import timedelta


def setup_distributed():
    """Initialize distributed training environment."""
    if not is_distributed():
        return None

    # Initialize process group with configurable timeout
    timeout_env = os.environ.get("TORCH_DIST_TIMEOUT")
    if timeout_env:
        timeout = timedelta(seconds=int(timeout_env))
        dist.init_process_group(backend="nccl", timeout=timeout)
    else:
        dist.init_process_group(backend="nccl")

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()
