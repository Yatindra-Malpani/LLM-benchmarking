import torch

def get_gpu_memory():
    """
    Return current GPU memory usage in GB.

    Returns:
        allocated_gb: Memory currently occupied by tensors.
        reserved_gb: Memory reserved by PyTorch.
    """

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    allocated_gb = allocated / (1024 ** 3)
    reserved_gb = reserved / (1024 ** 3)

    return allocated_gb, reserved_gb

def get_peak_gpu_memory():
    """
    Return peak GPU memory allocated since the last reset.

    Returns:
        peak_gb: Peak allocated GPU memory in GB.
    """

    peak = torch.cuda.max_memory_allocated()
    return peak / (1024 ** 3)

def reset_gpu_peak_memory():
    """
    Reset PyTorch's peak GPU memory counter.
    """
    torch.cuda.reset_peak_memory_stats()
