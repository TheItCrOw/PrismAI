import torch
import warnings

def get_best_device():
    """
    Returns cpu device if no cuda is available, otherwise returns the device with the most memory left.
    """

    if not torch.cuda.is_available():
        return torch.device('cpu')

    best_device = None
    max_free_mem = 0

    for i in range(torch.cuda.device_count()):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(i)

            if free_mem > max_free_mem:
                max_free_mem = free_mem
                best_device = i

        except RuntimeError as e:
            warnings.warn(f"Skipping cuda:{i} due to error: {e}")
            continue

    if best_device is not None:
        return torch.device(f'cuda:{best_device}')
    else:
        warnings.warn("All CUDA devices failed memory check; falling back to CPU.")
        return torch.device('cpu')
