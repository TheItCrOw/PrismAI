import torch

def get_best_device():
    """
    Returns cpu device if no cuda is available, otherwise returns the device with the most memory left.
    """

    if not torch.cuda.is_available():
        return torch.device('cpu')

    best_device = None
    max_free_mem = 0

    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        stats = torch.cuda.mem_get_info(i)  # (free, total)
        free_mem = stats[0]

        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_device = torch.device(f'cuda:{i}')

    return best_device or torch.device('cpu')
