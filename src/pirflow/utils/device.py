# SPDX-License-Identifier: MIT
"""Select device for PIRFlow."""

import torch

def select_device(device_str):
    """Return a PyTorch device with CPU fallback for unavailable CUDA or MPS.

    Parameters
    ----------
    device_str : str or None
        Device name, such as ``"cpu"``, ``"cuda"``, or ``"mps"``.
        If ``None``, choose CUDA when available, otherwise CPU.

    Returns
    -------
    torch.device
        Selected device, falling back to CPU if requested CUDA or MPS
        is unavailable.
    """
    
    if device_str is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
        device = torch.device(device_str)
    else:
        if "cuda" in device_str and not torch.cuda.is_available():
            print("---------------------------------------")
            print("CUDA requested but not available. Falling back to CPU")
            device = torch.device("cpu")
            device_str = "cpu"
        elif "mps" in device_str and not torch.backends.mps.is_available():
            print("---------------------------------------")
            print("MPS requested but not available. Falling back to CPU")
            device = torch.device("cpu")
            device_str = "cpu"
        else:
            device = torch.device(device_str)

    return device
