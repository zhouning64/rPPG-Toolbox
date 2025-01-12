import numpy as np
import torch
import os

torch.set_num_threads(os.cpu_count())
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def resample_ppg(input_signal, target_length):
    """Resamples a PPG sequence to a specific length."""
    return np.interp(
        np.linspace(1, input_signal.shape[0], target_length),
        np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
        input_signal
    )


# Original PPG signal (example)
original_signal = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

# Desired length
desired_length = 10

# Resample the signal
resampled_signal = resample_ppg(original_signal, desired_length)

print(resampled_signal)