import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------- Parameters --------------------

# Pulse rate and frequency
pulse_rate_bpm = 80  # beats per minute
pulse_freq_hz = pulse_rate_bpm / 60  # Hz

# Sampling parameters
sampling_rate = 100  # Hz
num_points = 800
duration = num_points / sampling_rate  # seconds

# Time vector
t = np.linspace(0, duration, num_points, endpoint=False)

# -------------------- Signal Generation --------------------

# Fundamental sine wave representing the pulse
fundamental = 1.0 * np.sin(2 * np.pi * pulse_freq_hz * t)

# Adding harmonics to mimic the complex nature of BVP signals
harmonic_1 = 0.5 * np.sin(2 * np.pi * 2 * pulse_freq_hz * t)  # Second harmonic
harmonic_2 = 0.25 * np.sin(2 * np.pi * 3 * pulse_freq_hz * t)  # Third harmonic

# Combined signal with harmonics
signal = fundamental + harmonic_1 + harmonic_2

# Adding Gaussian noise to simulate measurement noise
noise = 0.2 * np.random.normal(0, 1, num_points)
synthetic_bvp = signal + noise

# -------------------- Saving to CSV --------------------

# Combine time and BVP data
data = np.column_stack((t, synthetic_bvp))

# Define CSV file name
csv_filename = 'synthetic_bvp_signal.csv'

# Save to CSV
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Timestamp (s)', 'BVP'])
    # Write data rows
    writer.writerows(data)

print(f"Synthetic BVP data saved to '{csv_filename}'.")

# -------------------- Visualization --------------------

# Plot the synthetic BVP signal
plt.figure(figsize=(12, 6))
plt.plot(t, synthetic_bvp, label='Synthetic BVP Signal')
plt.title('Synthetic Blood Volume Pulse (BVP) Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()