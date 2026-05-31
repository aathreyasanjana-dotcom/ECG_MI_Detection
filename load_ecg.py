import wfdb
import matplotlib.pyplot as plt

# Load ECG record
record = wfdb.rdrecord(
    'mit-bih-arrhythmia-database-1.0.0/100'
)

# Load annotations
annotation = wfdb.rdann(
    'mit-bih-arrhythmia-database-1.0.0/100',
    'atr'
)

# Get first ECG channel
signal = record.p_signal[:, 0]

# Create plot
plt.figure(figsize=(15,4))

# Plot ECG signal
plt.plot(signal[:3000], label='ECG Signal')

# Plot annotation markers
for sample in annotation.sample:
    if sample < 3000:
        plt.axvline(x=sample, color='red', alpha=0.5)

plt.title("MIT-BIH ECG with Beat Annotations")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.show()

# Print first 20 beat labels
print("First 20 Beat Labels:")
print(annotation.symbol[:20])