import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Healthy:", np.sum(y == 0))
print("MI:", np.sum(y == 1))

print("\nFirst 10 labels:")
print(y[:10])