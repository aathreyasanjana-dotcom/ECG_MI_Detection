import numpy as np
import matplotlib.pyplot as plt

X = np.load("X.npy")
y = np.load("y.npy")

for i in range(3):
    plt.figure(figsize=(12,4))
    plt.plot(X[i,:,0])
    plt.title(f"Segment {i}  Label={y[i]}")
    plt.show()