import numpy as np
from sklearn.model_selection import train_test_split

# Load data
X = np.load("X.npy")
y = np.load("y.npy")
patient_ids = np.load("patient_ids.npy", allow_pickle=True)

print("Total patients:", len(np.unique(patient_ids)))

# Get unique patients
unique_patients = np.unique(patient_ids)

# Split patients (80% train, 20% test)
train_patients, test_patients = train_test_split(
    unique_patients,
    test_size=0.2,
    random_state=42
)

# Create masks
train_mask = np.isin(patient_ids, train_patients)
test_mask = np.isin(patient_ids, test_patients)

# Create datasets
X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nTrain set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Save
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\nSaved:")
print("X_train.npy")
print("y_train.npy")
print("X_test.npy")
print("y_test.npy")
