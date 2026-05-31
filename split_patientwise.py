import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")
patient_ids = np.load(
    "patient_ids.npy",
    allow_pickle=True
)

patients = np.unique(patient_ids)

print("Total patients:", len(patients))

trainval_patients, test_patients = train_test_split(
    patients,
    test_size=0.20,
    random_state=42
)

train_patients, val_patients = train_test_split(
    trainval_patients,
    test_size=0.20,
    random_state=42
)

train_mask = np.isin(
    patient_ids,
    train_patients
)

val_mask = np.isin(
    patient_ids,
    val_patients
)

test_mask = np.isin(
    patient_ids,
    test_patients
)

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\nTrain:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\nSaved")