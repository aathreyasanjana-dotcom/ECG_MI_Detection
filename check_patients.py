import numpy as np

patients = np.load(
    "patient_ids.npy",
    allow_pickle=True
)

print("Total Segments:", len(patients))
print("Unique Patients:", len(np.unique(patients)))

print("\nFirst 20 IDs:")
print(patients[:20])