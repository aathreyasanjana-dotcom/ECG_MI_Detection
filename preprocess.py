import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

BASE_PATH = os.path.expanduser(
    "~/Downloads/ptb-diagnostic-ecg-database-1.0.0"
)

WINDOW_SIZE = 3000

X = []
y = []
patient_ids = []

def bandpass_filter(signal, low=0.5, high=40, fs=1000, order=4):

    nyquist = fs / 2

    low_cut = low / nyquist
    high_cut = high / nyquist

    b, a = butter(
        order,
        [low_cut, high_cut],
        btype="band"
    )

    return filtfilt(b, a, signal, axis=0)

patients = sorted([
    p for p in os.listdir(BASE_PATH)
    if p.startswith("patient")
])

print("Patients found:", len(patients))

for patient in patients:

    patient_path = os.path.join(
        BASE_PATH,
        patient
    )

    records = []

    for file in os.listdir(patient_path):

        if file.endswith(".hea"):

            records.append(
                file.replace(".hea", "")
            )

    if len(records) == 0:
        continue

    try:

        first_record = os.path.join(
            patient_path,
            records[0]
        )

        header = wfdb.rdheader(first_record)

        comments = " ".join(
            header.comments
        ).lower()

        if "myocardial infarction" in comments:

            label = 1

        elif "healthy control" in comments:

            label = 0

        else:

            continue

    except:

        continue

    for rec in records:

        try:

            record_path = os.path.join(
                patient_path,
                rec
            )

            record = wfdb.rdrecord(
                record_path
            )

            # 12 standard ECG leads
            signal = record.p_signal[:, :12]

            signal = bandpass_filter(signal)

            scaler = StandardScaler()

            signal = scaler.fit_transform(signal)

            for start in range(
                0,
                len(signal) - WINDOW_SIZE,
                WINDOW_SIZE
            ):

                segment = signal[
                    start:start + WINDOW_SIZE
                ]

                X.append(segment)
                y.append(label)
                patient_ids.append(patient)

        except Exception as e:

            print("Skipped:", rec)

X = np.array(X, dtype=np.float32)
y = np.array(y)
patient_ids = np.array(patient_ids)

print("\nSegments:", len(X))

print("\nDataset Ready")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Healthy:", np.sum(y == 0))
print("MI:", np.sum(y == 1))

np.save("X.npy", X)
np.save("y.npy", y)
np.save("patient_ids.npy", patient_ids)

print("\nFiles Saved")