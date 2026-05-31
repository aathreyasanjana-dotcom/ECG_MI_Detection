ECG Myocardial Infarction Detection using CNN-BiLSTM

Overview

This project focuses on the automated detection of Myocardial Infarction (MI) from ECG signals using a hybrid CNN-BiLSTM deep learning architecture.

The model is trained on the PTB Diagnostic ECG Database and aims to identify ECG patterns associated with myocardial infarction while maintaining high sensitivity to reduce false negatives.

Objectives
	•	Preprocess and segment ECG signals
	•	Train a CNN-BiLSTM model for MI classification
	•	Evaluate performance using medically relevant metrics
	•	Analyze model behavior and misclassifications
	•	Explore methods to minimize false negatives

Dataset

Dataset: PTB Diagnostic ECG Database

The dataset contains 12-lead ECG recordings from healthy controls and patients with various cardiac conditions, including myocardial infarction.

Note: Dataset files are not included in this repository because of licensing and storage limitations.

Project Structure

ECG_Project/

├── load_ecg.py

├── preprocess.py

├── split_data.py

├── split_patientwise.py

├── train.py

├── evaluate.py

├── visualize.py

└── README.md

Current Progress

Completed
	•	ECG data loading pipeline
	•	Signal preprocessing
	•	Dataset preparation
	•	Training pipeline setup
	•	Initial CNN-BiLSTM implementation

In Progress
	•	Model optimization
	•	Performance evaluation
	•	Sensitivity and specificity analysis

Planned Improvements
	•	ECG denoising pipeline
	•	Explainable AI techniques
	•	Comparison with alternative architectures
	•	Real-time inference interface

Technologies Used
	•	Python
	•	TensorFlow / Keras
	•	NumPy
	•	Pandas
	•	Matplotlib

Future Work
	•	Improve recall for myocardial infarction detection
	•	Reduce false negatives
	•	Compare performance across different ECG leads
	•	Explore lightweight deployment options

