import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load model
model = tf.keras.models.load_model(
    "mi_cnn_bilstm_12lead.keras"
)

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Predict
pred = model.predict(X_test)

pred = (pred > 0.5).astype(int)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))
