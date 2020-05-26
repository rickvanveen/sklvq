"""
=====================================
Local Generalized Matrix LVQ (LGMLVQ)
=====================================

Example of how to use LGMLVQ.
"""
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

from sklvq import LGMLVQ

data, labels = load_iris(return_X_y=True)

model = LGMLVQ(
    localization="p",
    distance_type="local-adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="lbfgs",
)

# Train the model using the iris dataset
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))