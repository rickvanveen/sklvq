"""
======================
Generalized LVQ (GLVQ)
======================

Example of how to use the GLVQ algorithm on the classic iris dataset.
"""

from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklvq import GLVQ

data, labels = load_iris(return_X_y=True)

###############################################################################
# Create a GLVQ object and pass it a distance function, activation function and solver. See the API reference
# under documentation for defaults.

model = GLVQ(
    distance_type="squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="steepest-gradient-descent",
    solver_params={"max_runs": 20, "step_size": 0.1},
)

###############################################################################
# Fit the GLVQ object to the data and print the performance

# Train the model using the iris dataset
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))
