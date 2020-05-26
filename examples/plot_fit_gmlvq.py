"""
==============================
Generalized Matrix LVQ (GMLVQ)
==============================

Example of how to use GMLVQ on the classic iris dataset.
"""

import numpy as np


import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


from sklvq import GMLVQ

iris = load_iris()

data = iris.data
labels = iris.target

###############################################################################
# Predictor
# .........
# Create a GMLVQ object and pass it a distance function, activation function and solver. See the API reference
# under documentation for defaults.

# Object to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

# Initialize GMLVQ object
model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
)

# Train the model using the scaled data and true labels
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))


###############################################################################
# Relevance Matrix
# ................
# GMLVQ learns a "relevance matrix" which can tell us something about which features are most relevant for
# the classification (most discriminative).

# The relevance matrix is computed as follows from the models parameter "omega_"
relevance_matrix = model.omega_.T.dot(model.omega_)

# Plot the diagonal of the relevance matrix
plt.bar(iris.feature_names, np.diagonal(relevance_matrix))
plt.grid(False)


###############################################################################
# Transformer
# ...........
# In addition to making predictions GMLVQ can transform the data using the eigenvectors of the relevance matrix.

# Transform the data
transformed_data = model.transform(data, scale=True)

x_d = transformed_data[:, 0]
y_d = transformed_data[:, 1]

# Transform the model (prototypes)
transformed_model = model.transform(model.prototypes_, scale=True)

x_m = transformed_model[:, 0]
y_m = transformed_model[:, 1]

# Plot
plt.title("Discriminative projection Iris data and GMLVQ prototypes")
plt.scatter(x_d, y_d, c=labels, s=100, alpha=0.8, edgecolors="white")
plt.scatter(x_m, y_m, c=model.prototypes_labels_, s=150, alpha=0.8, edgecolors="black")
plt.xlabel("First eigenvector of the relevance matrix")
plt.ylabel("Second eigenvector of the relevance matrix")

plt.grid(True)
