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
# Fitting the Model
# .................
# Create a GMLVQ object and pass it a distance function, activation function and solver. See the
# API reference under documentation for defaults.

model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
)

###############################################################################
# Fit the GLVQ object to the data and print the performance

# Object to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

# Train the model using the scaled data and true labels
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))


###############################################################################
# Extracting the Relevance Matrix
# ...............................
# In addition to the prototypes (see GLVQ example), GMLVQ learns a "relevance matrix" which can
# tell us something about which features are most relevant for the classification.

# The relevance matrix is computed as follows from the models parameter "omega_"
relevance_matrix = model.lambda_

# Plot the diagonal of the relevance matrix
fig, ax = plt.subplots()
fig.suptitle("Relevance Matrix Diagonal")
ax.bar([name[:-5] for name in iris.feature_names], np.diagonal(relevance_matrix))
ax.set_ylabel("Weight")
ax.grid(False)


###############################################################################
#   Note that the relevance diagonal adds up to one.

###############################################################################
# Transforming the data
# .....................
# In addition to making predictions GMLVQ can transform the data using the eigenvectors of the
# relevance matrix.

# Transform the data (scaled by square root of eigenvalues "scale = True")
transformed_data = model.transform(data, scale=True)

x_d = transformed_data[:, 0]
y_d = transformed_data[:, 1]

# Transform the model, i.e., the prototypes (scaled by square root of eigenvalues "scale = True")
transformed_model = model.transform(model.prototypes_, scale=True)

x_m = transformed_model[:, 0]
y_m = transformed_model[:, 1]

# Plot
fig, ax = plt.subplots()
fig.suptitle("Discriminative projection Iris data and GMLVQ prototypes")

colors = ["blue", "red", "green"]
for cls, i in enumerate(model.classes_):
    ii = cls == labels
    ax.scatter(
        x_d[ii],
        y_d[ii],
        c=colors[i],
        s=100,
        alpha=0.7,
        edgecolors="white",
        label=iris.target_names[model.prototypes_labels_[i]],
    )
ax.scatter(x_m, y_m, c=colors, s=180, alpha=0.8, edgecolors="black", linewidth=2.0)
ax.set_xlabel("First eigenvector")
ax.set_ylabel("Second eigenvector")
ax.legend()
ax.grid(True)
