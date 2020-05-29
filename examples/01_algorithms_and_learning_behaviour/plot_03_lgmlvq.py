"""
=====================================
Local Generalized Matrix LVQ (LGMLVQ)
=====================================

Example of how to use LGMLVQ on the classic iris dataset.
"""
import numpy as np


import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklvq import LGMLVQ

iris = load_iris()

data = iris.data
labels = iris.target

###############################################################################
# Fitting the Model
# .................
# Create a LGMLVQ object and pass it a distance function, activation function and solver. See the API reference
# under documentation for defaults.

# Object to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

# Initialize GMLVQ object
model = LGMLVQ(
    localization="p",
    distance_type="local-adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="lbfgs",
)

# Train the model using the scaled data and true labels
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))

###############################################################################
# Plotting the Relevance Matrices
# ...............................
# GMLVQ learns a "relevance matrix" which can tell us something about which features are most relevant for
# the classification.

colors = ["blue", "red", "green"]
num_prototypes = model.prototypes_.shape[0]
num_features = model.prototypes_.shape[1]

fig, ax = plt.subplots(num_prototypes, 1)
fig.suptitle("Relevance Diagnoal of each Prototype")

for i, omega in enumerate(model.omega_):
    ax[i].bar(
        range(num_features),
        np.diagonal(omega.T.dot(omega)),
        color=colors[i],
        label=iris.target_names[model.prototypes_labels_[i]],
    )
    ax[i].set_xticks(range(num_features))
    if i == (num_prototypes - 1):
        ax[i].set_xticklabels([name[:-5] for name in iris.feature_names])
    else:
        ax[i].set_xticklabels([], visible=False)
        ax[i].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
    ax[i].set_ylabel("Weight")
    ax[i].legend()


###############################################################################
# Transforming the Data
# .....................
# In addition to making predictions GMLVQ can transform the data using the eigenvectors of the relevance matrix.

# Coming soon...
