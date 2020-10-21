"""
=====================================
Local Generalized Matrix LVQ (LGMLVQ)
=====================================

Example of how to use LGMLVQ on the classic iris dataset.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklvq import LGMLVQ

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

iris = load_iris()

data = iris.data
labels = iris.target

###############################################################################
# Fitting the Model
# .................
# Create a LGMLVQ object and pass it a distance function, activation function and solver.
# See the API reference under documentation for defaults.

# Object to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

# Initialize GMLVQ object
model = LGMLVQ(
    relevance_params={"localization": "class"}, # Can either be per "class" or "prototype"
    distance_type="local-adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="lbfgs",
)

# Train the model using the scaled X and true labels
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))

###############################################################################
# Extracting the Relevance Matrices
# .................................
# GMLVQ learns a "relevance matrix" which can tell us something about which features
# are most relevant for the classification.

colors = ["blue", "red", "green"]
num_prototypes = model.prototypes_.shape[0]
num_features = model.prototypes_.shape[1]

fig, ax = plt.subplots(num_prototypes, 1)
fig.suptitle("Relevance Diagonal of each Prototype's lambda matrix")

for i, lambda_ in enumerate(model.lambda_):
    ax[i].bar(
        range(num_features),
        np.diagonal(lambda_),
        color=colors[i],
        label=iris.target_names[model.prototypes_labels_[i]],
    )
    ax[i].set_xticks(range(num_features))
    if i == (num_prototypes - 1):
        ax[i].set_xticklabels([name[:-5] for name in iris.feature_names])
    else:
        ax[i].set_xticklabels([], visible=False)
        ax[i].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
    ax[i].set_ylabel("Weight")
    ax[i].legend()


###############################################################################
# Transforming the Data
# .....................
# In addition to making predictions GMLVQ can transform the data using the eigenvectors of the
# relevance matrix. And we can do this for every relevance matrix attached to the prototypes.

# This will return a 3D shape with the first axis the different relevance matrices'
# transformations of the data. The 2nd and 3rd axes represent the data within in this new space.
t_d = model.transform(data, omega_hat_index=[0, 1, 2])[:, :, :2]
t_m = model.transform(model.prototypes_, omega_hat_index=[0, 1, 2])[:, :, :2]

fig, ax = plt.subplots(num_prototypes, 1, figsize=(6.4, 14.4))
fig.tight_layout(pad=6.0)

colors = ["blue", "red", "green"]
for i, xy_dm in enumerate(zip(t_d, t_m)):
    xy_d = xy_dm[0]
    xy_m = xy_dm[1]
    for cls, j in enumerate(model.classes_):
        ii = cls == labels
        ax[i].scatter(
            xy_d[ii, 0],
            xy_d[ii, 1],
            c=colors[j],
            s=100,
            alpha=0.7,
            edgecolors="white",
            label=iris.target_names[model.prototypes_labels_[j]],
        )
    ax[i].scatter(
        xy_m[:, 0],
        xy_m[:, 1],
        c=colors,
        s=180,
        alpha=0.8,
        edgecolors="black",
        linewidth=2.0,
    )
    ax[i].title.set_text(
        "Relevance projection w.r.t. {}".format(
            iris.target_names[model.prototypes_labels_[i]]
        )
    )
    ax[i].set_xlabel("First eigenvector")
    ax[i].set_ylabel("Second eigenvector")
    ax[i].legend()
    ax[i].grid(True)
