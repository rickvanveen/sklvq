"""
=====================================
Local Generalized Matrix LVQ (LGMLVQ)
=====================================

Example of how to use LGMLVQ `[1]`_ on the classic iris dataset.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklvq import LGMLVQ

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

# Contains also the target_names and feature_names, which we will use for the plots.
iris = load_iris()

data = iris.data
labels = iris.target

###############################################################################
# Fitting the Model
# .................
# Scale the data and create a LGMLVQ object with, e.g., custom distance function, activation
# function and solver. See the API reference under documentation for defaults and other
# possible parameters.

# Sklearn's standardscaler to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

# The creation of the model object used to fit the data to.
model = LGMLVQ(
    relevance_localization="class",  # Can either be "class" or "prototypes"
    distance_type="local-adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="lbfgs",
)

###############################################################################
# The next step is to fit the LGMLVQ object to the data and use the predict method to make the
# predictions. Note that this example only works on the training data and therefore does not say
# anything about the generalizability of the fitted model.

# Train the model using the scaled data and true labels
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# To get a sense of the training performance we could print the classification report.
print(classification_report(labels, predicted_labels))

###############################################################################
# Extracting the Relevance Matrices
# .................................
# In addition to the prototypes (see GLVQ example), LGMLVQ learns a number of matrices `lambda_`
# which can tell us something about which features are most relevant for the classification per
# class.  The  number of relevance  matrices is determined by the number of prototypes used per
# class as well as which localization strategy is used. It can either be a relevance matrix per
# class (even if there are more prototypes for that class they will share the relevance matrix.
# Or a relevance matrix per prototype, where each prototype (even if they have the same class)
# has its own matrix.

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
        ax[i].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax[i].set_ylabel("Weight")
    ax[i].legend()

###############################################################################
# Note that each diagonal still adds up to one (See GMLVQ example). However, each diagonal
# summarizes the importance of the features for its corresponding class versus all other classes.

###############################################################################
# Transforming the Data
# .....................
# In addition to making predictions LGMLVQ can be used to transform the data using the
# eigenvectors of the relevance matrices. In contrast to GMLVQ this can be done for each of the
# matrices separately.

# This will return a 3D shape with the 1st and 2nd axes representing the data (n_observations,
# n_eigenvectors). The third axis are the different relevance matrices

t_d = model.transform(data, omega_hat_index=[0, 1, 2])[:, :2, :]
t_d = np.transpose(t_d, axes=(2, 0, 1))  # shape that is easier to work with

t_m = model.transform(model.prototypes_, omega_hat_index=[0, 1, 2])[:, :2, :]
t_m = np.transpose(t_m, axes=(2, 0, 1))  # shape that is easier to work with

fig, ax = plt.subplots(num_prototypes, 1, figsize=(6.4, 14.4))
fig.tight_layout(pad=6.0)

colors = ["blue", "red", "green"]
for i, xy_dm in enumerate(zip(t_d, t_m)):
    xy_d = xy_dm[0]
    xy_m = xy_dm[1]
    for j, cls in enumerate(model.classes_):
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
    ax[i].title.set_text(f"Relevance projection w.r.t. {iris.target_names[model.prototypes_labels_[i]]}")
    ax[i].set_xlabel("First eigenvector")
    ax[i].set_ylabel("Second eigenvector")
    ax[i].legend()
    ax[i].grid(True)

###############################################################################
# References
# ..........
# _`[1]` Schneider, P., Biehl, M., & Hammer, B. (2009). "Adaptive Relevance Matrices in Learning
# Vector Quantization" Neural Computation, 21(12), 3532–3561, 2009.
