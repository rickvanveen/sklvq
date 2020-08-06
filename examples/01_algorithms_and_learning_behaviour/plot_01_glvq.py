"""
======================
Generalized LVQ (GLVQ)
======================

Example of how to use the GLVQ algorithm on the classic iris dataset.
"""
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklvq import GLVQ

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

iris = load_iris()
data = iris.data
labels = iris.target

###############################################################################
# Fitting the Model
# .......................

# Create a GLVQ object and pass it a distance function, activation function and solver. See the
# API reference under documentation for defaults and other possible parameters.

# Object to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

model = GLVQ(
    distance_type="squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="steepest-gradient-descent",
    solver_params={"max_runs": 20, "step_size": 0.1},
)

###############################################################################
# Fit the GLVQ object to the X and print the performance

# Train the model using the iris dataset
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))

###############################################################################
# Plotting the Prototypes
# .......................

# The GLVQ model produces prototypes as representations for the different classes. These
# prototypes can be accessed and, e.g., plotted for visual inspection.

colors = ["blue", "red", "green"]
num_prototypes = model.prototypes_.shape[0]
num_features = model.prototypes_.shape[1]

fig, ax = plt.subplots(num_prototypes, 1)
fig.suptitle("Prototype of each class")

for i, prototype in enumerate(model.prototypes_):
    # Reverse the z-transform to go back to the original feature space.
    prototype = scaler.inverse_transform(prototype)

    ax[i].bar(
        range(num_features),
        prototype,
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
    ax[i].set_ylabel("cm")
    ax[i].legend()
