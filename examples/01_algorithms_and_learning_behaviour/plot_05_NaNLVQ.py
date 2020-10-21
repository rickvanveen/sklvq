"""
=========================
Not a Number LVQ (NaNLVQ)
=========================

An extension to the LVQ algorithms that provides a number of distance functions that can deal
with missing values.
"""

import matplotlib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklvq import GMLVQ

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

iris = load_iris()

data = iris.data
labels = iris.target

# Insert some random missingvalues
num_missing_values = 50
num_samples, num_dimensions = data.shape

i = np.random.choice(num_samples, num_missing_values, replace=False)
j = np.random.choice(num_dimensions, num_missing_values, replace=True)

data[i, j] = np.nan


model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
    force_all_finite="allow-nan",  # This will make the checks and distance function accept and
    # deal with np.nan values.
)

###############################################################################
# Fit the GLVQ object to the X and print the performance

# Object to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

# Train the model using the scaled X and true labels
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))
