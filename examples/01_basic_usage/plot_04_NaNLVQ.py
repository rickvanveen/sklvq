"""
=========================
Not a Number LVQ (NaNLVQ)
=========================

NanLVQ `[1]`_ refers to a extension that can be implemented for various distance functions. It uses
the partial distance strategy to ignore any NaN values in the data. Another interpretation would be
that it imputes the missing values with those of the prototypes. Hence, the distance will
be zero, which results in a zero update for the feature containing the NaN value.

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

# Insert some "random" missing values represented by np.nan
num_missing_values = 50
num_samples, num_dimensions = data.shape

i = np.random.choice(num_samples, num_missing_values, replace=False)
j = np.random.choice(num_dimensions, num_missing_values, replace=True)

data[i, j] = np.nan

###############################################################################
# Fitting the Model
# .................
# Scale the data and create a GMLVQ object with, e.g., custom distance function, activation
# function and solver. See the API reference under documentation for defaults and other
# possible parameters.

# Object to perform z-transform
scaler = StandardScaler()

# Compute (fit) and apply (transform) z-transform
data = scaler.fit_transform(data)

# The creation of the model object used to fit the data to.
model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
    force_all_finite="allow-nan",  # This will make the data  validation  and distance function
    # accept and deal with np.nan values.
)

###############################################################################
# The next step is to fit the GMLVQ object to the data and use the predict method to make the
# predictions. Note that this example only works on the training data and therefor does not say
# anything about the generalizability of the fitted model.

# Train the model using the data and labels
model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# To get a sense of the training performance we could print the classification report.
print(classification_report(labels, predicted_labels))

###############################################################################
# The examples uses GMLVQ but all models and their compatible distance functions support the
# `force_all_finite` option.

###############################################################################
# References
# ..........
# _`[1]` Rick van Veen (2016). Analysis of Missing Data Imputation Applied to Heart Failure Data (
# Master's Thesis, University  of Groningen, Groningen, The Netherlands). Retrieved from
# http://fse.studenttheses.ub.rug.nl/id/eprint/14679