"""
=========
Pipelines
=========

In these examples GMLVQ is used but the same applies to all the other algorithms.
Also the `pipelines`_ feature is provided by scikit-learn and we therefore refer to scikit-learn's
documentation  for more details.

.. _pipelines: https://scikit-learn.org/stable/modules/compose.html
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklvq import GMLVQ

data, labels = load_iris(return_X_y=True)

###############################################################################
# In previous examples we used a StandardScalar instance to process the data before fitting the
# model. Sklearn provides a very handy way of creating a connection between the scalar and the
# model called a pipeline. The pipeline can then be used and will first call the fit method of
# the standard scaler before the fit of the model. Now the data does not have to be scaled
# explicitly anymore.

# Create a scaler instance
scaler = StandardScaler()

# Create a GMLVQ model (or any other sklearn compatible estimator or other pre-processing method)
model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
)

# Link them together into a single object.
pipeline = make_pipeline(scaler, model)

# Fit the data to the pipeline. This will first call the scaler's  fit method before passing the
# result to  the model's fit function.
pipeline.fit(data, labels)

# Predict the labels using the trained pipeline. The pipeline will use the
# mean and standard deviation it found when fit was called and applies it to the data.
predicted_labels = pipeline.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))

###############################################################################
# When inspecting the resulting classifier and its prototypes,
# e.g., in a plot overlaid on a scatter plot of the data, don't forget to apply the scaling to the data:
transformed_data = pipeline.transform(data)
