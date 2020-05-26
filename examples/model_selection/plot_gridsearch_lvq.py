"""
=====================================
Parameter Selection using Grid Search
=====================================
"""

###############################################################################
# First, imports and load digits dataset

import numpy as np

from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
)

from sklvq import GMLVQ

# Slightly more interesting dataset
data, labels = load_digits(return_X_y=True)

###############################################################################
# Preprocessing using Pipelines
# .............................
# Pipelines are used to string processing methods and predictors together. More details can be found in the manuals
# of scikit-learn

# Initialize the standardScaler (z-transform) object
standard_scaler = StandardScaler()

# Initialize the GMLVQ model
model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="steepest-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.005])},
)

# Create pipeline that first scales the data and that inputs it to the GMLVQ model
pipeline = make_pipeline(
    standard_scaler,
    model
)

# The pipeline can then be used the same as any sklearn predictor object.
pipeline.fit(data, labels)

# Make the predictions
predicted_labels = pipeline.predict(data)

# Print the classification report
print(classification_report(labels, predicted_labels))

###############################################################################
# Until now the performance has been the training performance of the model. However, we are more interested in the
# performance of the model on unseen data. This can be simulated using, e.g., crossvalidation.

###############################################################################
# Generalizability from Crossvalidation
# .....................................
# There are many different variants of crossvalidation available in the scikit-learn package. More details can be found
# in the manuals of scikit-learn.

kfold = RepeatedStratifiedKFold()


###############################################################################
# Model selection using Gridsearch
# ................................
