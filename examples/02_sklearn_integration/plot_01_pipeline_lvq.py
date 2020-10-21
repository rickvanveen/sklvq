"""
=========
Pipelines
=========

In these examples GMLVQ is used but the same applies to all the other algorithms.
Also this feature is provided by scikit-learn and we therefore also refer to scikit-learn's
documentation.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklvq import GMLVQ

data, labels = load_iris(return_X_y=True)

###############################################################################
# Sklearn provides a very handy way of creating a pipeline for, e.g., preprocessing. A pipeline can
# be used in the same way as any sklearn and sklvq estimator.

scaler = StandardScaler()

model = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 10, "k": 3, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
)

pipeline = make_pipeline(scaler, model)

# As with the algorithm examples. However, the pipeline now first scales the X and then
# passes it to GMLVQ.
pipeline.fit(data, labels)

# Predict the labels using the trained pipeline. The pipeline will use the
# mean and standard deviation it found when fit was called and applies it to the X.
# Which is in this case the same X.
predicted_labels = pipeline.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))
