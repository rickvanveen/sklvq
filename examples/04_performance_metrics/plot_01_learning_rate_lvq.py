"""
===========================================
GMLVQ learning curve and feature projection
===========================================

Track the training objective and see how GMLVQ's projection of the four Iris
features changes during training. Each solver callback receives a detached
snapshot, so we can keep the history and plot it after fitting.
"""

from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklvq import GMLVQ

iris = load_iris()
data = StandardScaler().fit_transform(iris.data)

###############################################################################
# The initial snapshot is iteration zero. Subsequent snapshots report the
# completed iteration and let us estimate the time remaining to the run limit.


class ProcessLogger:
    def __init__(self):
        self.states = []

    def __call__(self, state):
        self.states.append(state)
        iteration = state["nit"]
        if iteration == "Initial":
            self.start = monotonic()
            return
        limit = state["max_runs"]
        elapsed = monotonic() - self.start
        eta = elapsed / iteration * (limit - iteration)
        print(f"{iteration}/{limit}: cost {state['fun']:.3f}, elapsed {elapsed:.2f}s, ETA to limit {eta:.2f}s")
        # Return True here to stop training at this snapshot.


logger = ProcessLogger()
model = GMLVQ(
    relevance_n_components=2,
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 30, "callback": logger},
    random_state=1428,
)

model.fit(data, iris.target)

###############################################################################
# The solver reports the training objective directly; no need to score the
# model again. Lower values indicate a better fit to this objective.

fig, ax = plt.subplots()
ax.plot([state["fun"] for state in logger.states])
ax.set(title="Learning curve", xlabel="Completed iterations", ylabel="Training objective")

###############################################################################
# To isolate changes in the projection, follow four *fixed* input directions.
# A one-standard-deviation increase in
# feature j projects to column j of omega. At each height, the four points
# show where that iteration's projection sends the same four input directions.

omegas = np.stack([state["model_params"]["omega"] for state in logger.states])
iterations = np.arange(len(omegas))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, layout="constrained")
for j, feature in enumerate(iris.feature_names):
    ax.plot(omegas[:, 0, j], omegas[:, 1, j], iterations, label=feature)

ax.set(
    title="How the learned projection changes",
    xlabel="Relevance coordinate 1",
    ylabel="Relevance coordinate 2",
    zlabel="Iteration",
)
ax.legend()
