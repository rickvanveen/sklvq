=============
API Reference
=============

If you would like to use sklvq algorithms the most relevant part to look at is
the "Predictors and Transformers" section. However, the other sections provide information
about accepted parameters their range and default values.

Predictors and Transformers
===========================
.. currentmodule:: sklvq.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    LVQBaseClass

.. autosummary::
    :toctree: generated/
    :template: class.rst

    GLVQ

.. autosummary::
   :toctree: generated/
   :template: class.rst

   GMLVQ

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LGMLVQ

Objective Functions
===================
.. currentmodule:: sklvq.objectives

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    GeneralizedLearningObjective

Activation Functions
====================
.. currentmodule:: sklvq.activations

.. autosummary::
    :toctree: generated/
    :template: callable.rst

    ActivationBaseClass

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    Identity

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    Sigmoid

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    SoftPlus

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    Swish

Discriminant Functions
======================
.. currentmodule:: sklvq.discriminants

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    DiscriminantBaseClass

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    RelativeDistance

Distance Functions
==================
.. currentmodule:: sklvq.distances

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    DistanceBaseClass

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    Euclidean

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    SquaredEuclidean

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    AdaptiveSquaredEuclidean

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    LocalAdaptiveSquaredEuclidean

Solvers
=======

.. currentmodule:: sklvq.solvers

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    SolverBaseClass

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    SteepestGradientDescent

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    WaypointGradientDescent

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    AdaptiveMomentEstimation

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    BroydenFletcherGoldfarbShanno

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    LimitedMemoryBfgs

