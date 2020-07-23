=============
API Reference
=============

API of included algorithms in sklvq.

.. currentmodule:: sklvq

Predictor
=========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   GLVQ

Predictor and Transformer
=========================

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

    DiscriminativeBaseClass

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    RelativeDistance

Distance Functions
==================
.. currentmodule:: sklvq.distances

Distance functions ordered by compatible predictor.

GLVQ
----

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    Euclidean

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    SquaredEuclidean

GMLVQ
-----

.. autosummary::
   :toctree: generated/
   :template: callable.rst

    AdaptiveSquaredEuclidean

LGMLVQ
--------

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

