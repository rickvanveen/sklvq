from .template import (TemplateEstimator, TemplateClassifier,
                       TemplateTransformer)
from .glvqmodel import (GeneralizedLearningVectorQuantization)

from . import template

__all__ = ['TemplateEstimator', 'TemplateClassifier',
           'TemplateTransformer', 'template']
