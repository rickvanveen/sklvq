# sklvq - SciKit Learning Vector Quantization (A scikit-learn extensions)

[![Build Status](https://travis-ci.org/rickvanveen/LVQToolbox.svg?branch=develop)](https://travis-ci.org/rickvanveen/LVQToolbox)
[![Coverage Status](https://coveralls.io/repos/github/rickvanveen/LVQToolbox/badge.svg?branch=develop)](https://coveralls.io/github/rickvanveen/LVQToolbox?branch=master

Template based on: https://github.com/scikit-learn-contrib/project-template
Implementation based on: http://matlabserver.cs.rug.nl/gmlvqweb/web/
Developer manual: https://scikit-learn.org/stable/developers/index.html

Current status: I am not doing anything with the sklearn build/testing stuff anymore. So that doesn't work. Should be added in the future again (and it has been changed by sklearn people). The structure and code in the repository has undergone multiple design changes and is fairly consistent (hopefully), but not everywhere.

# Developer notes
I thought it would be useful to write down some of the choices made. Some design decisions might not always be that obvious and it is easy to forget.

## Callable classes
It's useful because it makes the algorithms more expandable. Now we don't have to worry about that one distance function needs different parameters than another one. As we can simply provide an interface to select a function and pass it's arguments (whatever they are) as dictionary. This is what you can find in the model classes. For example, the models accept `distance_type`, a string and `distance_params` a dictionary. This wouldn't work as easy with regular functions.

