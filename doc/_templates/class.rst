:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
<<<<<<< HEAD
    :special-members: __init__
=======

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}
>>>>>>> e137efd835422bc26161ea02d0bc1b488c79ef6a

.. include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div style='clear:both'></div>
