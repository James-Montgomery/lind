"""
Lind
====

Lind is a package for experimental design and analysis.

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
the documentation site https://james-montgomery.github.io/lind/build/html/index.html

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as `np`::
  >>> import lind as ld
Code snippets are indicated by three greater-than signs::
  >>> x = 42
  >>> x = x + 1
Use the built-in ``help`` function to view a function's docstring::
  >>> help(ld.design.design_full_factorial)

"""

from importlib.util import find_spec as _find_spec # requires python >= 3.3
if _find_spec('rpy2') is not None:
    from lind import r_backends
if _find_spec('lind_static_resources') is not None:
    from lind_static_resources import static_files_abs_path as _sfap
else:
    _sfap = None

from ._version import __version__

from lind import (
    design, analysis
)
