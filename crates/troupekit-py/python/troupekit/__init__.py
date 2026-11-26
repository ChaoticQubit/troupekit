"""Troupekit for Python: build and run troupekit agents.

Re-exports every public name of the native module `troupekit._troupekit`.
Calls block with the GIL released; `troupekit.aio` has async forms.
Input: none. Output: the package namespace.
"""

from troupekit import _troupekit
from troupekit._troupekit import *

__all__ = sorted(name for name in dir(_troupekit) if not name.startswith("_"))
