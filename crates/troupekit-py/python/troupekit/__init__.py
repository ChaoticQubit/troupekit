"""Troupekit for Python: build and run troupekit agents.

Re-exports the native module `troupekit._troupekit`. Input: none.
Output: the public names of the native module.
"""

from troupekit import _troupekit
from troupekit._troupekit import version

__all__ = ["version"]
