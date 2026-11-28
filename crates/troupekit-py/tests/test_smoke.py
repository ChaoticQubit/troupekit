import re

import troupekit


def test_version_is_exported():
    """The Python package exports `version` from the native module.

    Input: the installed `troupekit` package. Output: passes when `version()`
    returns a non-empty semantic version string equal to the native module's.
    """
    value = troupekit.version()
    assert isinstance(value, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", value)
    assert value == troupekit._troupekit.version()
