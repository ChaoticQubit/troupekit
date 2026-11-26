import pathlib

import pytest

import troupekit
from troupekit import _troupekit

CRATES = pathlib.Path(__file__).resolve().parents[2]
FREEZE = CRATES / "troupekit-ffi-core" / "contract" / "freeze.txt"
RENAMES = {
    "FfiError": "TroupekitError",
    "FfiEvent": "Event",
    "FfiEventKind": "EventKind",
    "FfiUsage": "Usage",
    "FfiPause": "Pause",
}
FREEZE_NAMES_MIN = 50


def freeze_names():
    """Reads the neutral names of every exposed freeze row.

    Input: none. Output: the non-blank lines of `contract/freeze.txt`; the
    file sits beside the crate in the source tree the tests run from.
    """
    names = [line.strip() for line in FREEZE.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    assert len(names) >= FREEZE_NAMES_MIN
    return names


def exported(module, name):
    """Tells whether `module` exports one freeze name, after the Q3 renames.

    Input: the native module and `Type` or `Type::method`. Output: True when
    the attribute exists; a neutral `new` is the class itself.
    """
    owner, _, method = name.partition("::")
    target = getattr(module, RENAMES.get(owner, owner), None)
    if target is None:
        return False
    if not method or method == "new":
        return True
    return hasattr(target, method)


@pytest.mark.xfail(
    not hasattr(troupekit, "AgentBuilder"),
    reason="B5: troupekit-py exports only version() until the PyO3 adapter lands",
    raises=AssertionError,
    strict=True,
)
def test_freeze_names_match_adapters():
    """Every freeze name is exported by the native module and the package.

    Input: `freeze.txt` and the installed `troupekit`. Output: passes when no
    name is missing from `troupekit._troupekit` or from `troupekit`.
    """
    names = freeze_names()
    missing_native = [name for name in names if not exported(_troupekit, name)]
    missing_package = [name for name in names if not exported(troupekit, name)]
    assert missing_native == []
    assert missing_package == []
