import functools

import pytest

import troupekit

UNSUPPORTED = int(troupekit.ErrorCode.Unsupported)


def neutral_is_skeleton():
    """Tells whether troupekit-ffi-core still answers with the K5 skeleton.

    Input: none. Output: True when building a model-less agent raises
    `TroupekitError` with code `Unsupported` instead of a real build error.
    """
    try:
        troupekit.AgentBuilder("probe").build()
    except troupekit.TroupekitError as error:
        return error.code == UNSUPPORTED
    return False


SKELETON = neutral_is_skeleton()
SKELETON_REASON = "K6 to K8: troupekit-ffi-core bodies still return Unsupported"


def needs_neutral(test):
    """Skips a contract case while the neutral call it reaches is a stub.

    Input: the test function. Output: a wrapper that turns a `TroupekitError`
    with code `Unsupported` into a skip naming the unbuilt call.
    """

    @functools.wraps(test)
    def run(*args, **kwargs):
        """Runs the case and maps an Unsupported failure to a skip.

        Input: the case fixtures. Output: the case result, or a skip.
        """
        try:
            return test(*args, **kwargs)
        except troupekit.TroupekitError as error:
            if error.code != UNSUPPORTED:
                raise
            pytest.skip(f"not built in troupekit-ffi-core yet: {error.message}")

    return run
