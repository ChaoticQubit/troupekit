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
