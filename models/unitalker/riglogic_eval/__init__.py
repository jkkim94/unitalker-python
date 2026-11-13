"""RigLogic vertex-space evaluation for UniTalker MetaHuman Head Decoder."""

from .config import load_riglogic_config

__all__ = ["compare_clip", "load_riglogic_config"]


def compare_clip(*args, **kwargs):
    from .compare_clip import compare_clip as _compare_clip

    return _compare_clip(*args, **kwargs)
