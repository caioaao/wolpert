from .cross_val import CVStackableTransformer, CVWrapper
from .holdout import HoldoutStackableTransformer, HoldoutWrapper
from .base import BaseWrapper

__all__ = ["CVStackableTransformer", "CVWrapper",
           "HoldoutStackableTransformer", "HoldoutWrapper"]


def _choose_wrapper(blending_wrapper):
    """Choose between wrappers"""
    if issubclass(blending_wrapper, BaseWrapper):
        return blending_wrapper
    elif blending_wrapper == "cv":
        return CVWrapper()
    elif blending_wrapper == "holdout":
        return HoldoutWrapper()
