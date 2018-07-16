from .cross_val import CVStackableTransformer, CVWrapper
from .holdout import HoldoutStackableTransformer, HoldoutWrapper

__all__ = ["CVStackableTransformer", "CVWrapper",
           "HoldoutStackableTransformer", "HoldoutWrapper"]


def _choose_wrapper(blending_wrapper):
    """Choose between wrappers

    Parameters
    ----------
    blending_wrapper: string or Wrapper object, optional (default='cv')
        The strategy to be used when blending. Possible string values are 'cv'
        and 'holdout'. If a wrapper object is passed, it will be used instead.
    """
    if isinstance(blending_wrapper, str):
        if blending_wrapper == "cv":
            return CVWrapper()
        elif blending_wrapper == "holdout":
            return HoldoutWrapper()
    else:
        return blending_wrapper
