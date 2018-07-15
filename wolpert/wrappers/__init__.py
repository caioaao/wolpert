from .cross_val import CVStackableTransformer
from .holdout import HoldoutStackableTransformer

__all__ = ["CVStackableTransformer", "HoldoutStackableTransformer"]


def _choose_wrapper(blending_wrapper):
    """Placeholder function to choose between transformers"""
    if blending_wrapper == "cv":
        return CVStackableTransformer
    elif blending_wrapper == "holdout":
        return HoldoutStackableTransformer
