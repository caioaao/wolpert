from .cross_val import CVStackableTransformer
from .holdout import HoldoutStackableTransformer

__all__ = ["CVStackableTransformer", "HoldoutStackableTransformer"]


def _choose_wrapper(blending_type):
    """Placeholder function to choose between transformers"""
    if blending_type == "cv":
        return CVStackableTransformer
    elif blending_type == "holdout":
        return HoldoutStackableTransformer
