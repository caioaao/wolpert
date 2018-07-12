from .cross_val import CVStackableTransformer

__all__ = ["CVStackableTransformer"]


def _choose_wrapper(blending_type):
    """Placeholder function to choose between transformers"""
    return CVStackableTransformer
