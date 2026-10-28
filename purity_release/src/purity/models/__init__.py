"""The main reconstruction and ATAR-only tail-reveal models.

Imports are lazy so selecting the tail model does not capture the main model's
profile environment before its own import.
"""

__all__ = ["PURITYModel", "PURITYTailModel"]


def __getattr__(name):
    if name == "PURITYModel":
        from .main import PURITYModel
        return PURITYModel
    if name == "PURITYTailModel":
        from .tail import PURITYTailModel
        return PURITYTailModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
