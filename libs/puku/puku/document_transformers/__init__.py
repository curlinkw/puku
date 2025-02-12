import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from puku.document_transformers.pandoc import PandocBlobTransformer

__all__ = ["PandocBlobTransformer"]

_module_lookup = {
    "PandocBlobTransformer": "puku.document_transformers.pandoc",  # noqa: E501
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
