from typing import TypeVar
from langchain_core.load.serializable import Serializable

from puku_core.load.hashable import UUIDHashable


class BaseNode(UUIDHashable, Serializable):
    """Base node class."""

    def __hash__(self) -> int:
        return super().__hash__()


NodeType = TypeVar("NodeType", bound=BaseNode, covariant=True)
