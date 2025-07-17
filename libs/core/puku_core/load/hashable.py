from typing import Hashable
from pydantic import BaseModel, PrivateAttr
from uuid import UUID, uuid4


class UUIDHashable(BaseModel, Hashable):
    _uuid: UUID = PrivateAttr(default_factory=uuid4)

    def __hash__(self) -> int:
        return hash(self._uuid)
