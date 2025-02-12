from __future__ import annotations

from subprocess import run
from abc import ABC, abstractmethod
from collections.abc import Sequence, Iterable
from typing import TYPE_CHECKING, Any
from pydantic import BaseModel
from functools import partial

from langchain_core.documents.base import Blob
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from langchain_core.documents.base import Blob


class BaseBlobTransformer(ABC):
    """Abstract base class for blob transformation.

    A blob transformation takes a sequence of blobs and returns a
    sequence of transformed blobs.
    """

    @abstractmethod
    def transform_blobs(self, blobs: Sequence[Blob], **kwargs: Any) -> Sequence[Blob]:
        """Transform a list of blobs.

        Args:
            blobs: A sequence of blobs to be transformed.

        Returns:
            A sequence of transformed blobs.
        """

    async def atransform_blobs(
        self, blobs: Sequence[Blob], **kwargs: Any
    ) -> Sequence[Blob]:
        """Asynchronously transform a list of blobs.

        Args:
            blobs: A sequence of blobs to be transformed.

        Returns:
            A sequence of transformed blobs.
        """
        return await run_in_executor(None, self.transform_blobs, blobs, **kwargs)


class SubprocessBlobTransformer(BaseBlobTransformer, BaseModel):
    """Transform each blob in the subprocess by calling the command."""

    timeout: int = 1

    def run_commands(
        self, commands: Iterable[list[str]], blobs: Iterable[Blob]
    ) -> Sequence[Blob]:
        """Run commands and return successfuly transformed blobs.

        Args:
            commands (Sequence[list[str]]): Pandoc commands.
            blobs (Sequence[Blob]): Output blobs to be returned.

        Returns:
            Sequence[Blob]: Output blobs that transformed successfully.
        """

        successfuly_transformed_blobs: list[Blob] = []

        for command, blob in zip(commands, blobs):
            try:
                run(command, timeout=self.timeout)
                successfuly_transformed_blobs.append(blob)
            except Exception:
                continue

        return successfuly_transformed_blobs

    @abstractmethod
    def get_output_blob(self, blob: Blob, **kwargs) -> Blob:
        """Generate output blob.

        Args:
            blob (Blob): Input blob.

        Returns:
            Blob: Output blob to be returned in `transform_blobs`.
        """

    @abstractmethod
    def get_command(self, blob: Blob, **kwargs) -> list[str]:
        """Generate a command that will be executed in the subprocess.

        Args:
            blob (Blob): Input blob.

        Returns:
            list[str]: Returned command.
        """

    def transform_blobs(self, blobs: Sequence[Blob], **kwargs: Any) -> Sequence[Blob]:
        """Transform blobs. For now, just run the commands in subprocess and return the created blobs.

        Args:
            blobs (Sequence[Blob]): Blobs to be transformed

        Returns:
            Sequence[Blob]: Output blobs that transformed successfully.
        """

        if "blob" in kwargs:
            raise ValueError("`blob` argument can't be in kwargs")

        output_blobs = map(partial(self.get_output_blob, **kwargs), blobs)
        commands = map(partial(self.get_command, **kwargs), blobs)

        return self.run_commands(commands=commands, blobs=output_blobs)
