import os
import shutil
from typing import Optional, Sequence, Any
from langchain_core.documents.base import Blob, PathLike

from puku_core.documents.transformers import SubprocessBlobTransformer

PANDOC_FORMAT_TO_EXTENSION = {
    "markdown": "md",
    "html": "html",
    "latex": "tex",
    "pdf": "pdf",
    "docx": "docx",
    "odt": "odt",
    "plain": "txt",
    "rst": "rst",
    "epub": "epub",
    "asciidoc": "adoc",
}


class PandocBlobTransformer(SubprocessBlobTransformer):
    """Runs pandoc to convert extension."""

    from_format: str
    to_format: str

    def _get_output_path(
        self, blob: Blob, output_dir: Optional[PathLike] = None
    ) -> str:
        """Get path for output blob."""

        # all blobs must contain path
        if blob.path is None:
            raise ValueError(f"Blob {blob} does not have path.")

        # change extension
        without_extension = os.path.splitext(blob.path)[0]

        if not without_extension:
            raise ValueError(f"{blob.path} file has bad name")

        output_path = (
            f"{without_extension}.{PANDOC_FORMAT_TO_EXTENSION[self.to_format]}"
        )

        # change folder
        if not (output_dir is None):
            output_path = os.path.join(output_dir, os.path.basename(output_path))

        return output_path

    def get_output_blob(
        self, blob: Blob, output_dir: Optional[PathLike] = None, **kwargs
    ) -> Blob:
        """Creates new blob from output path."""

        return Blob.from_path(
            path=self._get_output_path(blob=blob, output_dir=output_dir)
        )

    def get_command(
        self,
        blob: Blob,
        output_dir: Optional[PathLike] = None,
        standalone: bool = False,
        **kwargs,
    ) -> list[str]:
        """Creates pandoc command."""

        output_path = self._get_output_path(blob=blob, output_dir=output_dir)

        return (
            ["pandoc"]
            + (["-s"] if standalone else [])
            + [str(blob.path)]
            + ["-f", self.from_format]
            + ["-t", self.to_format]
            + ["-o", output_path]
        )

    def transform_blobs(
        self,
        blobs: Sequence[Blob],
        output_dir: Optional[PathLike] = None,
        standalone: bool = False,
        **kwargs: Any,
    ) -> Sequence[Blob]:
        """Convert blob extensions using pandoc.

        Args:
            blobs (Sequence[Blob]): Blobs to be converted. Must contain path.
            output_dir (Optional[PathLike], optional): The directory for saving the results. \
                If not, the files are saved in the same location.
            standalone (bool, optional): Pandoc standalone flag.

        Returns:
            Sequence[Blob]: _description_
        """

        if shutil.which("pandoc") is None:
            raise FileNotFoundError(
                "Pandoc is not installed. Please install it from https://pandoc.org/installing.html"
            )

        return super().transform_blobs(
            blobs=blobs, output_dir=output_dir, standalone=standalone, **kwargs
        )
