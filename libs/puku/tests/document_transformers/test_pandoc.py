from langchain_core.documents.base import Blob
from puku.document_transformers.pandoc import PandocBlobTransformer


def test_basic_functions():
    transformer = PandocBlobTransformer(from_format="docx", to_format="markdown")
    blob = Blob.from_path("/must/not/exist/file.docx")
    output_dir = "/not/exist/"

    assert transformer.get_output_blob(blob=blob).path == "/must/not/exist/file.md"
    assert (
        transformer.get_output_blob(blob=blob, output_dir=output_dir).path
        == "/not/exist/file.md"
    )

    assert (
        " ".join(transformer.get_command(blob=blob, output_dir=output_dir))
        == "pandoc /must/not/exist/file.docx -f docx -t markdown -o /not/exist/file.md"
    )
