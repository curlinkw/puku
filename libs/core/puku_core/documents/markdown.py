from marko import Markdown
from marko.renderers.md_renderer import MarkdownRenderer
from marko.elements.block import Document

MarkdownDocument = Document

# Inner instance, use the bare convert/parse/render function instead
_markdown = Markdown(renderer=MarkdownRenderer)


def convert(text: str) -> str:
    """Parse and render the given text.

    :param text: text to convert.
    :returns: The rendered result.
    """
    return _markdown.convert(text)


def parse(text: str) -> Document:
    """Parse the text to a structured data object.

    :param text: text to parse.
    :returns: the parsed object
    """
    return _markdown.parse(text)


def render(parsed: Document) -> str:
    """Render the parsed object to text.

    :param parsed: the parsed object
    :returns: the rendered result.
    """
    return _markdown.render(parsed)
