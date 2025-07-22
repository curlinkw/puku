from langchain_core.output_parsers import BaseOutputParser

from puku_core.documents.markdown import MarkdownDocument, parse


class MarkdownOutputParser(BaseOutputParser[MarkdownDocument]):
    def parse(self, text: str) -> MarkdownDocument:
        return parse(text)
