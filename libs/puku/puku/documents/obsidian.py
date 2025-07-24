import os

from puku_core.documents.markdown import render
from puku_core.graphs.knowledge_trees.nodes.markdown import MarkdownNode


def dump_knowledge_tree(node: MarkdownNode, path: str) -> None:
    if not (os.path.isdir(path)):
        raise FileNotFoundError(f"{path} does not exists or it is not a directory.")

    for traversal_node in node.descendants():
        current_node = traversal_node.node
        with open(os.path.join(path, current_node.title + ".md"), "w") as f:
            f.write(render(current_node.data))

    return None
