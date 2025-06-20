from typing import List

from puku_core.graphs.knowledge_trees.writer import BaseKnowledgeWriter
from puku.graphs.knowledge_trees.nodes import Node
from puku.graphs.knowledge_trees.edges import Edge

class KnowledgeWriter(BaseKnowledgeWriter):
    def _get_