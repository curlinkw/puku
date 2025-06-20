from puku_core.graphs.knowledge_trees.edges import BaseEdge


class Edge(BaseEdge):
    description: str
    """what information can be clarified further"""
