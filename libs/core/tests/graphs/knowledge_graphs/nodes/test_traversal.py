from typing import List, Tuple

from puku_core.graphs.knowledge_trees.edges import BaseEdge
from puku_core.graphs.knowledge_trees.nodes import BaseNode, TraversalNode


def test_descendants():
    class IndexNode(BaseNode):
        idx: int

    nodes = [IndexNode(idx=i) for i in range(11)]

    assert nodes[0].descendants() == [TraversalNode(node=nodes[0])]

    def add_edge(i: int, j: int) -> None:
        nodes[i].children.append(BaseEdge(child=nodes[j]))

    add_edge(1, 2)
    add_edge(2, 3)
    add_edge(2, 4)
    add_edge(2, 10)
    add_edge(1, 5)
    add_edge(5, 6)
    add_edge(6, 7)
    add_edge(7, 9)
    add_edge(6, 8)

    def convert_traversal(traversal: List[TraversalNode]) -> List[Tuple[int, int]]:
        return [
            (-1 if x.parent is None else x.parent.idx, x.node.idx) for x in traversal
        ]

    assert convert_traversal(nodes[6].descendants()) == [
        (-1, 6),
        (6, 7),
        (6, 8),
        (7, 9),
    ]

    assert convert_traversal(nodes[1].descendants()) == [
        (-1, 1),
        (1, 2),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 10),
        (5, 6),
        (6, 7),
        (6, 8),
        (7, 9),
    ]
