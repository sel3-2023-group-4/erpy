from __future__ import annotations

import abc
from typing import List, TYPE_CHECKING

from erpy.framework.specification import Specification

if TYPE_CHECKING:
    pass


class DirectedGraph(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.root = None

    def parameterize(self, root: DirectedNode) -> None:
        self.root = root
        to_visit: List[DirectedNode] = [root]
        while len(to_visit) > 0:
            node: DirectedNode = to_visit.pop()
            node.parameterize()
            to_visit.extend(node.next)

    @abc.abstractmethod
    def unfold(self) -> Specification:
        raise NotImplementedError


class DirectedNode(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.next: List[DirectedNode] = []

    @abc.abstractmethod
    def parameterize(self) -> None:
        raise NotImplementedError
