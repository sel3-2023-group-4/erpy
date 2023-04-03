from __future__ import annotations

import abc
from typing import List, TYPE_CHECKING

from erpy.framework.specification import Specification

if TYPE_CHECKING:
    pass


class DirectedGraph(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def unfold(self) -> Specification:
        raise NotImplementedError

    @staticmethod
    def unfold_list(nodes: List[DirectedNode]) -> [Specification]:
        specifications = []
        for node in nodes:
            specification = node.unfold()
            specifications.extend([specification] * (node.repeated + 1))
        return specification


class DirectedNode(metaclass=abc.ABCMeta):
    def __init__(self, repeated: int = 0) -> None:
        # number of times a node points to itself
        self.repeated: int = repeated

    @abc.abstractmethod
    def unfold(self) -> Specification:
        raise NotImplementedError
