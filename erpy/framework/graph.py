from __future__ import annotations

import abc
import copy
from typing import List, TYPE_CHECKING

from erpy import random_state
from erpy.framework.parameters import FixedParameter, SynchronizedParameter, Parameter, ContinuousParameter, \
    DiscreteParameter
from erpy.framework.specification import Specification

if TYPE_CHECKING:
    pass


def unfold_list(nodes: List[DirectedNode]) -> [Specification]:
    specifications = []
    for node in nodes:
        specification = node.unfold()
        specifications.extend([specification] * (node.repeated + 1))
    return specifications


class DirectedGraph(metaclass=abc.ABCMeta):
    def __init__(self, root: DirectedNode) -> None:
        self.root = root

    def generate(self) -> DirectedGraph:
        graph = copy.deepcopy(self)
        graph.root.generate()
        return graph

    def mutate(self) -> DirectedGraph:
        graph = copy.deepcopy(self)
        graph.root.mutate()
        return graph

    def get_labels_and_values(self, node: DirectedNode, idx: int = 0):
        """
        Get labels and parameters for this node and all its children.
        """
        pars = []
        for name in vars(node):
            var = node.__getattribute__(name)
            if isinstance(var, Parameter) and not (
                    isinstance(var, FixedParameter) or isinstance(var, SynchronizedParameter)):
                pars.append((node.__class__.__name__ + "_" + str(idx) + "_" + name, var.value))
        j = 0
        for child in node.children:
            pars.extend(self.get_labels_and_values(child, idx=j))
            i = 1
            while i <= node.repeated:
                pars.extend(self.get_labels_and_values(child, idx=j + i))
                i += 1
            j += i
        return pars

    @abc.abstractmethod
    def unfold(self) -> Specification:
        raise NotImplementedError


class DirectedNode(metaclass=abc.ABCMeta):
    def __init__(self, children: List[DirectedNode] = [], repeated: int = 0) -> None:
        self.children = children
        # number of times a node points to itself
        self.repeated = repeated

    @property
    def mutable_parameters(self):
        pars: List[Parameter] = []
        for name in vars(self):
            var = self.__getattribute__(name)
            if isinstance(var, Parameter) and not (
                    isinstance(var, FixedParameter) or isinstance(var, SynchronizedParameter)):
                pars.append(var)
        return pars

    def is_recursive(self) -> bool:
        return self.repeated > 0

    def generate(self):
        for par in self.mutable_parameters:
            par.set_random_value()
        for child in self.children:
            child.generate()

    def mutate(self):
        self.mutate_mutable_parameters()
        for child in self.children:
            child.mutate()

    def mutate_mutable_parameters(self):
        for par in self.mutable_parameters:
            if isinstance(par, ContinuousParameter):
                value_range = par.high - par.low

                noise = value_range * random_state.normal(scale=0.01)
                par.value += noise

    @abc.abstractmethod
    def unfold(self) -> Specification:
        raise NotImplementedError
