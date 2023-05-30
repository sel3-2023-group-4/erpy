from __future__ import annotations

import abc
import copy
from typing import List, TYPE_CHECKING

from erpy import random_state
from erpy.framework.parameters import FixedParameter, SynchronizedParameter, Parameter, ContinuousParameter
from erpy.framework.specification import Specification

if TYPE_CHECKING:
    pass


def unfold_list(nodes: List[DirectedNode]) -> [Specification]:
    """
    This function unfolds a list of directed nodes to a list of corresponding specifications.

    The remarkable part of this function is that the unfolded node (read specification) will
    appear as many times in the list of specifications as the original node was repeated (+ 1).
    For example, a node that is points 2 times to itself (is 2 times repeated), will appear
    as a specification 3 times.

    :param nodes: the list of nodes to be unfolded.
    :return: the list of unfolded nodes (read: specifications).
    """
    specifications = []
    for node in nodes:
        specification = node.unfold()
        specifications.extend([specification] * (node.repeated + 1))
    return specifications


class DirectedGraph(metaclass=abc.ABCMeta):
    """
    A directed graph used for indirect genetic encodings.

    :var root: the root of the graph (this root should link to the other nodes of the graph).
    """
    def __init__(self, root: DirectedNode) -> None:
        self.root = root

    def generate(self) -> DirectedGraph:
        """
        Generate a new directed graph from this directed graph; keeping the structure
        but with random values for the nodes. Useful in evolutionary algorithms.

        :return: A new directed graph generated from this one.
        """
        graph = copy.deepcopy(self)
        graph.root.generate()
        return graph

    def mutate(self) -> DirectedGraph:
        """
        Generate a new directed graph from this directed graph; keeping the structure
        but with slightly mutated values for the nodes. Useful in evolutionary
        algorithms.

        :return: A new mutated graph generated from this one.
        """
        graph = copy.deepcopy(self)
        graph.root.mutate()
        return graph

    def get_labels_and_values(self, node: DirectedNode, idx: int = 0):
        """
        Get labels and parameters for this node and all its children.

        :param node: the node to get labels and parameters for.
        :param idx: idx used to distinguish children of the same class.
        :return: a list of tuples with the name of the parameters and their values.
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
        """
        Unfold this graph to a specification used in the simulation environment.

        :return: A spec.
        """
        raise NotImplementedError


class DirectedNode(metaclass=abc.ABCMeta):
    """
    A directed node used in DirectedGraph.

    :var children: a list of children nodes directly linked to this node (default = []).
    :var repeated: the amount of times this node links to itself (default = 0).
    """
    def __init__(self, children: List[DirectedNode] = [], repeated: int = 0) -> None:
        assert repeated >= 0
        self.children = children
        # number of times a node points to itself
        self.repeated = repeated

    @property
    def mutable_parameters(self):
        """
        A list of mutable parameters this node has. A mutable parameter is an erpy Parameter which is not
        Fixed or Synchronized.
        """
        pars: List[Parameter] = []
        for name in vars(self):
            var = self.__getattribute__(name)
            if isinstance(var, Parameter) and not (
                    isinstance(var, FixedParameter) or isinstance(var, SynchronizedParameter)):
                pars.append(var)
        return pars

    def is_recursive(self) -> bool:
        """
        A recursive node is a node that points to itself; i.e. is not repeated.
        """
        return self.repeated > 0

    def generate(self):
        """
        Set random values for the mutable parameters and recursively do this for all the children.

        NOTE: this change happens in place.
        """
        for par in self.mutable_parameters:
            par.set_random_value()
        for child in self.children:
            child.generate()

    def mutate(self):
        """
        Mutate the values for the mutable parameters and recursively do this for all the children.

        NOTE: this change happens in place.
        """
        self.mutate_mutable_parameters()
        for child in self.children:
            child.mutate()

    def mutate_mutable_parameters(self):
        """
        Actually mutate the mutable parameters.

        erpy Parameters that get mutated.
        - ContinuousParemeters
        """
        for par in self.mutable_parameters:
            if isinstance(par, ContinuousParameter):
                value_range = par.high - par.low

                noise = value_range * random_state.normal(scale=0.01)
                par.value += noise

    @abc.abstractmethod
    def unfold(self) -> Specification:
        """
        Unfold this node to a specification used in the simulation environment.

        :return: A spec.
        """
        raise NotImplementedError
