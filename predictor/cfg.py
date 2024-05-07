from __future__ import annotations

import ast
import enum
import typing


@enum.unique
class Sequential(enum.Enum):
    ALWAYS = enum.auto()
    MAYBE = enum.auto()
    NEVER = enum.auto()
    UNKNOWN = enum.auto()


class CFG:

    def __init__(self):
        self.forward_edges: typing.MutableMapping[ast.AST, typing.Set[ast.AST]] = {}
        self.backward_edges: typing.MutableMapping[ast.AST, typing.Set[ast.AST]] = {}
        self.exits: typing.Set[ast.AST] = set()

    def insert_edge(self, start: ast.AST, end: ast.AST):
        if start not in self.forward_edges:
            self.forward_edges[start] = set()
        if end not in self.backward_edges:
            self.backward_edges[end] = set()
        self.forward_edges[start].add(end)
        self.backward_edges[end].add(start)

    def define_exit(self, e: ast.AST):
        self.exits.add(e)

    def define_exits(self, exits: typing.Set[ast.AST]):
        self.exits.update(exits)

    def __is_sequential(self, start: ast.AST, end: ast.AST, visited: typing.Set[ast.AST]) -> Sequential:
        result: Sequential = Sequential.UNKNOWN
        # if end is visited before and exit is not visited, result will be ALWAYS
        # if end is not visited before and exit is visited, result will be NEVER
        # if both are visited, result will be MAYBE
        # otherwise, result will be UNKNOWN
        if start not in self.forward_edges:
            return result
        visited.add(start)
        for node in self.forward_edges[start]:
            if node == end:
                match result:
                    case Sequential.NEVER | Sequential.MAYBE:
                        return Sequential.MAYBE
                    case Sequential.UNKNOWN:
                        result = Sequential.ALWAYS
            else:
                if node in self.exits:
                    match result:
                        case Sequential.ALWAYS | Sequential.MAYBE:
                            return Sequential.MAYBE
                        case Sequential.UNKNOWN:
                            result = Sequential.NEVER
                if node not in visited:
                    match self.__is_sequential(node, end, visited):
                        case Sequential.ALWAYS:
                            match result:
                                case Sequential.NEVER | Sequential.MAYBE:
                                    return Sequential.MAYBE
                                case Sequential.UNKNOWN:
                                    result = Sequential.ALWAYS
                        case Sequential.MAYBE:
                            return Sequential.MAYBE
                        case Sequential.NEVER:
                            match result:
                                case Sequential.ALWAYS | Sequential.MAYBE:
                                    return Sequential.MAYBE
                                case Sequential.UNKNOWN:
                                    result = Sequential.NEVER
        return result

    def is_sequential(self, start: ast.AST, end: ast.AST) -> Sequential:
        result = self.__is_sequential(start, end, set())
        assert result != Sequential.UNKNOWN
        return result

    def delete_edge(self, start: ast.AST, end: ast.AST):
        if start in self.forward_edges:
            self.forward_edges[start].remove(end)
        if end in self.backward_edges:
            self.backward_edges[end].remove(start)
