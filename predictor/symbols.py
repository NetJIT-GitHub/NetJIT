from __future__ import annotations

import ast
import builtins
import typing


class Symbol(object):
    def __init__(self, name: str, alias: Symbol = None, is_builtin: bool = False, is_free=False, is_global: bool = False):
        self.name = name
        self.alias = alias
        self.is_builtin = is_builtin
        self.is_free = is_free
        self.is_global = is_global
        self.loads: typing.Set[ast.AST] = set()
        self.stores: typing.Set[ast.AST] = set()
        self.deletes: typing.Set[ast.AST] = set()

    def __repr__(self) -> str:
        return f"Symbol({self.name}, alias={self.alias}, is_builtin={self.is_builtin}, is_global={self.is_global})"

    def load(self, node: ast.AST):
        if self.alias is not None:
            self.alias.load(node)
        self.loads.add(node)

    def store(self, node: ast.AST):
        if self.alias is not None:
            self.alias.store(node)
        self.stores.add(node)

    def delete(self, node: ast.AST):
        if self.alias is not None:
            self.alias.delete(node)
        self.deletes.add(node)

    def in_global(self) -> bool:
        return self.is_global or (self.is_free and (self.alias is None or self.alias.in_global()))


class SymbolTable(object):
    def __init__(self, parent: SymbolTable = None):
        self.parent = parent
        self.memory: typing.MutableMapping[str, Symbol] = {}
        self.builtins: typing.MutableMapping[str, Symbol] = {
            name: Symbol(name, is_builtin=True)
            for name in dir(builtins)
        } if parent is None else None

    def __getitem__(self, item: str):
        if item in self.memory:
            return self.memory[item]
        elif self.builtins is not None and item in self.builtins:
            return self.builtins[item]
        elif self.parent is not None:
            return self.parent[item]
        else:
            raise KeyError(f"{item} is not defined")

    def __contains__(self, item: str):
        return (item in self.memory
                or (self.parent is not None and item in self.parent)
                or (self.builtins is not None and item in self.builtins))

    def __setitem__(self, key: str, value: Symbol):
        if key in self.memory:
            self.memory[key] = value
        elif self.parent is not None and key in self.parent:
            self.parent[key] = value
        elif self.builtins is not None and key in self.builtins:
            self.builtins[key] = value
        else:
            self.memory[key] = value

    def __delitem__(self, key: str):
        if key in self.memory:
            del self.memory[key]
        elif self.parent is not None and key in self.parent:
            del self.parent[key]
        elif self.builtins is not None and key in self.builtins:
            del self.builtins[key]
        else:
            raise KeyError(f"{key} is not defined")

    def __repr__(self) -> str:
        return f"SymbolTable:\nmemory:\n{self.memory}"

    def new(self) -> SymbolTable:
        return SymbolTable(self)

    def define_local(self, name: str, alias: typing.Optional[Symbol] = None) -> Symbol:
        if name in self.memory:
            symbol = self.memory[name]
            if symbol.is_free:
                symbol.is_free = False
                symbol.alias = alias
                symbol.is_global = self.is_root()
        else:
            self.memory[name] = Symbol(name, alias=alias, is_global=self.is_root())
        return self[name]

    def define_nonlocal(self, name: str) -> Symbol:
        alias = self.parent.define_free(name) if self.parent is not None else None
        if name in self.memory:
            symbol = self.memory[name]
            symbol.is_free = False
            symbol.alias = alias
        else:
            symbol = Symbol(name, alias=alias)
            self.memory[name] = symbol
        return self[name]

    def define_global(self, name: str) -> Symbol:
        alias = self.root().define_free(name) if not self.is_root() else None
        if name in self.memory:
            symbol = self.memory[name]
            symbol.is_free = False
            symbol.is_global = True
            symbol.alias = alias
        else:
            self.memory[name] = Symbol(name, alias=alias, is_global=True)
        return self[name]

    def define_free(self, name: str) -> Symbol:
        if name not in self.memory:
            alias = self[name] if name in self else None if self.parent is None else self.parent.define_free(name)
            symbol = Symbol(name, alias=alias, is_free=True, is_builtin=alias.is_builtin if alias is not None else False)
            self.memory[name] = symbol
        return self[name]

    def root(self) -> SymbolTable:
        return self.parent.root() if self.parent is not None else self

    def is_root(self) -> bool:
        return self.parent is None
