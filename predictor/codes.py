import ast
import importlib.machinery
import inspect
import os.path
import types
import typing

from predictor import cfg
from predictor import symbols
from predictor import visitors

ast_storage: typing.MutableMapping[types.CodeType | types.ModuleType, ast.AST] = {}
cfg_storage: typing.MutableMapping[types.CodeType, cfg.CFG] = {}
symbol_table_storage: typing.MutableMapping[ast.AST, typing.Mapping[ast.AST, symbols.SymbolTable]] = {}


def check_scope(o: typing.Any, scope: typing.Optional[str] = None) -> bool:
    f = retrieve_function(o)
    if f is None or inspect.isbuiltin(f):
        return False
    file = inspect.getsourcefile(f)
    if file is None:
        return False
    if not any(file.endswith(suffix) for suffix in importlib.machinery.SOURCE_SUFFIXES):
        return False
    if scope is None:
        return True
    file = os.path.realpath(file)
    return file.startswith(scope)


def retrieve_code_object(o: typing.Any) -> typing.Optional[types.CodeType]:
    o = inspect.unwrap(o)
    if callable(o):
        o = retrieve_function(o)
    if o is None:
        return None
    if inspect.isfunction(o):
        return o.__code__
    if inspect.isframe(o):
        return o.f_code
    if inspect.iscode(o):
        return o
    raise ValueError(f"Cannot retrieve code object from {o}")


def is_method(c: typing.Callable) -> bool:
    return isinstance(c, types.MethodType) or isinstance(c, types.BuiltinMethodType)


def retrieve_function(o: typing.Any) -> typing.Optional[types.FunctionType]:
    if o is None:
        return None
    o = inspect.unwrap(o)
    if isinstance(o, types.WrapperDescriptorType) or isinstance(o, types.MethodWrapperType) or inspect.isbuiltin(o):
        return None
    if inspect.isfunction(o):
        return o
    if inspect.ismethod(o):
        return o.__func__.__get__(None, object)
    if hasattr(o, "__call__"):
        return retrieve_function(o.__call__)
    raise ValueError(f"Cannot retrieve function from {o}")


def rebuild_ast(o: typing.Any) -> typing.Tuple[ast.AST, ast.AST]:
    if inspect.ismodule(o):
        module = ast.parse(inspect.getsource(o))
        return module, module
    code = retrieve_code_object(o)
    module, _ = rebuild_ast(inspect.getmodule(code))
    finder = visitors.ASTFinder(code)
    node = finder.visit(module)
    if node is None:
        raise ValueError(f"Cannot rebuild ast from {o}")
    return module, node


def retrieve_ast(o: typing.Any) -> ast.AST:
    if inspect.ismodule(o):
        if o not in ast_storage:
            ast_storage[o], _ = rebuild_ast(o)
        return ast_storage[o]
    else:
        c = retrieve_code_object(o)
        if c not in ast_storage:
            module = retrieve_ast(inspect.getmodule(c))
            finder = visitors.ASTFinder(c)
            node = finder.visit(module)
            if node is None:
                raise ValueError(f"Cannot retrieve ast from {o}")
            ast_storage[c] = node
        return ast_storage[c]


def rebuild_symbol_tables(module: ast.AST) -> typing.Mapping[ast.AST, symbols.SymbolTable]:
    symbol_table_builder = visitors.SymbolTableBuilder()
    symbol_table_builder.visit(module)
    load_store_analyzer = visitors.LoadStoreAnalyzer(symbol_table_builder.symbolTables)
    load_store_analyzer.visit(module)
    return load_store_analyzer.symbolTables


def retrieve_symbol_tables(o: typing.Any) -> typing.Mapping[ast.AST, symbols.SymbolTable]:
    module = o if inspect.ismodule(o) else inspect.getmodule(o)
    module_node = retrieve_ast(module)
    if module_node not in symbol_table_storage:
        symbol_table_storage[module_node] = rebuild_symbol_tables(module_node)
    return symbol_table_storage[module_node]


def rebuild_symbol_table(module: ast.AST, node: ast.AST) -> symbols.SymbolTable:
    return rebuild_symbol_tables(module)[node]


def retrieve_symbol_table(o: typing.Any) -> symbols.SymbolTable:
    return retrieve_symbol_tables(o)[retrieve_ast(o)]


def retrieve_cfg(o: typing.Any):
    c = retrieve_code_object(o)
    if c not in cfg_storage:
        builder = visitors.CFGBuilder()
        a = retrieve_ast(c)
        builder.visit(a)
        cfg_storage[c] = builder.result()[a]
    return cfg_storage[c]
