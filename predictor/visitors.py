import ast
import builtins
import copy
import functools
import sys
import types
import typing

from predictor import base
from predictor import cfg
from predictor import expressions
from predictor import symbols
from predictor import utility

Patch = typing.Mapping[ast.AST, ast.AST]
ASTType = ast.AST | typing.Any
if sys.version_info >= (3, 11):
    BranchAST = ast.AsyncWith | ast.If | ast.Match | ast.Try | ast.TryStar | ast.With
else:
    BranchAST = ast.AsyncWith | ast.If | ast.Match | ast.Try | ast.With


class VisitorWithSymbolTable(ast.NodeVisitor):
    def __init__(self, symbol_tables: typing.Optional[typing.Mapping[ast.AST, symbols.SymbolTable]] = None, symbol_table: typing.Optional[symbols.SymbolTable] = None):
        self.symbolTables: typing.Optional[typing.Mapping[ast.AST, symbols.SymbolTable]] = symbol_tables
        self.currentTable: typing.Optional[symbols.SymbolTable] = symbol_table

    def visit(self, node: ASTType):
        match node:
            case ast.Module() | ast.ClassDef() | ast.FunctionDef():
                self.enter_scope(node)
                super().visit(node)
                self.exit_scope(node)
            case _:
                super().visit(node)

    def enter_scope(self, node: ast.AST):
        assert isinstance(node, ast.Module) or isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef)
        if self.symbolTables is not None:
            self.currentTable = self.symbolTables[node]

    def exit_scope(self, node: ast.AST):
        assert isinstance(node, ast.Module) or isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef)
        if self.currentTable is not None:
            self.currentTable = self.currentTable.parent


class SymbolTableBuilder(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.symbolTables: typing.MutableMapping[ast.AST, symbols.SymbolTable] = {}
        self.currentTable: typing.Optional[symbols.SymbolTable] = None

    def result(self) -> typing.Mapping[ast.AST, symbols.SymbolTable]:
        return self.symbolTables

    def visit_ClassDef(self, node: ast.ClassDef):
        symbol = self.currentTable.define_local(node.name)
        symbol.store(node)
        table = self.currentTable.new()
        self.currentTable = table
        self.symbolTables[node] = table
        self.generic_visit(node)
        self.currentTable = self.currentTable.parent

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name is not None:
            self.currentTable.define_local(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        symbol = self.currentTable.define_local(node.name)
        symbol.store(node)
        table = self.currentTable.new()
        for argument in node.args.posonlyargs:
            table.define_local(argument.arg)
        for argument in node.args.args:
            table.define_local(argument.arg)
        for argument in node.args.kwonlyargs:
            table.define_local(argument.arg)
        if node.args.vararg is not None:
            table.define_local(node.args.vararg.arg)
        if node.args.kwarg is not None:
            table.define_local(node.args.kwarg.arg)
        self.currentTable = table
        self.symbolTables[node] = table
        self.generic_visit(node)
        self.currentTable = self.currentTable.parent

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.currentTable.define_global(name)

    def visit_Import(self, node: ast.Import):
        for name in node.names:
            if name.name == "*":
                continue
            symbol = self.currentTable.define_local(name.name)
            if name.asname is not None:
                self.currentTable.define_local(name.asname, alias=symbol)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for name in node.names:
            if name.name == "*":
                continue
            symbol = self.currentTable.define_local(name.name)
            if name.asname is not None:
                self.currentTable.define_local(name.asname, alias=symbol)

    def visit_Module(self, node: ast.Module):
        table = symbols.SymbolTable()
        self.symbolTables[node] = table
        self.currentTable = table
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        match node.ctx:
            case ast.Load():
                self.currentTable.define_free(node.id)
            case ast.Store():
                self.currentTable.define_local(node.id)
            case ast.Del():
                self.currentTable.define_local(node.id)
            case _:
                raise NotImplementedError(f"Unknown context {node.ctx}")

    def visit_Nonlocal(self, node: ast.Nonlocal):
        for name in node.names:
            self.currentTable.define_nonlocal(name)


class LoadStoreAnalyzer(VisitorWithSymbolTable):
    def __init__(self, symbol_tables: typing.Mapping[ast.AST, symbols.SymbolTable]):
        super().__init__(symbol_tables=symbol_tables)
        self.currentStatement: typing.Optional[ast.stmt] = None

    def generic_visit(self, node: ast.AST):
        if isinstance(node, ast.stmt):
            self.currentStatement = node
        super().generic_visit(node)

    def result(self) -> typing.Mapping[ast.AST, symbols.SymbolTable]:
        return self.symbolTables

    def visit_Name(self, node: ast.Name):
        assert self.currentTable is not None
        assert self.currentStatement is not None
        match node.ctx:
            case ast.Load():
                self.currentTable[node.id].load(self.currentStatement)
            case ast.Store():
                self.currentTable[node.id].store(self.currentStatement)
            case ast.Del():
                self.currentTable[node.id].delete(self.currentStatement)
            case _:
                raise NotImplementedError(f"Unknown context {node.ctx}")


class CFGBuilder(ast.NodeVisitor):
    Loop = typing.NamedTuple("Loop", [("loop", ast.stmt), ("breaks", typing.Set[ast.stmt])])
    EmptySet = set()

    def __init__(self):
        super().__init__()
        self.cfgs: typing.MutableMapping[ast.AST, cfg.CFG] = {}
        self.current_statement: typing.Optional[ast.stmt] = None
        self.currentCFG: typing.Optional[cfg.CFG] = None
        self.currentLoop: typing.Optional[CFGBuilder.Loop] = None
        self.exits: typing.Set[ast.stmt] | typing.Optional[ast.stmt] = None

    def connect(self, current_node: ast.stmt):
        assert self.currentCFG is not None
        match self.exits:
            case set():
                for e in self.exits:
                    self.currentCFG.insert_edge(e, current_node)
            case ast.stmt():
                self.currentCFG.insert_edge(self.exits, current_node)
            case _:
                raise NotImplementedError(f"Unknown exits type {self.exits}")
        self.exits = None

    def merge(self, exits: typing.Set[ast.stmt]):
        match self.exits:
            case set():
                exits.update(self.exits)
            case ast.stmt():
                exits.add(self.exits)
            case None:
                pass
            case _:
                raise NotImplementedError(f"Unknown exits type {self.exits}")
        self.exits = None

    def visit_branch(self, node: BranchAST, *branches: typing.Optional[typing.Iterable[ast.stmt]]):
        exits = set()
        for branch in branches:
            if branch is not None:
                self.exits = node
                for s in branch:
                    self.visit(s)
                self.merge(exits)
        self.exits = exits

    def visit_exit(self, node: ast.stmt):
        self.exits = CFGBuilder.EmptySet
        self.currentCFG.define_exit(node)

    def visit_loop(self, node: ast.AsyncFor | ast.For | ast.While):
        exits = set()
        breaks = set()
        current_loop = self.currentLoop
        self.currentLoop = CFGBuilder.Loop(node, breaks)
        self.exits = node
        for s in node.body:
            self.visit(s)
        self.merge(exits)
        for e in exits:
            self.currentCFG.insert_edge(e, node)
        self.exits = node
        self.currentLoop = current_loop
        if node.orelse is not None:
            for s in node.orelse:
                self.visit(s)
        self.merge(breaks)
        self.exits = breaks

    def visit_scope(self, node: ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef | ast.Module):
        current_cfg = self.currentCFG
        self.currentCFG = cfg.CFG()
        self.cfgs[node] = self.currentCFG
        self.generic_visit(node)
        match self.exits:
            case set():
                self.currentCFG.define_exits(self.exits)
            case ast.stmt():
                self.currentCFG.define_exit(self.exits)
            case _:
                raise NotImplementedError(f"Unknown exits type {self.exits}")
        self.currentCFG = current_cfg
        self.exits = node

    def result(self) -> typing.Mapping[ast.AST, cfg.CFG]:
        return self.cfgs

    def visit(self, node: ASTType):
        if isinstance(node, ast.stmt):
            self.current_statement = node
            if self.exits is not None:
                self.connect(node)
        super().visit(node)
        if self.exits is None and isinstance(node, ast.stmt):
            self.exits = node

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.visit_loop(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_scope(node)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self.visit_branch(node, node.body)

    def visit_Break(self, node: ast.Break):
        assert self.currentLoop is not None
        self.currentLoop.breaks.add(node)
        self.exits = CFGBuilder.EmptySet

    def visit_ClassDef(self, node: ast.ClassDef):
        self.visit_scope(node)

    def visit_Continue(self, node: ast.Continue):
        assert self.currentLoop is not None
        self.currentCFG.insert_edge(node, self.currentLoop.loop)
        self.exits = CFGBuilder.EmptySet

    def visit_For(self, node: ast.For):
        self.visit_loop(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.visit_scope(node)

    def visit_If(self, node: ast.If):
        self.visit_branch(node, node.body, node.orelse)

    def visit_Match(self, node: ast.Match):
        self.visit_branch(node, *(case.body for case in node.cases))

    def visit_Module(self, node: ast.Module):
        self.visit_scope(node)

    def visit_Return(self, node: ast.Return):
        self.visit_exit(node)

    def visit_Try(self, node: ast.Try):
        branches = []
        for handler in node.handlers:
            branches.append([*node.body, *handler.body])
        if node.orelse is None:
            branches.append(node.body)
        else:
            branches.append([*node.body, *node.orelse])
        self.visit_branch(node, *branches)
        for n in node.finalbody:
            self.visit(n)

    if sys.version_info >= (3, 11):
        def visit_TryStar(self, node: ast.TryStar):
            branches = []
            for handler in node.handlers:
                branches.append([*node.body, *handler.body])
            if node.orelse is None:
                branches.append(node.body)
            else:
                branches.append([*node.body, *node.orelse])
            self.visit_branch(node, *branches)
            for n in node.finalbody:
                self.visit(n)

    def visit_While(self, node: ast.While):
        self.visit_loop(node)

    def visit_With(self, node: ast.With):
        self.visit_branch(node, node.body)

    def visit_Yield(self, node: ast.Yield):
        assert self.current_statement is not None
        self.visit_exit(self.current_statement)

    def visit_YieldFrom(self, node: ast.YieldFrom):
        assert self.current_statement is not None
        self.visit_exit(self.current_statement)


class ExpressionBuilder(VisitorWithSymbolTable):
    def __init__(self, symbol_table: symbols.SymbolTable):
        super().__init__(symbol_table=symbol_table)
        self.expression: typing.Optional[expressions.Expression] = None

    def visit(self, node: ASTType):
        super().visit(node)
        return self.expression

    def generic_visit(self, node):
        raise NotImplementedError(f"Unknown node {node}")

    def visit_Attribute(self, node: ast.Attribute):
        self.expression = expressions.Attribute(self.visit(node.value), node.attr, utility.convert_context(node.ctx))

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        match node.op:
            case ast.Add():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.ADD)
            case ast.Sub():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.SUBTRACT)
            case ast.Mult():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.MULTIPLY)
            case ast.Div():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.TRUE_DIVIDE)
            case ast.FloorDiv():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.FLOOR_DIVIDE)
            case ast.Mod():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.MODULO)
            case ast.Pow():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.POWER)
            case ast.LShift():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.LEFT_SHIFT)
            case ast.RShift():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.RIGHT_SHIFT)
            case ast.BitOr():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.BITWISE_OR)
            case ast.BitXor():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.BITWISE_XOR)
            case ast.BitAnd():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.BITWISE_AND)
            case ast.MatMult():
                self.expression = expressions.BinaryOperation(left, right, expressions.BinaryOperator.MATRIX_MULTIPLY)
            case _:
                raise NotImplementedError(f"Unknown binary operator {node.op}")

    def visit_BoolOp(self, node: ast.BoolOp):
        values = [self.visit(value) for value in node.values]
        match node.op:
            case ast.And():
                op = expressions.BinaryOperator.LOGICAL_AND
            case ast.Or():
                op = expressions.BinaryOperator.LOGICAL_OR
            case _:
                raise NotImplementedError(f"Unknown binary operator {node.op}")
        builder: typing.Callable[[typing.Any, typing.Any], typing.Any] = lambda v1, v2: expressions.BinaryOperation(v1, v2, op)
        self.expression = functools.reduce(builder, values)

    def visit_Call(self, node: ast.Call):
        args = [self.visit(argument) for argument in node.args]
        kwargs = {keyword.arg: self.visit(keyword.value) for keyword in node.keywords}
        self.expression = expressions.FunctionCall(self.visit(node.func), args, kwargs)

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        comparators = [self.visit(comparator) for comparator in node.comparators]
        operations = []
        assert len(node.ops) == len(comparators)
        for right, op in zip(comparators, node.ops):
            match op:
                case ast.Eq():
                    operations.append(expressions.BinaryOperation(left, right, expressions.BinaryOperator.EQUAL))
                case ast.NotEq():
                    operations.append(expressions.BinaryOperation(left, right, expressions.BinaryOperator.NOT_EQUAL))
                case ast.Lt():
                    operations.append(expressions.BinaryOperation(left, right, expressions.BinaryOperator.LESS_THAN))
                case ast.LtE():
                    operations.append(expressions.BinaryOperation(left, right, expressions.BinaryOperator.LESS_EQUAL))
                case ast.Gt():
                    operations.append(expressions.BinaryOperation(left, right, expressions.BinaryOperator.GREATER_THAN))
                case ast.GtE():
                    operations.append(expressions.BinaryOperation(left, right, expressions.BinaryOperator.GREATER_EQUAL))
                case ast.Is():
                    operations.append(expressions.keyword_is(left, right))
                case ast.IsNot():
                    operations.append(expressions.keyword_is_not(left, right))
                case ast.In():
                    operations.append(expressions.Contains(right, left))
                case ast.NotIn():
                    operations.append(expressions.logical_not(expressions.Contains(right, left)))
                case _:
                    raise NotImplementedError(f"Unknown binary operator {node.ops[0]}")
            left = right
        self.expression = functools.reduce(expressions.logical_and, operations)

    def visit_Constant(self, node: ast.Constant):
        self.expression = node.value

    def visit_comprehension(self, node: ast.comprehension):
        self.expression = expressions.Comprehension(self.visit(node.target), self.visit(node.iter), (self.visit(i) for i in node.ifs), node.is_async > 0)

    def visit_Dict(self, node: ast.Dict):
        keys = (self.visit(key) for key in node.keys)
        values = (self.visit(value) for value in node.values)
        self.expression = expressions.Dictionary(keys, values)

    def visit_DictComp(self, node: ast.DictComp):
        key = self.visit(node.key)
        value = self.visit(node.value)
        generators: typing.Iterable[expressions.Comprehension] = (self.visit(g) for g in node.generators)  # type: ignore
        self.expression = expressions.DictionaryComprehension(key, value, generators)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        value = self.visit(node.value)
        format_specification = self.visit(node.format_spec) if node.format_spec is not None else None
        self.expression = expressions.Format(value, node.conversion, format_specification)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        element = self.visit(node.elt)
        generators: typing.Iterable[expressions.Comprehension] = (self.visit(g) for g in node.generators)  # type: ignore
        self.expression = expressions.GeneratorComprehension(element, generators)

    def visit_IfExp(self, node: ast.IfExp):
        self.expression = expressions.Branch(self.visit(node.test), self.visit(node.body), self.visit(node.orelse))

    def visit_JoinedStr(self, node: ast.JoinedStr):
        self.expression = expressions.Join([self.visit(value) for value in node.values])

    def visit_List(self, node: ast.List):
        self.expression = expressions.List([self.visit(element) for element in node.elts], utility.convert_context(node.ctx))

    def visit_ListComp(self, node: ast.ListComp):
        element = self.visit(node.elt)
        generators: typing.Iterable[expressions.Comprehension] = (self.visit(g) for g in node.generators)  # type: ignore
        self.expression = expressions.ListComprehension(element, generators)

    def visit_Name(self, node: ast.Name):
        symbol: symbols.Symbol = self.currentTable[node.id]
        assert symbol is not None
        if symbol.is_builtin:
            self.expression = getattr(builtins, symbol.name)
        elif symbol.in_global():
            self.expression = expressions.Global(symbol.name, utility.convert_context(node.ctx))
        else:
            self.expression = expressions.Local(symbol.name, utility.convert_context(node.ctx))

    def visit_Set(self, node: ast.Set):
        self.expression = expressions.Set(self.visit(element) for element in node.elts)

    def visit_SetComp(self, node: ast.SetComp):
        element = self.visit(node.elt)
        generators: typing.Iterable[expressions.Comprehension] = (self.visit(g) for g in node.generators)  # type: ignore
        self.expression = expressions.SetComprehension(element, generators)

    def visit_Slice(self, node: ast.Slice):
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        self.expression = expressions.Slice(lower, upper, step)

    def visit_Starred(self, node: ast.Starred):
        self.expression = expressions.Starred(self.visit(node.value))

    def visit_Subscript(self, node: ast.Subscript):
        target = self.visit(node.value)
        index = self.visit(node.slice)
        self.expression = expressions.Item(target, index, utility.convert_context(node.ctx))

    def visit_Tuple(self, node: ast.Tuple):
        self.expression = expressions.Tuple((self.visit(element) for element in node.elts), utility.convert_context(node.ctx))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        match node.op:
            case ast.Invert():
                self.expression = expressions.UnaryOperation(operand, expressions.UnaryOperator.INVERT)
            case ast.Not():
                self.expression = expressions.logical_not(operand)
            case ast.UAdd():
                self.expression = expressions.UnaryOperation(operand, expressions.UnaryOperator.POSITIVE)
            case ast.USub():
                self.expression = expressions.UnaryOperation(operand, expressions.UnaryOperator.NEGATIVE)
            case _:
                raise NotImplementedError(f"Unknown unary operator {node.op}")


class ASTFinder(ast.NodeVisitor):
    def __init__(self, target: types.CodeType):
        super().__init__()
        self.target = target
        self.found: typing.Optional[ast.AST] = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.lineno >= self.target.co_firstlineno and node.name == self.target.co_name:
            self.found = node
            return self.found
        super().generic_visit(node)
        return self.found

    def visit(self, node: ASTType):
        super().visit(node)
        return self.found

    def generic_visit(self, node):
        if self.found is not None:
            return self.found
        return super().generic_visit(node)


class Patcher(ast.NodeVisitor):
    def __init__(self, patch: Patch):
        super().__init__()
        self.patch = patch

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                setattr(node, field, new_values)
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit(self, node: ASTType):
        node = self.patch[node] if node in self.patch else copy.copy(node)
        super().visit(node)
        return node


HookFunction = ast.Attribute(
    value=ast.Subscript(
        value=ast.Attribute(
            value=ast.Call(
                func=ast.Name(id="__import__", ctx=ast.Load()),
                args=[ast.Constant("sys")], keywords=[]),
            attr="modules",
            ctx=ast.Load()
        ),
        slice=ast.Constant("predictor.analyzer"),
        ctx=ast.Load()
    ),
    attr="hook",
    ctx=ast.Load())


def handle_assignment(target: expressions.Expression, value: expressions.Expression, modifications: typing.MutableMapping[expressions.Expression, expressions.MaybeExpression]):
    match target:
        case expressions.Tuple() | expressions.List():
            items = expressions.retrieve_attribute(target, "items")
            for index, item in enumerate(items):
                handle_assignment(item, value[index], modifications)
        case expressions.Attribute() | expressions.Item() | expressions.Local() | expressions.Global():
            modifications[target] = value
        case _:
            raise NotImplementedError(f"Unknown target {target}")


class Hooker(VisitorWithSymbolTable):
    def __init__(self, symbol_tables: typing.Mapping[ast.AST, symbols.SymbolTable]):
        super().__init__(symbol_tables)
        self.hooks: typing.MutableMapping[ast.AST, typing.Set[base.HookIndex]] = {}
        self.patch = {}
        self.exit_statements = set()
        self.last_statement: typing.Optional[ast.AST] = None

    def visit(self, node):
        if isinstance(node, ast.expr):
            if hasattr(node, "ctx") and not isinstance(node.ctx, ast.Load):
                return  # left expressions
            hooked = self.hook(node)
            if hooked is not node:
                self.patch[node] = hooked
            return
        if isinstance(node, ast.stmt):
            self.last_statement = node
            if isinstance(node, ast.Return):
                self.exit_statements.add(node)
        if isinstance(node, ast.FunctionDef) and len(node.body) > 0:
            self.exit_statements.add(node.body[-1])
        super().visit(node)

    def hook(self, node: ast.expr) -> ast.AST | ast.Call:
        expression_builder = ExpressionBuilder(self.currentTable)
        expression = expression_builder.visit(node)
        if not expressions.is_expression(expression):
            return node
        leaves = expression.leaves()
        hook_id = len(base.hooks)
        if self.last_statement not in self.hooks:
            self.hooks[self.last_statement] = set()
        self.hooks[self.last_statement].add(hook_id)
        parameters = [ast.Name(id=symbol.name, ctx=ast.Load()) for leaf in leaves if (symbol := self.currentTable[expressions.variable_name(leaf)]).is_free]
        modifications: typing.MutableMapping[expressions.Expression, expressions.MaybeExpression] = {}
        match self.last_statement:
            case ast.Assign() as s:
                for t in s.targets:
                    handle_assignment(expression_builder.visit(t), expression, modifications)
            case ast.AugAssign() as s:
                handle_assignment(expression_builder.visit(s.target), expression, modifications)
            case _:
                pass
        base.hooks[hook_id] = base.HookInfo(hook_id, expression, modifications, self.last_statement, is_exit=self.last_statement in self.exit_statements)
        return ast.Call(func=ast.Call(func=HookFunction, args=parameters, keywords=[]), args=[ast.Constant(hook_id)], keywords=[])
