from __future__ import annotations

import types
import typing

import torch.distributed

from .. import expressions, codes, analyzer, utility

ClassicLosses = (
    torch.nn.L1Loss, torch.nn.NLLLoss, torch.nn.NLLLoss2d, torch.nn.PoissonNLLLoss, torch.nn.GaussianNLLLoss, torch.nn.KLDivLoss,
    torch.nn.MSELoss, torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss, torch.nn.HingeEmbeddingLoss, torch.nn.MultiLabelMarginLoss,
    torch.nn.SmoothL1Loss, torch.nn.HuberLoss, torch.nn.SoftMarginLoss, torch.nn.CrossEntropyLoss, torch.nn.MultiLabelSoftMarginLoss,
    torch.nn.CosineEmbeddingLoss, torch.nn.MarginRankingLoss, torch.nn.MultiMarginLoss, torch.nn.TripletMarginLoss,
    torch.nn.TripletMarginWithDistanceLoss, torch.nn.CTCLoss
)

BinaryArithmeticOperators = {
    expressions.BinaryOperator.ADD,
    expressions.BinaryOperator.MATRIX_MULTIPLY,
    expressions.BinaryOperator.MULTIPLY,
    expressions.BinaryOperator.SUBTRACT,
    expressions.BinaryOperator.TRUE_DIVIDE
}

DDP_MODEL = "ddp_model"


def ddp_hook(environment: analyzer.AnalyzeEnvironment, expression: expressions.Expression, operator: types.CodeType, function: typing.Any, arguments, keyword_arguments):
    if not isinstance(function, torch.nn.parallel.DistributedDataParallel):
        return
    modified, result = environment.result()
    if not modified:
        result = function(*arguments, **keyword_arguments)
        environment.modify(result)
    function_expression = expressions.retrieve_attribute(expression, "function")
    match result:
        case torch.Tensor():
            result.ddp_model = function
            environment.trace_relation(expression.ddp_model, function_expression)
            environment.trace_forward(expression.ddp_model)
        case list() | tuple():
            for i, r in enumerate(result):
                r.ddp_model = function
                environment.trace_relation(expression[i].ddp_model, function_expression)
                environment.trace_forward(expression[i].ddp_model)
        case _:
            raise TypeError(f"Unsupported type {type(result)}")


def attribute_hook(environment: analyzer.AnalyzeEnvironment, expression: expressions.Expression, operator: typing.Any, target: typing.Any):
    if not isinstance(target, torch.Tensor):
        return
    environment.trace_backward_relation(expression.__self__, expressions.retrieve_attribute(expression, "target"))


def amp_scale_hook(environment: analyzer.AnalyzeEnvironment, expression: expressions.Expression, operator: types.CodeType, function: typing.Any, arguments, keyword_arguments):
    modified, result = environment.result()
    if not modified:
        result = function(*arguments, **keyword_arguments)
        environment.modify(result)
    args = expressions.retrieve_attribute(expression, "args")
    kwargs = dict(expressions.retrieve_attribute(expression, "kwargs"))
    loss = utility.retrieve_positional_or_keyword_argument(0, "outputs")(*arguments, **keyword_arguments)
    loss_expression = utility.retrieve_positional_or_keyword_argument(0, "outputs")(*args, **kwargs)
    result.ddp_model = getattr(loss, DDP_MODEL)
    environment.trace_relation(expression.ddp_model, loss_expression.ddp_model)


def loss_hook(environment: analyzer.AnalyzeEnvironment, expression: expressions.Expression, operator: types.CodeType, function: typing.Any, arguments, keyword_arguments):
    if not isinstance(function, ClassicLosses):
        return
    input_tensor = utility.retrieve_positional_or_keyword_argument(0, "input", None)(*arguments, **keyword_arguments)
    target_tensor = utility.retrieve_positional_or_keyword_argument(1, "target", None)(*arguments, **keyword_arguments)
    input_tensor_qualified = isinstance(input_tensor, torch.Tensor) and hasattr(input_tensor, DDP_MODEL)
    target_tensor_qualified = isinstance(target_tensor, torch.Tensor) and hasattr(target_tensor, DDP_MODEL)
    if not input_tensor_qualified and not target_tensor_qualified:
        return
    if input_tensor_qualified and target_tensor_qualified:
        raise ValueError("Input and target tensors used to calculate loss cannot be both tensors from DDP.")
    modified, result = environment.result()
    if not modified:
        result = function(*arguments, **keyword_arguments)
        environment.modify(result)
    args = expressions.retrieve_attribute(expression, "args")
    kwargs = dict(expressions.retrieve_attribute(expression, "kwargs"))
    input_expression = utility.retrieve_positional_or_keyword_argument(0, "input", None)(*args, **kwargs)
    target_expression = utility.retrieve_positional_or_keyword_argument(1, "target", None)(*args, **kwargs)
    if input_tensor_qualified:
        result.ddp_model = getattr(input_tensor, DDP_MODEL)
        environment.trace_relation(expression.ddp_model, input_expression.ddp_model)
    if target_tensor_qualified:
        result.ddp_model = getattr(target_tensor, DDP_MODEL)
        environment.trace_relation(expression.ddp_model, target_expression.ddp_model)


def ddp_tensor_method_hook(environment: analyzer.AnalyzeEnvironment, expression: expressions.Expression, operator: types.CodeType, function: typing.Any, arguments, keyword_arguments):
    if not codes.is_method(function):
        return
    if not hasattr(function, "__self__"):
        return
    t = function.__self__
    if not isinstance(t, torch.Tensor):
        return
    if not hasattr(t, DDP_MODEL):
        return
    tensor = expressions.retrieve_attribute(expression, "function").__self__
    match function.__name__:
        case "backward":
            environment.trace_backward(expressions.Tuple(("pytorch tensor backward", expressions.call(model_parameters_size)(tensor.ddp_model)), expressions.Context.Load))
        case "sum":
            modified, result = environment.result()
            if not modified:
                result = function(*arguments, **keyword_arguments)
                environment.modify(result)
            result.ddp_model = getattr(t, DDP_MODEL)
            environment.trace_relation(expression.ddp_model, tensor.ddp_model)


def binary_tensor_operation_hook(environment: analyzer.AnalyzeEnvironment, expression: expressions.Expression, operator: expressions.BinaryOperator, lhs: typing.Any, rhs: typing.Any):
    if operator not in BinaryArithmeticOperators:
        return
    model = None
    tensor = None
    if isinstance(lhs, torch.Tensor) and hasattr(lhs, DDP_MODEL):
        model = getattr(lhs, DDP_MODEL)
        tensor = expressions.retrieve_attribute(expression, "lhs")
    if isinstance(rhs, torch.Tensor) and hasattr(rhs, DDP_MODEL):
        model = getattr(rhs, DDP_MODEL)
        tensor = expressions.retrieve_attribute(expression, "rhs")
    if model is None:
        return
    modified, result = environment.result()
    if not modified:
        operator: expressions.BinaryOperator = expressions.retrieve_attribute(expression, "operator")
        result = operator.apply(lhs, rhs)
        environment.modify(result)
    result.ddp_model = model
    environment.trace_relation(expression.ddp_model, tensor.ddp_model)


def register_rules():
    analyzer.define_light_rule(t=expressions.FunctionCall, o=codes.retrieve_code_object(torch.nn.parallel.DistributedDataParallel.__call__), r=ddp_hook)
    analyzer.define_heavy_rule(t=expressions.FunctionCall, o=None, r=ddp_tensor_method_hook)
    analyzer.define_heavy_rule(t=expressions.FunctionCall, o=None, r=loss_hook)
    analyzer.define_heavy_rule(t=expressions.FunctionCall, o=codes.retrieve_code_object(torch.cuda.amp.GradScaler.scale), r=amp_scale_hook)
    analyzer.define_heavy_rule(t=expressions.Attribute, o=None, r=attribute_hook)
    for operator in BinaryArithmeticOperators:
        analyzer.define_heavy_rule(t=expressions.BinaryOperation, o=operator, r=binary_tensor_operation_hook)


def enable_pytorch_rules(c: typing.Callable) -> typing.Callable:
    register_rules()
    return c


def model_parameters_size(m: torch.nn.parallel.DistributedDataParallel) -> int:
    assert isinstance(m, torch.nn.parallel.DistributedDataParallel)
    if not hasattr(m, "parameter_size"):
        # noinspection PyProtectedMember
        parameters = m._module_parameters
        size = 0
        for p in parameters:
            if p.grad is None:
                size += p.nelement() * p.element_size()
            else:
                size += p.grad.nelement() * p.grad.element_size()
        m.parameter_size = size
    return m.parameter_size


def multiply(size: typing.Iterable[int] | int):
    match size:
        case int():
            return size
        case _:
            result = 1
            for s in size:
                result *= s
            return result
