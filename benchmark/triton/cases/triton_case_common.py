from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:
    torch = None
    torch_f = None

try:
    import triton  # noqa: F401
except ImportError:
    triton = None


SYMBOL_VALUES = {
    "B": 8,
    "C": 16,
    "C1": 16,
    "C2": 8,
    "D": 128,
    "D1": 64,
    "D2": 96,
    "E": 128,
    "E1": 64,
    "E2": 96,
    "H": 30,
    "N": 8,
    "P": 24,
    "S": 20,
    "S1": 8,
    "S2": 12,
    "T": 16,
    "V": 256,
    "V1": 128,
    "V2": 64,
    "W": 30,
}

MAX_EXECUTION_ELEMENTS = 32_000_000
ATOM_RE = re.compile(r"[A-Z][A-Za-z0-9]*|\d+")


@dataclass(frozen=True)
class ValueSpec:
    token: str
    dims: tuple[str, ...]
    is_scalar: bool = False

    def contract(self) -> str:
        if self.is_scalar:
            return self.token
        return f"[{' x '.join(self.dims)}]"


@dataclass(frozen=True)
class CaseSpec:
    category: str
    case_name: str
    scenario: str
    inputs: tuple[ValueSpec, ...]
    output: ValueSpec
    params: tuple[str, ...] = ()


def product(values: tuple[int, ...] | list[int]) -> int:
    return reduce(mul, values, 1)


def parse_case(case_path: str | Path) -> CaseSpec:
    path = Path(case_path)
    category = path.parent.name
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unsupported case name: {path.name}")

    case_name = path.stem
    scenario = parts[1]
    remainder = parts[2:]

    if category == "batch_norm":
        input_tokens = remainder[:3]
        output_token = remainder[3]
        inputs = tuple(ValueSpec(token, split_dims(token)) for token in input_tokens)
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    if category == "concat":
        input_tokens = remainder[:-1]
        output_token = remainder[-1]
        inputs = tuple(ValueSpec(token, split_dims(token)) for token in input_tokens)
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    if category == "conv2d":
        if len(remainder) != 6:
            raise ValueError(f"Unexpected conv2d shape layout: {path.name}")
        input_token, weight_token, output_token, stride, padding, dilation = remainder
        inputs = (
            ValueSpec(input_token, split_dims(input_token)),
            ValueSpec(weight_token, split_dims(weight_token)),
        )
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output, (stride, padding, dilation))

    if category == "elemwise_add":
        lhs_token, rhs_token, output_token = remainder
        rhs_spec = ValueSpec(rhs_token, () if is_scalar_token(rhs_token) else split_dims(rhs_token), is_scalar_token(rhs_token))
        inputs = (
            ValueSpec(lhs_token, split_dims(lhs_token)),
            rhs_spec,
        )
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    if category == "embedding":
        id_token, weight_token, output_token = remainder
        inputs = (
            ValueSpec(id_token, split_dims(id_token)),
            ValueSpec(weight_token, split_dims(weight_token)),
        )
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    if category in {"gelu", "sigmoid"}:
        input_token = remainder[0]
        value = ValueSpec(input_token, split_dims(input_token))
        return CaseSpec(category, case_name, scenario, (value,), value)

    if category in {"relu", "softmax"}:
        input_token = remainder[0]
        output_token = remainder[1] if len(remainder) > 1 else input_token
        inputs = (ValueSpec(input_token, split_dims(input_token)),)
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    if category == "layer_normalization":
        input_token, weight_token, bias_token = remainder
        inputs = tuple(ValueSpec(token, split_dims(token)) for token in (input_token, weight_token, bias_token))
        output = ValueSpec(input_token, split_dims(input_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    if category == "matmul":
        input_tokens = remainder[:2]
        output_token = remainder[2]
        inputs = tuple(ValueSpec(token, split_dims(token)) for token in input_tokens)
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    if category in {"max_pool2d", "reduce_mean", "reshape", "transpose"}:
        input_token, output_token = remainder
        inputs = (ValueSpec(input_token, split_dims(input_token)),)
        output = ValueSpec(output_token, split_dims(output_token))
        return CaseSpec(category, case_name, scenario, inputs, output)

    raise ValueError(f"Unsupported category: {category}")


def split_dims(token: str) -> tuple[str, ...]:
    return tuple(part for part in token.split("x") if part)


def is_scalar_token(token: str) -> bool:
    return token.isdigit()


def env_for_case(case: CaseSpec) -> dict[str, int]:
    env = dict(SYMBOL_VALUES)
    for value in case.inputs + (case.output,):
        if value.is_scalar:
            continue
        for dim in value.dims:
            for symbol in symbols_in_expr(dim):
                env.setdefault(symbol, SYMBOL_VALUES.get(symbol, 8))
    return env


def symbols_in_expr(expr: str) -> list[str]:
    return [match.group(0) for match in ATOM_RE.finditer(expr) if match.group(0)[0].isalpha()]


def evaluate_shape(value: ValueSpec, env: dict[str, int]) -> tuple[int, ...]:
    return tuple(evaluate_dim(dim, env) for dim in value.dims)


def evaluate_dim(expr: str, env: dict[str, int]) -> int:
    total = 0
    for term in expr.split("p"):
        if not term:
            raise ValueError(f"Invalid dimension expression: {expr}")
        index = 0
        product_value = 1
        while index < len(term):
            atom, index = parse_atom(term, index)
            value = atom_value(atom, env)
            if index < len(term) and term[index] == "d":
                divisor_atom, index = parse_atom(term, index + 1)
                divisor = atom_value(divisor_atom, env)
                value //= divisor
            product_value *= value
        total += product_value
    return total


def parse_atom(term: str, index: int) -> tuple[str, int]:
    match = ATOM_RE.match(term, index)
    if not match:
        raise ValueError(f"Invalid term '{term}'")
    return match.group(0), match.end()


def atom_value(atom: str, env: dict[str, int]) -> int:
    if atom.isdigit():
        return int(atom)
    if atom not in env:
        raise KeyError(f"Missing symbol '{atom}'")
    return env[atom]


def format_contract(case: CaseSpec) -> str:
    input_contract = " + ".join(value.contract() for value in case.inputs)
    return f"{input_contract} -> {case.output.contract()}"


def mode_for_category(category: str) -> str:
    if category in {"reshape", "transpose"}:
        return "triton-host-view"
    if category == "matmul":
        return "triton-host-matmul"
    return "triton-host-torch"


def describe_case(case_path: str | Path) -> dict[str, Any]:
    case = parse_case(case_path)
    env = env_for_case(case)
    try:
        assertions, notes = validate_case(case, env)
        valid_contract = True
    except ValueError as error:
        assertions = []
        notes = [f"contract-invalid {error}"]
        valid_contract = False
    return {
        "case_name": case.case_name,
        "category": case.category,
        "mode": mode_for_category(case.category),
        "explicit_assertions": len(assertions),
        "input_shape": " + ".join(value.contract() for value in case.inputs),
        "output_shape": case.output.contract(),
        "contract": format_contract(case),
        "valid_contract": valid_contract,
        "notes": "; ".join(notes),
    }


def validate_case(case: CaseSpec, env: dict[str, int]) -> tuple[list[str], list[str]]:
    assertions: list[str] = []
    notes: list[str] = []

    input_shapes = [evaluate_shape(value, env) for value in case.inputs if not value.is_scalar]
    output_shape = evaluate_shape(case.output, env)

    def add_assert(condition: bool, message: str) -> None:
        assertions.append(message)
        if not condition:
            raise ValueError(f"{case.case_name}: {message}")

    if case.category in {"gelu", "relu", "sigmoid", "softmax"}:
        input_shape = input_shapes[0]
        add_assert(len(input_shape) >= 1, "activation input rank must be positive")
        add_assert(output_shape == input_shape, "activation output must match input shape")
        if case.category == "softmax":
            add_assert(input_shape[-1] > 0, "softmax last dimension must be positive")
            notes.append("softmax over last dimension")
        return assertions, notes

    if case.category == "elemwise_add":
        lhs_shape = input_shapes[0]
        rhs_shape = () if case.inputs[1].is_scalar else input_shapes[1]
        broadcast_shape = infer_broadcast_shape(lhs_shape, rhs_shape)
        add_assert(len(lhs_shape) >= 1, "elementwise add lhs rank must be positive")
        if rhs_shape:
            add_assert(broadcast_shape is not None, "elementwise add inputs must be broadcast compatible")
        add_assert(output_shape == broadcast_shape, "elementwise add output must match broadcast result")
        return assertions, notes

    if case.category == "matmul":
        lhs_shape, rhs_shape = input_shapes
        add_assert(len(lhs_shape) >= 2, "matmul lhs rank must be at least 2")
        add_assert(len(rhs_shape) >= 2, "matmul rhs rank must be at least 2")
        add_assert(lhs_shape[-1] == rhs_shape[-2], "matmul inner dimensions must match")
        expected = infer_matmul_output_shape(lhs_shape, rhs_shape)
        add_assert(expected is not None, "matmul batch dimensions must be broadcast compatible")
        add_assert(output_shape == expected, "matmul output shape must match broadcasted contract")
        return assertions, notes

    if case.category == "reshape":
        input_shape = input_shapes[0]
        add_assert(product(input_shape) == product(output_shape), "reshape must preserve element count")
        return assertions, notes

    if case.category == "transpose":
        input_shape = input_shapes[0]
        permutation = infer_permutation(case.inputs[0].dims, case.output.dims)
        add_assert(permutation is not None, "transpose output must be a permutation of input dimensions")
        if permutation is not None:
            expected = tuple(input_shape[index] for index in permutation)
            add_assert(output_shape == expected, "transpose output must equal permuted input shape")
            notes.append(f"perm={'.'.join(str(index) for index in permutation)}")
        return assertions, notes

    if case.category == "concat":
        add_assert(len(input_shapes) >= 2, "concat requires at least two inputs")
        add_assert(all(len(shape) == len(input_shapes[0]) for shape in input_shapes), "concat inputs must share rank")
        axis = infer_concat_axis(input_shapes, output_shape)
        add_assert(axis is not None, "concat output must differ along exactly one axis")
        if axis is not None:
            for axis_index in range(len(output_shape)):
                if axis_index == axis:
                    continue
                add_assert(all(shape[axis_index] == input_shapes[0][axis_index] for shape in input_shapes), "concat non-concatenated dimensions must agree")
            add_assert(output_shape[axis] == sum(shape[axis] for shape in input_shapes), "concat output size must equal summed input size on concatenated axis")
            notes.append(f"axis={axis}")
        return assertions, notes

    if case.category == "reduce_mean":
        input_shape = input_shapes[0]
        reduction = infer_reduction(case.inputs[0].dims, case.output.dims)
        add_assert(reduction is not None, "reduce_mean output must derive from removing or keeping reduced dimensions")
        if reduction is not None:
            axes, keepdim = reduction
            add_assert(len(axes) > 0, "reduce_mean must reduce at least one dimension")
            computed = apply_reduction_shape(input_shape, axes, keepdim)
            add_assert(output_shape == computed, "reduce_mean output shape must match reduced contract")
            notes.append(f"axes={'.'.join(str(axis) for axis in axes)}")
            if keepdim:
                notes.append("keepdim")
        return assertions, notes

    if case.category == "embedding":
        id_shape, weight_shape = input_shapes
        add_assert(len(id_shape) == 2, "embedding ids must be rank 2")
        add_assert(len(weight_shape) == 2, "embedding weight must be rank 2")
        add_assert(len(output_shape) == 3, "embedding output must be rank 3")
        add_assert(output_shape[:2] == id_shape, "embedding output leading dimensions must match ids")
        add_assert(output_shape[2] == weight_shape[1], "embedding hidden dimension must match weight width")
        return assertions, notes

    if case.category == "batch_norm":
        input_shape, scale_shape, bias_shape = input_shapes
        add_assert(len(input_shape) >= 3, "batch_norm input rank must be at least 3")
        add_assert(len(scale_shape) == 1, "batch_norm scale must be rank 1")
        add_assert(len(bias_shape) == 1, "batch_norm bias must be rank 1")
        add_assert(scale_shape == bias_shape, "batch_norm scale and bias shapes must match")
        axis = infer_batch_norm_axis(input_shape, scale_shape)
        add_assert(axis is not None, "batch_norm parameter length must match a non-batch axis")
        add_assert(output_shape == input_shape, "batch_norm output must match input shape")
        if axis is not None:
            notes.append(f"param-axis={axis}")
        notes.append("normalize across batch axis")
        return assertions, notes

    if case.category == "layer_normalization":
        input_shape, scale_shape, bias_shape = input_shapes
        add_assert(len(scale_shape) == len(bias_shape), "layer_norm scale and bias ranks must match")
        add_assert(scale_shape == bias_shape, "layer_norm scale and bias shapes must match")
        add_assert(len(scale_shape) < len(input_shape), "layer_norm normalized rank must be smaller than input rank")
        add_assert(output_shape == input_shape, "layer_norm output must match input shape")
        add_assert(tuple(input_shape[-len(scale_shape):]) == scale_shape, "layer_norm normalized shape must match input suffix")
        return assertions, notes

    if case.category == "conv2d":
        input_shape, weight_shape = input_shapes
        add_assert(len(input_shape) == 4, "conv2d input must be NCHW rank 4")
        add_assert(len(weight_shape) == 4, "conv2d weight must be OIHW rank 4")
        add_assert(input_shape[1] == weight_shape[1], "conv2d input channels must match weight channels")
        stride, padding, dilation = infer_conv2d_params(case, input_shape, weight_shape, output_shape)
        add_assert(stride is not None, "conv2d stride padding dilation must be inferable")
        if stride is not None and padding is not None and dilation is not None:
            expected = conv2d_output_shape(input_shape, weight_shape, stride, padding, dilation)
            add_assert(expected == output_shape, "conv2d output must satisfy convolution formula")
            notes.append(f"stride={stride}")
            notes.append(f"padding={padding}")
            notes.append(f"dilation={dilation}")
        return assertions, notes

    if case.category == "max_pool2d":
        input_shape = input_shapes[0]
        add_assert(len(input_shape) == 4, "max_pool2d input must be NCHW rank 4")
        add_assert(len(output_shape) == 4, "max_pool2d output must be NCHW rank 4")
        add_assert(output_shape[0] == input_shape[0], "max_pool2d batch dimension must be preserved")
        add_assert(output_shape[1] == input_shape[1], "max_pool2d channel dimension must be preserved")
        kernel, stride, padding = infer_pool2d_params(input_shape, output_shape)
        add_assert(kernel is not None, "max_pool2d parameters must be inferable or adaptive")
        if kernel is not None and stride is not None and padding is not None:
            notes.append(f"kernel={kernel}")
            notes.append(f"stride={stride}")
            notes.append(f"padding={padding}")
        return assertions, notes

    raise ValueError(f"Unsupported category {case.category}")


def infer_broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...] | None:
    lhs_rev = list(reversed(lhs))
    rhs_rev = list(reversed(rhs))
    out: list[int] = []
    for index in range(max(len(lhs_rev), len(rhs_rev))):
        left = lhs_rev[index] if index < len(lhs_rev) else 1
        right = rhs_rev[index] if index < len(rhs_rev) else 1
        if left != right and left != 1 and right != 1:
            return None
        out.append(max(left, right))
    return tuple(reversed(out))


def infer_matmul_output_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...] | None:
    batch_shape = infer_broadcast_shape(lhs[:-2], rhs[:-2])
    if batch_shape is None:
        return None
    return batch_shape + (lhs[-2], rhs[-1])


def infer_permutation(input_dims: tuple[str, ...], output_dims: tuple[str, ...]) -> tuple[int, ...] | None:
    if len(input_dims) != len(output_dims):
        return None

    used = [False] * len(input_dims)
    result: list[int] = []

    def backtrack(index: int) -> bool:
        if index == len(output_dims):
            return True
        target = output_dims[index]
        for candidate, token in enumerate(input_dims):
            if used[candidate] or token != target:
                continue
            used[candidate] = True
            result.append(candidate)
            if backtrack(index + 1):
                return True
            result.pop()
            used[candidate] = False
        return False

    if not backtrack(0):
        return None
    return tuple(result)


def infer_concat_axis(input_shapes: list[tuple[int, ...]], output_shape: tuple[int, ...]) -> int | None:
    rank = len(output_shape)
    for axis in range(rank):
        if any(shape[:axis] + shape[axis + 1 :] != input_shapes[0][:axis] + input_shapes[0][axis + 1 :] for shape in input_shapes[1:]):
            continue
        if any(shape[index] != output_shape[index] for shape in input_shapes for index in range(rank) if index != axis):
            continue
        if sum(shape[axis] for shape in input_shapes) == output_shape[axis]:
            return axis
    return None


def infer_batch_norm_axis(input_shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> int | None:
    if len(scale_shape) != 1:
        return None
    for axis in range(1, len(input_shape)):
        if input_shape[axis] == scale_shape[0]:
            return axis
    return None


def infer_reduction(input_dims: tuple[str, ...], output_dims: tuple[str, ...]) -> tuple[tuple[int, ...], bool] | None:
    if len(output_dims) == len(input_dims):
        axes = tuple(index for index, (source, target) in enumerate(zip(input_dims, output_dims)) if source != target and target == "1")
        if axes and apply_reduction_tokens(input_dims, axes, True) == output_dims:
            return axes, True
    axes: list[int] = []
    output_index = 0
    for input_index, token in enumerate(input_dims):
        if output_index < len(output_dims) and output_dims[output_index] == token:
            output_index += 1
            continue
        axes.append(input_index)
    if output_index == len(output_dims) and apply_reduction_tokens(input_dims, tuple(axes), False) == output_dims:
        return tuple(axes), False
    return None


def apply_reduction_tokens(input_dims: tuple[str, ...], axes: tuple[int, ...], keepdim: bool) -> tuple[str, ...]:
    if keepdim:
        return tuple("1" if index in axes else token for index, token in enumerate(input_dims))
    return tuple(token for index, token in enumerate(input_dims) if index not in axes)


def apply_reduction_shape(input_shape: tuple[int, ...], axes: tuple[int, ...], keepdim: bool) -> tuple[int, ...]:
    if keepdim:
        return tuple(1 if index in axes else value for index, value in enumerate(input_shape))
    return tuple(value for index, value in enumerate(input_shape) if index not in axes)


def conv2d_output_shape(
    input_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    stride: int,
    padding: int,
    dilation: int,
) -> tuple[int, int, int, int]:
    batch, _, height, width = input_shape
    out_channels, _, kernel_h, kernel_w = weight_shape
    output_h = ((height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride) + 1
    output_w = ((width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride) + 1
    return batch, out_channels, output_h, output_w


def infer_conv2d_params(
    case: CaseSpec,
    input_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
) -> tuple[int | None, int | None, int | None]:
    stride_token, padding_token, dilation_token = case.params
    if stride_token.isdigit() and padding_token.isdigit() and dilation_token.isdigit():
        return int(stride_token), int(padding_token), int(dilation_token)
    for stride in range(1, 17):
        for padding in range(0, 9):
            for dilation in range(1, 5):
                if conv2d_output_shape(input_shape, weight_shape, stride, padding, dilation) == output_shape:
                    return stride, padding, dilation
    return None, None, None


def infer_pool2d_params(
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
) -> tuple[int | None, int | None, int | None]:
    _, _, input_h, input_w = input_shape
    _, _, output_h, output_w = output_shape
    for kernel in range(2, 17):
        for stride in range(1, 17):
            for padding in range(0, 9):
                computed_h = ((input_h + 2 * padding - kernel) // stride) + 1
                computed_w = ((input_w + 2 * padding - kernel) // stride) + 1
                if computed_h == output_h and computed_w == output_w:
                    return kernel, stride, padding
    return None, None, None


def run_case(case_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    metadata = describe_case(case_path)
    case = parse_case(case_path)
    if not metadata["valid_contract"]:
        return summary_dict(metadata, "n/a", 0.0, ["Invalid contract"])
    env = env_for_case(case)
    _, notes = validate_case(case, env)

    if triton is None:
        return summary_dict(metadata, "n/a", 0.0, notes + ["Missing triton package"])
    if torch is None or torch_f is None:
        return summary_dict(metadata, "n/a", 0.0, notes + ["Missing torch package"])
    if device == "cuda" and not torch.cuda.is_available():
        return summary_dict(metadata, "n/a", 0.0, notes + ["CUDA not available"])

    input_shapes = [evaluate_shape(value, env) for value in case.inputs if not value.is_scalar]
    output_shape = evaluate_shape(case.output, env)
    total_elements = sum(product(shape) for shape in input_shapes) + product(output_shape)
    if total_elements > MAX_EXECUTION_ELEMENTS:
        return summary_dict(metadata, "n/a", 0.0, notes + [f"Execution budget exceeded {total_elements} elements"])

    torch_device = torch.device(device)
    tensors = materialize_inputs(case, input_shapes, torch_device)

    start = time.perf_counter_ns()
    result = execute_case(case, tensors, output_shape)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0

    if tuple(result.shape) != output_shape:
        raise ValueError(f"{case.case_name}: expected {output_shape} but saw {tuple(result.shape)}")

    if case.category in {"reshape", "transpose"}:
        notes.append("view-op")

    return summary_dict(metadata, "success", elapsed_ms, notes)


def summary_dict(metadata: dict[str, Any], status: str, time_ms: float, notes: list[str]) -> dict[str, Any]:
    merged_notes = [note for note in [metadata.get("notes", "")] + notes if note]
    return {
        "status": status,
        "mode": metadata["mode"],
        "explicit_assertions": metadata["explicit_assertions"],
        "input_shape": metadata["input_shape"],
        "output_shape": metadata["output_shape"],
        "time_ms": round(time_ms, 4),
        "notes": "; ".join(dict.fromkeys(merged_notes)),
    }


def materialize_inputs(case: CaseSpec, input_shapes: list[tuple[int, ...]], device: torch.device) -> list[Any]:
    tensors: list[Any] = []
    tensor_index = 0
    for index, value in enumerate(case.inputs):
        if value.is_scalar:
            tensors.append(float(value.token))
            continue

        shape = input_shapes[tensor_index]
        tensor_index += 1
        if case.category == "embedding" and index == 0:
            vocab_size = input_shapes[1][0]
            tensors.append(torch.randint(0, vocab_size, shape, device=device, dtype=torch.int64))
            continue

        tensors.append(torch.randn(shape, device=device, dtype=torch.float32))
    return tensors


def execute_case(case: CaseSpec, tensors: list[Any], output_shape: tuple[int, ...]) -> torch.Tensor:
    if case.category == "gelu":
        return torch_f.gelu(tensors[0])
    if case.category == "relu":
        return torch.relu(tensors[0])
    if case.category == "sigmoid":
        return torch.sigmoid(tensors[0])
    if case.category == "softmax":
        return torch.softmax(tensors[0], dim=-1)
    if case.category == "elemwise_add":
        return tensors[0] + tensors[1]
    if case.category == "matmul":
        return torch.matmul(tensors[0], tensors[1])
    if case.category == "reshape":
        return tensors[0].reshape(output_shape)
    if case.category == "transpose":
        permutation = infer_permutation(case.inputs[0].dims, case.output.dims)
        if permutation is None:
            raise ValueError(f"{case.case_name}: could not infer transpose permutation")
        return tensors[0].permute(permutation)
    if case.category == "concat":
        axis = infer_concat_axis([tuple(tensor.shape) for tensor in tensors], output_shape)
        if axis is None:
            raise ValueError(f"{case.case_name}: could not infer concat axis")
        return torch.cat(tensors, dim=axis)
    if case.category == "reduce_mean":
        reduction = infer_reduction(case.inputs[0].dims, case.output.dims)
        if reduction is None:
            raise ValueError(f"{case.case_name}: could not infer reduction")
        axes, keepdim = reduction
        return tensors[0].mean(dim=axes, keepdim=keepdim)
    if case.category == "embedding":
        return torch_f.embedding(tensors[0], tensors[1])
    if case.category == "batch_norm":
        input_tensor, scale, bias = tensors
        stats_dims = (0,)
        mean = input_tensor.mean(dim=stats_dims, keepdim=True)
        var = input_tensor.var(dim=stats_dims, unbiased=False, keepdim=True)
        shape = [1] * input_tensor.ndim
        axis = infer_batch_norm_axis(tuple(input_tensor.shape), tuple(scale.shape))
        if axis is None:
            raise ValueError(f"{case.case_name}: could not infer batch norm parameter axis")
        shape[axis] = scale.shape[0]
        scale = scale.reshape(shape)
        bias = bias.reshape(shape)
        return ((input_tensor - mean) / torch.sqrt(var + 1e-5)) * scale + bias
    if case.category == "layer_normalization":
        input_tensor, scale, bias = tensors
        return torch_f.layer_norm(input_tensor, scale.shape, scale, bias)
    if case.category == "conv2d":
        input_tensor, weight = tensors
        stride, padding, dilation = infer_conv2d_params(case, tuple(input_tensor.shape), tuple(weight.shape), output_shape)
        if stride is None or padding is None or dilation is None:
            raise ValueError(f"{case.case_name}: could not infer conv2d parameters")
        return torch_f.conv2d(input_tensor, weight, stride=stride, padding=padding, dilation=dilation)
    if case.category == "max_pool2d":
        input_tensor = tensors[0]
        kernel, stride, padding = infer_pool2d_params(tuple(input_tensor.shape), output_shape)
        if kernel is None or stride is None or padding is None:
            return torch_f.adaptive_max_pool2d(input_tensor, output_shape[-2:])
        return torch_f.max_pool2d(input_tensor, kernel_size=kernel, stride=stride, padding=padding)

    raise ValueError(f"Unsupported execution category {case.category}")