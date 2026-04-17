from __future__ import annotations

import enum
import pathlib
from typing import TYPE_CHECKING, Any
from .proxy import Proxy

import numpy as np

if TYPE_CHECKING:
    from .ir import TraceIR

DEFAULT_ARRAY_INLINE_THRESHOLD: int = 500

def serializeValue(
    value: Any,
    trace_ir: TraceIR | None = None,
    *,
    array_threshold: int = DEFAULT_ARRAY_INLINE_THRESHOLD,
) -> str:

    if isinstance(value, Proxy):
        return object.__getattribute__(value, "_var_name")

    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)

    if isinstance(value, np.integer):
        return repr(int(value))
    if isinstance(value, np.floating):
        return repr(float(value))
    if isinstance(value, np.bool_):
        return repr(bool(value))

    if isinstance(value, np.ndarray):
        return _serializeArray(value, trace_ir, array_threshold=array_threshold)

    if isinstance(value, pathlib.PurePath):
        return _serializePath(value)

    if _isHistogramLike(value):
        return _serializeHistogram(value, trace_ir, array_threshold=array_threshold)

    if isinstance(value, tuple):
        inner = ", ".join(
            serializeValue(v, trace_ir, array_threshold=array_threshold) for v in value
        )
        if len(value) == 1:
            return f"({inner},)"
        return f"({inner})"

    if isinstance(value, list):
        inner = ", ".join(
            serializeValue(v, trace_ir, array_threshold=array_threshold) for v in value
        )
        return f"[{inner}]"

    if isinstance(value, dict):
        pairs = ", ".join(
            f"{serializeValue(k, trace_ir, array_threshold=array_threshold)}: "
            f"{serializeValue(v, trace_ir, array_threshold=array_threshold)}"
            for k, v in value.items()
        )
        return "{" + pairs + "}"

    if isinstance(value, set):
        inner = ", ".join(
            serializeValue(v, trace_ir, array_threshold=array_threshold)
            for v in sorted(value, key=repr)
        )
        return "{" + inner + "}"

    if isinstance(value, slice):
        parts = [repr(value.start), repr(value.stop)]
        if value.step is not None:
            parts.append(repr(value.step))
        return f"slice({', '.join(parts)})"

    if isinstance(value, enum.Enum):
        return f"{type(value).__module__}.{type(value).__qualname__}.{value.name}"

    return repr(value)



def _serializeArray(
    arr: np.ndarray,
    trace_ir: TraceIR | None,
    *,
    array_threshold: int,
) -> str:
    if arr.size <= array_threshold:
        return _inlineArray(arr)

    if trace_ir is not None:
        name = trace_ir.nextArrayName()
        trace_ir.data_arrays[name] = arr
        return f'np.load("{name}.npy")'

    return _inlineArray(arr)


def _inlineArray(arr: np.ndarray) -> str:
    list_repr = repr(arr.tolist())
    dtype_str = _dtypeString(arr.dtype)
    if dtype_str:
        return f"np.array({list_repr}, dtype={dtype_str})"
    return f"np.array({list_repr})"


def _dtypeString(dtype: np.dtype) -> str:
    if dtype == np.float64 or dtype == np.int64:
        return ""
    return f"np.{dtype}"


def _serializePath(p: pathlib.PurePath) -> str:
    cls_name = type(p).__name__
    return f'Path({repr(str(p))})'


def _isHistogramLike(value: Any) -> bool:
    return (
        hasattr(value, "to_numpy")
        and hasattr(value, "axes")
        and hasattr(value, "values")
        and callable(getattr(value, "to_numpy", None))
    )


def _serializeHistogram(
    h: Any,
    trace_ir: TraceIR | None,
    *,
    array_threshold: int,
) -> str:
    numpy_tuple = h.to_numpy()
    values = numpy_tuple[0]
    edges_list = list(numpy_tuple[1:])

    values_str = _serializeArray(values, trace_ir, array_threshold=array_threshold)
    edges_strs = [
        _serializeArray(e, trace_ir, array_threshold=array_threshold)
        for e in edges_list
    ]

    var_str = "None"
    if hasattr(h, "variances") and callable(getattr(h, "variances", None)):
        var_array = h.variances()
        if var_array is not None:
            var_str = _serializeArray(var_array, trace_ir, array_threshold=array_threshold)

    if var_str != "None":
        return f"uhi.numpy_plottable.NumPyPlottableHistogram({values_str}, {', '.join(edges_strs)}, variances={var_str})"

    all_parts = [values_str] + edges_strs
    return f"({', '.join(all_parts)})"
