from __future__ import annotations

import enum
import inspect
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


#def _serializeCallable(fn: Any) -> str:
#    module = getattr(fn, "__module__", None)
#    qualname = getattr(fn, "__qualname__", None)
#    if module and qualname and "<lambda>" not in (qualname or ""):
#        return f"{module}.{qualname}"
#
#    try:
#        source = inspect.getsource(fn).strip()
#        return source
#    except (OSError, TypeError):
#        pass
#
#    return f"# <unserializable callable: {fn!r}>"
