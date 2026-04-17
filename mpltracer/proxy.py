from __future__ import annotations

import numpy as np
from typing import Any, Iterator

from .ir import TraceIR


_TRACED_TYPES: tuple[type, ...] | None = None

def _getTracedTypes() -> tuple[type, ...]:
    global _TRACED_TYPES

    if _TRACED_TYPES is None:
        import matplotlib.axes  # noqa: PLC0415
        import matplotlib.figure  # noqa: PLC0415
        import matplotlib.text  # noqa: PLC0415
        import matplotlib.lines  # noqa: PLC0415
        import matplotlib.image  # noqa: PLC0415
        import matplotlib.patches  # noqa: PLC0415
        import matplotlib.collections  # noqa: PLC0415
        import matplotlib.container  # noqa: PLC0415
        import matplotlib.legend  # noqa: PLC0415
        import matplotlib.colorbar  # noqa: PLC0415

        _TRACED_TYPES = (
            matplotlib.axes.Axes,
            matplotlib.figure.Figure,
            matplotlib.text.Text,
            matplotlib.lines.Line2D,
            matplotlib.image.AxesImage,
            matplotlib.patches.Patch,
            matplotlib.collections.Collection,
            matplotlib.container.Container,
            matplotlib.legend.Legend,
            matplotlib.colorbar.Colorbar,
        )
    return _TRACED_TYPES


_SKIP_METHODS: frozenset[str] = frozenset({
    "draw", "draw_artist", "get_renderer", "get_window_extent",
    "get_tightbbox", "get_children", "findobj", "pchanged",
    "stale", "set_figure", "set_transform",
    "__repr__", "__str__", "__len__", "__bool__", "__hash__",
    "__eq__", "__ne__", "__format__",
})


def unwrapValue(value: Any) -> Any:
    if isinstance(value, Proxy):
        return object.__getattribute__(value, "_obj")
    if isinstance(value, list):
        return [unwrapValue(v) for v in value]
    if isinstance(value, tuple):
        return tuple(unwrapValue(v) for v in value)
    if isinstance(value, dict):
        return {unwrapValue(k): unwrapValue(v) for k, v in value.items()}
    return value


class Proxy:
    __slots__ = ("_obj", "_trace_ir", "_var_name")

    def __init__(self, obj: Any, trace_ir: TraceIR, var_name: str) -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_trace_ir", trace_ir)
        object.__setattr__(self, "_var_name", var_name)

    def __getattr__(self, name: str) -> Any:
        obj = object.__getattribute__(self, "_obj")
        trace_ir = object.__getattribute__(self, "_trace_ir")
        var_name = object.__getattribute__(self, "_var_name")

        attr = getattr(obj, name)

        if callable(attr) and name not in _SKIP_METHODS and not name.startswith("_"):
            return _MethodRecorder(attr, trace_ir, var_name, name)

        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        obj = object.__getattribute__(self, "_obj")
        trace_ir = object.__getattribute__(self, "_trace_ir")
        var_name = object.__getattribute__(self, "_var_name")

        real_value = unwrapValue(value)
        setattr(obj, name, real_value)

        setter = f"set_{name}"
        if hasattr(obj, setter):
            trace_ir.addCall(
                target=var_name,
                method=setter,
                args=(value,),
            )

    def __getitem__(self, key: Any) -> Any:
        obj = object.__getattribute__(self, "_obj")
        trace_ir = object.__getattribute__(self, "_trace_ir")
        var_name = object.__getattribute__(self, "_var_name")

        result = obj[key]
        child_name = _indexedVarName(var_name, key)

        if _shouldWrap(result):
            return Proxy(result, trace_ir, child_name)
        return result

    def __iter__(self) -> Iterator[Any]:
        obj = object.__getattribute__(self, "_obj")
        trace_ir = object.__getattribute__(self, "_trace_ir")
        var_name = object.__getattribute__(self, "_var_name")

        for idx, item in enumerate(obj):
            child_name = f"{var_name}_{idx}"
            if _shouldWrap(item):
                yield Proxy(item, trace_ir, child_name)
            else:
                yield item

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_obj"))


    def __bool__(self) -> bool:
        return bool(object.__getattribute__(self, "_obj"))

    def __repr__(self) -> str:
        obj = object.__getattribute__(self, "_obj")
        var_name = object.__getattribute__(self, "_var_name")
        return f"Proxy({var_name!r}, {obj!r})"

    def __str__(self) -> str:
        return str(object.__getattribute__(self, "_obj"))

    def __eq__(self, other: Any) -> bool:
        obj = object.__getattribute__(self, "_obj")
        other_obj = unwrapValue(other)
        return obj == other_obj

    def __hash__(self) -> int:
        return hash(object.__getattribute__(self, "_obj"))


    def __array__(self, dtype: Any = None) -> np.ndarray:
        obj = object.__getattribute__(self, "_obj")
        return np.asarray(obj, dtype=dtype)


class _MethodRecorder:
    __slots__ = ("_real_method", "_trace_ir", "_target_var", "_method_name")

    def __init__(
        self,
        real_method: Any,
        trace_ir: TraceIR,
        target_var: str,
        method_name: str,
    ) -> None:
        self._real_method = real_method
        self._trace_ir = trace_ir
        self._target_var = target_var
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from .patches import _getCallDepth, _incCallDepth, _decCallDepth  # noqa: PLC0415

        is_top_level = _getCallDepth() == 0
        _incCallDepth()
        try:
            real_args = tuple(unwrapValue(a) for a in args)
            real_kwargs = {k: unwrapValue(v) for k, v in kwargs.items()}

            result = self._real_method(*real_args, **real_kwargs)

            if not is_top_level:
                return result

            self._trace_ir.addCall(
                target=self._target_var,
                method=self._method_name,
                args=args,
                kwargs=kwargs,
            )

            if _shouldWrap(result):
                ret_name = self._trace_ir.nextVarName(self._method_name)
                return Proxy(result, self._trace_ir, ret_name)

            return result
        finally:
            _decCallDepth()


def _shouldWrap(obj: Any) -> bool:
    if isinstance(obj, Proxy):
        return False
    return isinstance(obj, _getTracedTypes())


def _indexedVarName(base: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{base}_{key}"
    if isinstance(key, tuple):
        suffix = "_".join(str(k) for k in key)
        return f"{base}_{suffix}"
    return f"{base}_item"
