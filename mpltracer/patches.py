from __future__ import annotations

import functools
import threading
from typing import Any
import matplotlib.pyplot as plt  
import matplotlib.figure 
import matplotlib.axes
import matplotlib

import mplhep 
import numpy as np

from .ir import TraceIR
from .proxy import Proxy, unwrapValue, _shouldWrap


_originals: dict[tuple[Any, str], Any] = {}
_patched: bool = False
_call_depth: threading.local = threading.local()


def _getCallDepth() -> int:
    return getattr(_call_depth, "depth", 0)


def _incCallDepth() -> None:
    _call_depth.depth = _getCallDepth() + 1


def _decCallDepth() -> None:
    _call_depth.depth = max(0, _getCallDepth() - 1)


_PLT_FUNCTIONS: list[str] = [
    "subplots", "figure", "show", "savefig", "close",
    "tight_layout", "suptitle",
    "gca", "gcf", "sca",
    "xlim", "ylim", "xlabel", "ylabel", "title",
    "legend", "colorbar", "grid",
    "subplot", "subplot2grid",
    "axes", "twinx", "twiny",
    "text", "annotate",
    "plot", "scatter", "bar", "barh", "hist", "hist2d",
    "errorbar", "fill_between", "fill_betweenx",
    "imshow", "contour", "contourf", "pcolormesh",
    "axhline", "axvline", "axhspan", "axvspan",
    "xticks", "yticks", "tick_params",
    "xscale", "yscale",
    "clim", "clabel",
    "subplots_adjust", "margins",
]

_MPLHEP_FUNCTIONS: list[str] = [
    "histplot", "hist", "hist2dplot",
    "set_style",
    "add_text", "append_text",
]

_MPLHEP_EXPERIMENT_FUNCTIONS: list[str] = [
    "label", "text",
]

_MPLHEP_EXPERIMENTS: list[str] = [
    "cms", "atlas", "lhcb", "alice", "dune",
]

_INTERNAL_RC_KEYS: frozenset[str] = frozenset({
})

def applyPatches(trace_ir: TraceIR, *, capture_rc_params: bool = True) -> None:
    global _patched
    if _patched:
        return
    _patched = True


    for name in _PLT_FUNCTIONS:
        original = getattr(plt, name, None)
        if original is None or not callable(original):
            continue
        _originals[(plt, name)] = original
        wrapper = _makePltWrapper(original, name, trace_ir)
        setattr(plt, name, wrapper)

    if capture_rc_params:
        _patchRcParams(trace_ir)

    try:

        for name in _MPLHEP_FUNCTIONS:
            original = getattr(mplhep, name, None)
            if original is None or not callable(original):
                continue
            _originals[(mplhep, name)] = original
            wrapper = _makeMplhepWrapper(original, name, "mplhep", trace_ir)
            setattr(mplhep, name, wrapper)

        for exp_name in _MPLHEP_EXPERIMENTS:
            exp_module = getattr(mplhep, exp_name, None)
            if exp_module is None:
                continue
            for fn_name in _MPLHEP_EXPERIMENT_FUNCTIONS:
                original = getattr(exp_module, fn_name, None)
                if original is None or not callable(original):
                    continue
                _originals[(exp_module, fn_name)] = original
                qual = f"mplhep.{exp_name}"
                wrapper = _makeMplhepWrapper(original, fn_name, qual, trace_ir)
                setattr(exp_module, fn_name, wrapper)

    except ImportError:
        pass  


def removePatches() -> None:
    global _patched

    _rc_key = ("__rcParams__", "__setitem__")
    if _rc_key in _originals:
        import matplotlib  
        if hasattr(matplotlib.rcParams, "__setitem__"):
            try:
                delattr(matplotlib.rcParams, "__setitem__")
            except AttributeError:
                pass
        del _originals[_rc_key]

    for (module, name), original in _originals.items():
        setattr(module, name, original)
    _originals.clear()
    _patched = False


def _makePltWrapper(original: Any, name: str, trace_ir: TraceIR) -> Any:

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        is_top_level = _getCallDepth() == 0
        _incCallDepth()
        try:
            real_args = tuple(unwrapValue(a) for a in args)
            real_kwargs = {k: unwrapValue(v) for k, v in kwargs.items()}
            result = original(*real_args, **real_kwargs)
            if not is_top_level:
                return result
            return_var, return_count, wrapped = _wrapPltResult(
                name, result, trace_ir
            )

            trace_ir.addCall(
                target="plt",
                method=name,
                args=args,
                kwargs=kwargs,
                return_var=return_var,
                return_unpack_count=return_count,
            )
            trace_ir.addImport("matplotlib.pyplot", "plt")

            return wrapped if wrapped is not None else result
        finally:
            _decCallDepth()

    return wrapper


def _makeMplhepWrapper(
    original: Any, name: str, module_path: str, trace_ir: TraceIR
) -> Any:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        is_top_level = _getCallDepth() == 0
        _incCallDepth()
        try:
            real_args = tuple(unwrapValue(a) for a in args)
            real_kwargs = {k: unwrapValue(v) for k, v in kwargs.items()}

            result = original(*real_args, **real_kwargs)

            if not is_top_level:
                return result

            trace_ir.addCall(
                target=module_path,
                method=name,
                args=args,
                kwargs=kwargs,
            )

            trace_ir.addImport("mplhep")

            return result
        finally:
            _decCallDepth()

    return wrapper

def _wrapPltResult(
    fn_name: str,
    result: Any,
    trace_ir: TraceIR,
) -> tuple[str | None, int, Any]:

    if fn_name == "subplots":
        fig, axes = result
        fig_proxy = Proxy(fig, trace_ir, "fig")
        axes_proxy = _wrapAxes(axes, trace_ir)
        return "fig, ax", 2, (fig_proxy, axes_proxy)

    if fn_name == "figure":
        if isinstance(result, matplotlib.figure.Figure):
            proxy = Proxy(result, trace_ir, "fig")
            return "fig", 1, proxy
        return None, 0, None

    if fn_name in ("gca", "subplot", "subplot2grid", "axes", "twinx", "twiny"):
        if _shouldWrap(result):
            proxy = Proxy(result, trace_ir, "ax")
            return "ax", 1, proxy
        return None, 0, None

    if fn_name == "gcf":
        if _shouldWrap(result):
            proxy = Proxy(result, trace_ir, "fig")
            return "fig", 1, proxy
        return None, 0, None

    if fn_name == "colorbar" and _shouldWrap(result):
        proxy = Proxy(result, trace_ir, "cbar")
        return "cbar", 1, proxy

    return None, 0, None


def _wrapAxes(axes: Any, trace_ir: TraceIR) -> Any:

    if isinstance(axes, matplotlib.axes.Axes):
        return Proxy(axes, trace_ir, "ax")

    if isinstance(axes, np.ndarray):
        shape = axes.shape
        proxied = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            if len(idx) == 1:
                child_name = f"ax_{idx[0]}"
            else:
                child_name = "ax_" + "_".join(str(i) for i in idx)
            proxied[idx] = Proxy(axes[idx], trace_ir, child_name)
        return proxied

    return axes





def _patchRcParams(trace_ir: TraceIR) -> None:
    rc = matplotlib.rcParams
    original_setitem = type(rc).__setitem__

    def _recording_setitem(self: Any, key: str, value: Any) -> None:
        original_setitem(self, key, value)
        if _getCallDepth() == 0 and key not in _INTERNAL_RC_KEYS:
            trace_ir.addCall(
                target="plt.rcParams",
                method="__setitem__",
                args=(key, value),
            )

    _originals[("__rcParams__", "__setitem__")] = original_setitem
    import types  

    rc.__setitem__ = types.MethodType(_recording_setitem, rc)  
