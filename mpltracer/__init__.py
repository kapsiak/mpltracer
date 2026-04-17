from __future__ import annotations

from pathlib import Path
from typing import Any

from .codegen import generateScript as _generateScript
from .ir import TraceIR
from .patches import applyPatches, removePatches
from .proxy import Proxy
from .serialize import DEFAULT_ARRAY_INLINE_THRESHOLD

__all__ = [
    "Trace",
    "generateScript",
    "TraceIR",
    "Proxy",
]



def generateScript(
    trace: TraceIR | "Trace",
    path: str | Path | None = None,
    *,
    array_threshold: int = DEFAULT_ARRAY_INLINE_THRESHOLD,
) -> str:
    if isinstance(trace, Trace):
        trace_ir = trace.ir
    else:
        trace_ir = trace

    return _generateScript(
        trace_ir,
        path=path,
        array_threshold=array_threshold,
    )


class Trace:
    def __init__(self, *, capture_rc_params: bool = True) -> None:
        self._capture_rc_params = capture_rc_params
        self.ir: TraceIR = TraceIR()

    def __enter__(self) -> "Trace":
        self.ir = TraceIR()
        applyPatches(self.ir, capture_rc_params=self._capture_rc_params)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        removePatches()

    def generateScript(self, *args,**kwargs) -> str:
        return _generateScript(self.ir, *args, **kwargs)
