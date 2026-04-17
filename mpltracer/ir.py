from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class ImportNode:
    module: str
    alias: str | None = None

    def toCode(self) -> str:
        if self.alias:
            return f"import {self.module} as {self.alias}"
        return f"import {self.module}"


@dataclasses.dataclass
class CallNode:
    target: str
    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    return_var: str | None = None
    return_unpack_count: int = 0


@dataclasses.dataclass
class TraceIR:
    imports: set[ImportNode] = dataclasses.field(default_factory=set)
    calls: list[CallNode] = dataclasses.field(default_factory=list)
    data_arrays: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    _var_counter: int = dataclasses.field(default=0, repr=False)
    _array_counter: int = dataclasses.field(default=0, repr=False)

    def nextVarName(self, prefix: str = "var") -> str:
        name = f"{prefix}_{self._var_counter}"
        self._var_counter += 1
        return name

    def nextArrayName(self) -> str:
        name = f"data_{self._array_counter}"
        self._array_counter += 1
        return name

    def addImport(self, module: str, alias: str | None = None) -> None:
        self.imports.add(ImportNode(module=module, alias=alias))

    def addCall(
        self,
        target: str,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        return_var: str | None = None,
        return_unpack_count: int = 0,
    ) -> CallNode:
        node = CallNode(
            target=target,
            method=method,
            args=args,
            kwargs=kwargs or {},
            return_var=return_var,
            return_unpack_count=return_unpack_count,
        )
        self.calls.append(node)
        return node
