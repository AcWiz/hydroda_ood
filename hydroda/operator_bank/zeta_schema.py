"""Schema helpers for packing/unpacking lightweight zeta parameters."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import torch


@dataclass
class ZetaBlock:
    name: str
    shape: List[int]
    start: int
    end: int


class ZetaPacker:
    def __init__(self, schema: List[ZetaBlock]) -> None:
        self.schema = schema

    @classmethod
    def from_zeta(cls, zeta: Dict[str, torch.Tensor]) -> "ZetaPacker":
        schema: List[ZetaBlock] = []
        offset = 0
        for name, tensor in zeta.items():
            n = int(tensor.numel())
            schema.append(ZetaBlock(name=name, shape=list(tensor.shape), start=offset, end=offset + n))
            offset += n
        return cls(schema)

    def pack(self, zeta: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for block in self.schema:
            if block.name not in zeta:
                raise KeyError(f"Missing zeta block {block.name!r}")
            parts.append(zeta[block.name].reshape(-1))
        return torch.cat(parts, dim=0)

    def unpack(self, vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for block in self.schema:
            out[block.name] = vector[block.start : block.end].reshape(block.shape)
        return out

    def to_jsonable(self) -> List[dict]:
        return [asdict(b) for b in self.schema]
