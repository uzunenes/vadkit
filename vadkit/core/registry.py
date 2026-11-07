"""Name -> class registry, so YAML configs pick back-ends:  {"kind": "flow", "name": "neuflow", "args": {...}}."""
from __future__ import annotations

_REG: dict[tuple[str, str], type] = {}


def register(kind: str, name: str):
    def deco(cls):
        key = (kind, name)
        if key in _REG and _REG[key] is not cls:
            raise KeyError(f"duplicate registration {key}")
        _REG[key] = cls
        return cls
    return deco


def build(kind: str, spec: dict | str):
    """spec: "name" or {"name": ..., "args": {...}}. Imports vadkit modules lazily so heavy back-ends load on demand."""
    import vadkit.plugins  # noqa: F401  (registers every built-in back-end)
    if isinstance(spec, str):
        spec = {"name": spec}
    key = (kind, spec["name"])
    if key not in _REG:
        raise KeyError(f"unknown {kind} '{spec['name']}'; known: {sorted(n for k, n in _REG if k == kind)}")
    return _REG[key](**spec.get("args", {}))


def available(kind: str) -> list[str]:
    import vadkit.plugins  # noqa: F401
    return sorted(n for k, n in _REG if k == kind)
