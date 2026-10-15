"""Artifact persistence and per-stage input-hash checkpoints.

Models are saved as .json or .yaml (by suffix), atomically (tmp + os.replace).
Checkpoints store {input_hash, data} under data_dir/checkpoints/<stage>/ so a
stage is re-run only when its inputs changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel, TypeAdapter
from .config import Workspace


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_model(path: Path, obj: BaseModel | list[BaseModel]) -> None:
    """Save one model or a list of models as .json or .yaml (atomic)."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = (
            obj.model_dump(mode="json")
            if isinstance(obj, BaseModel)
            else [o.model_dump(mode="json") for o in obj]
        )
        raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    elif suffix in {".yaml", ".yml"}:
        data = (
            obj.model_dump(mode="json")
            if isinstance(obj, BaseModel)
            else [o.model_dump(mode="json") for o in obj]
        )
        raw = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf-8")
    else:
        raise ValueError(f"unsupported model file suffix: {path.suffix!r} (use .json/.yaml)")
    _atomic_write(path, raw)


def _read_structured(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported model file suffix: {path.suffix!r} (use .json/.yaml)")


def load_model[T: BaseModel](path: Path, cls: type[T]) -> T:
    return cls.model_validate(_read_structured(path))


def load_models[T: BaseModel](path: Path, cls: type[T]) -> list[T]:
    return TypeAdapter(list[cls]).validate_python(_read_structured(path))


def input_hash(*parts: str | bytes | BaseModel) -> str:
    """sha256 hex[:16] over each part's canonical bytes.

    str is encoded utf-8; BaseModel is serialized as canonical JSON
    (model_dump_json, deterministic key order is not guaranteed by pydantic —
    sort via json round-trip); bytes used as-is. Each part is length- and
    type-tagged to keep concatenations unambiguous.
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, BaseModel):
            # canonical: re-dump parsed JSON with sorted keys
            payload = json.dumps(
                json.loads(part.model_dump_json()),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            tag = b"m"
        elif isinstance(part, str):
            payload = part.encode("utf-8")
            tag = b"s"
        elif isinstance(part, bytes):
            payload = part
            tag = b"b"
        else:
            raise TypeError(f"input_hash part must be str|bytes|BaseModel, got {type(part).__name__}")
        h.update(tag)
        h.update(len(payload).to_bytes(8, "big"))
        h.update(payload)
    return h.hexdigest()[:16]


class _CheckpointEntry(BaseModel):
    input_hash: str
    data: dict


class Checkpoints:
    """Per-stage checkpoint store; a hit requires the stored input hash to match."""

    def __init__(self, ws: Workspace, stage: str):
        self.dir = ws.data_dir / "checkpoints" / stage
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _file_name(key: str) -> str:
        # keys may contain characters unsafe for filenames; keep a readable prefix
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        return f"{safe}.json"

    def _file(self, key: str) -> Path:
        return self.dir / self._file_name(key)

    def get[T: BaseModel](self, key: str, input_hash: str, cls: type[T]) -> T | None:
        f = self._file(key)
        if not f.exists():
            return None
        try:
            entry = _CheckpointEntry.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            return None  # corrupt checkpoint: treat as miss
        if entry.input_hash != input_hash:
            return None
        return cls.model_validate(entry.data)

    def put(self, key: str, input_hash: str, obj: BaseModel) -> None:
        entry = _CheckpointEntry(input_hash=input_hash, data=obj.model_dump(mode="json"))
        _atomic_write(self._file(key), entry.model_dump_json().encode("utf-8"))

    def invalidate(self, key: str) -> None:
        try:
            self._file(key).unlink()
        except FileNotFoundError:
            pass
