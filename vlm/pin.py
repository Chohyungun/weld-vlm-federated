"""착수 고정(`PIN.json`) — 조기 좌표 점검과 본실험 다섯 칸이 공유하는 출발 조건을 한 파일에 묶는다(11번 §4).

네 칸을 묶는다.

| 칸 | 무엇 |
|---|---|
| `code` | 코드 저장소의 HEAD 커밋 · 추적 `*.py` 와 `vlm/prompts/` 의 작업 트리 바이트가 커밋 blob 과 같다(export 의 청결 검사 `code_commit` 과 같은 규칙) |
| `env` | `uv.lock` 의 sha256 · 설치된 꾸러미 목록의 digest · torch · CUDA · bitsandbytes · transformers 판 · GPU 이름 · 드라이버 |
| `common` | 통합형 설정 블록(`configs/base.yaml` 의 `uni_*`) · 프롬프트 바이트 · 템플릿 모드 · 좌표 규약의 digest |
| `init_adapter` | 공통 초기 어댑터(캐시 파일)의 digest — `init_adapter_digest` 와 같은 식 |

- **쓰기**(`write_pin`)는 코드가 커밋과 다르면 거부하고, 있는 파일을 덮지 않는다.
- **대조**(`check_pin`)는 시작 · 재개마다 지금 값을 다시 내 맞댄다. 다른 칸을 모두 사유에 싣고 `PinMismatch`(`pin_mismatch`)로 멈춘다.
- 시험은 네 칸의 측정 함수를 인자로 바꾼다(파이썬 인자로만 — 이음새와 같다). 기본 측정은 실제 저장소 · 설치 · 장치를 읽는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PIN_SCHEMA", "PinMismatch", "build_pin", "check_pin", "code_probe", "common_probe", "env_probe",
           "init_probe", "pin_sha256", "read_pin", "write_pin"]

PIN_SCHEMA = 1
REPO_ROOT = Path(__file__).resolve().parents[1]


class PinMismatch(ValueError):
    """착수 고정과 지금 값이 다르다 — 시작 · 재개하지 않는다."""

    code = "pin_mismatch"

    def __init__(self, fields: list[str], message: str = ""):
        self.fields = list(fields)
        super().__init__(f"pin_mismatch: {', '.join(fields)}" + (f" — {message}" if message else ""))


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 네 칸의 측정
def code_probe(repo: Path | None = None) -> dict[str, Any]:
    """HEAD 커밋과 청결하지 않은 까닭(빈 목록이어야 쓴다)."""
    from vlm.export_preflight import code_commit

    commit, dirty = code_commit(Path(repo) if repo is not None else REPO_ROOT)
    return {"commit": commit, "dirty": list(dirty)}


def env_probe(repo: Path | None = None) -> dict[str, Any]:
    """설치 · 장치. GPU 가 보이지 않으면 GPU 칸은 None 이다(장치가 바뀌면 다른 고정이다)."""
    import importlib.metadata as md

    import torch

    root = Path(repo) if repo is not None else REPO_ROOT
    pkgs = sorted({f"{(d.metadata.get('Name') or '').lower()}=={d.version}" for d in md.distributions()})

    def ver(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    gpu = driver = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20, check=False)
            driver = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
        except (OSError, subprocess.SubprocessError):
            driver = None
    lock = root / "uv.lock"
    return {"uv_lock_sha256": _sha_file(lock) if lock.is_file() else None,
            "packages_sha256": _sha_bytes("\n".join(pkgs).encode("utf-8")), "n_packages": len(pkgs),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "bitsandbytes": ver("bitsandbytes"),
            "transformers": ver("transformers"), "gpu_name": gpu, "driver": driver}


def common_probe(config_path: Path | None = None, *, repo: Path | None = None) -> dict[str, Any]:
    """통합형 설정 블록 · 프롬프트 바이트 · 템플릿 모드 · 좌표 규약."""
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import template_mode_of
    from vlm.uni_config import load_uni_config

    root = Path(repo) if repo is not None else REPO_ROOT
    cfg = load_uni_config(config_path)
    block = {k: v for k, v in asdict(cfg).items() if k != "source"}
    prompt = None
    if cfg.prompt_path:
        p = Path(cfg.prompt_path)
        p = p if p.is_absolute() else root / p
        prompt = _sha_file(p) if p.is_file() else None
    return {"uni_block_sha256": _sha_bytes(_canonical(block)), "prompt_sha256": prompt,
            "template_mode": template_mode_of(cfg.chat_template_kwargs or {}),
            "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space=cfg.coord_space or "ABS_ORIG")),
            "processor_kwargs_sha256": _sha_bytes(_canonical(cfg.processor_kwargs)),
            "config_file_sha256": _sha_file(Path(config_path)) if config_path is not None else None}


def init_probe(init_adapter_path: Path) -> dict[str, Any]:
    """공통 초기 어댑터 캐시의 digest — 캐시에 적힌 키 순서로(`read_initial_cache` 와 같다)."""
    from vlm.init_adapter import init_adapter_digest

    p = Path(init_adapter_path)
    if not p.is_file():
        raise PinMismatch(["init_adapter"], f"공통 초기 어댑터가 없다: {p.name}")
    with np.load(p) as loaded:
        arrays = [loaded[k] for k in loaded.files]
    proof = p.with_suffix(".proof.json")
    return {"init_adapter_digest": init_adapter_digest(arrays),
            "proof_sha256": _sha_file(proof) if proof.is_file() else None}


# ---------------------------------------------------------------- 묶기 · 쓰기 · 대조
def build_pin(*, init_adapter_path: Path, config_path: Path | None = None, repo: Path | None = None,
              probes: Mapping[str, Callable[[], dict[str, Any]]] | None = None) -> dict[str, Any]:
    """지금 값으로 착수 고정을 낸다. `probes` 는 시험의 이음새다(`code` · `env` · `common` · `init_adapter`)."""
    pr = dict(probes or {})
    return {"schema": PIN_SCHEMA,
            "code": (pr["code"] if "code" in pr else lambda: code_probe(repo))(),
            "env": (pr["env"] if "env" in pr else lambda: env_probe(repo))(),
            "common": (pr["common"] if "common" in pr else lambda: common_probe(config_path, repo=repo))(),
            "init_adapter": (pr["init_adapter"] if "init_adapter" in pr else lambda: init_probe(init_adapter_path))()}


def pin_sha256(pin: Mapping[str, Any]) -> str:
    return _sha_bytes(_canonical(dict(pin)))


def write_pin(path: Path, pin: Mapping[str, Any]) -> str:
    """쓴다 — 코드가 커밋과 다르면 거부, 있는 파일은 덮지 않는다. 쓴 파일의 sha256 을 돌려준다."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"착수 고정이 이미 있다 — 덮지 않는다: {p}")
    if pin.get("schema") != PIN_SCHEMA:
        raise ValueError(f"착수 고정의 판이 {PIN_SCHEMA} 가 아니다")
    dirty = (pin.get("code") or {}).get("dirty")
    if dirty:
        raise PinMismatch(["code"], f"코드가 커밋과 다르다 — 고정하지 않는다: {list(dirty)[:5]}")
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(pin), sort_keys=True, ensure_ascii=False, indent=1).encode("utf-8") + b"\n"
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "xb") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return _sha_bytes(raw)


def read_pin(path: Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    pin = json.loads(raw.decode("utf-8"))
    if not isinstance(pin, dict) or pin.get("schema") != PIN_SCHEMA:
        raise PinMismatch(["schema"], "착수 고정 파일의 꼴이 아니다")
    return pin, _sha_bytes(raw)


def check_pin(path: Path, current: Mapping[str, Any]) -> str:
    """파일의 고정과 지금 값을 맞댄다. 같으면 파일의 sha256 을, 다르면 `PinMismatch`(다른 칸 전부)."""
    pin, sha = read_pin(path)
    cur = dict(current)
    bad = []
    for part in ("code", "env", "common", "init_adapter"):
        a, b = pin.get(part) or {}, cur.get(part) or {}
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                bad.append(f"{part}.{key}")
    if (cur.get("code") or {}).get("dirty"):
        bad.append("code.dirty")
    if bad:
        raise PinMismatch(sorted(set(bad)))
    return sha
