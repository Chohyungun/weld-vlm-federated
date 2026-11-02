"""레시피별 행별 감독 토큰 표 — 학습 렌더의 감독 구간(타깃 + `<|im_end|>` + 줄바꿈)의 토큰 수를 학습 행마다 센다.

    python -X utf8 -m vlm.token_table --recipe json_line_then_basis --out outputs/target_tokens_v2_20261023/per_row.csv [--config configs/base.yaml]

- **셈** — `len(tokenizer(build_target(행, recipe), 특수 토큰 없음)) + 2`. 2026-10-20 설계 복구 02a 의 갈래 (A) 와 같은 셈이고, 그 셈이
  채팅 템플릿 렌더(갈래 B) · 학습 렌더(갈래 C)와 같은 값을 냈다(02 §2-1). 타깃은 학습 코드의 `vlm.pilot_vlm.build_target` 그대로다.
- **행** — 페어의 학습 행(split = train)을 파일 순서로. 이미지는 열지 않는다 — ABS_ORIG 좌표는 기하가 바꾸지 않으므로 페어의 `width_px` ·
  `height_px` 로 만든다(하네스 04 의 `target_len_fn` 과 같다).
- **프로세서** — 설정의 모델 · 판 · 프로세서 인자로 로컬 캐시에서만 연다. 모델 가중치 0 · GPU 0.
- **산출** — 표(`image_id,client,supervised_tokens`, UTF-8 · LF)와 곁의 출처 파일(`<표>.provenance.json` — 꼴 · 레시피 · 타깃 계약 해시 ·
  페어 경로 · 페어 sha256 · 페어 digest · 계측 코드의 LF sha256 · 토크나이저 신원 · 행 수 · 참여자별 합 · 표 sha256). 있는 파일은 덮지 않는다.
- **신원** — 토크나이저 신원(`tokenizer_identity`)은 모델 저장소 · 판(revision) · 그 스냅샷의 토크나이저 · 프로세서 파일(`IDENTITY_FILES` 가운데
  있는 것)의 sha256 · transformers · tokenizers 판이다. 모델 가중치를 읽지 않는다. 착수 조건(`vlm.budget.token_table_problems` — 실행 가드 ·
  학습 등록 · 착수 고정)이 출처의 신원(`IDENTITY_KEYS`)과 계측 코드 sha256 을 지금 환경 · 지금 이 파일과 맞댄다(외부 검토 200 M-01) —
  페어 · 레시피 · 합이 같아도 판만 바뀌면 멈춘다. **이 파일을 고치면 표를 다시 낸다**(출처의 계측 코드 sha256 이 지금 파일과 갈린다).

1 차 레시피의 표(봉인 폴더 안)는 이 모듈로 다시 쓰지 않는다 — 읽기만 한다(`vlm.budget.TOKEN_TABLES`). 그 표에는 출처 파일이 없어
신원이 기록되지 않았다(v1: 신원 미기록) — 1 차는 표 지문 고정만 본다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["IDENTITY_FILES", "IDENTITY_KEYS", "METHOD", "PROVENANCE_SCHEMA", "count_rows", "identity_problems", "provenance",
           "source_sha256", "table_bytes", "tokenizer_identity"]

METHOD = "len(tokenizer(build_target(행, recipe), add_special_tokens=False)) + 2 — 타깃 + <|im_end|> + 줄바꿈"
CHUNK = 512
#: 출처 파일의 꼴. 2 는 토크나이저 · 프로세서 파일의 신원(`tokenizer.files`)을 든다(2026-10-23). 10-22 의 첫 출처에는 이 칸이 없다.
PROVENANCE_SCHEMA = 2
#: 감독 토큰 셈을 가르는 모델 저장소 파일 — 토크나이저(어휘 · 병합 · 설정 · 특수 토큰 · 채팅 틀 — 셈의 + 2 는 틀이 붙이는
#: `<|im_end|>` 와 줄바꿈이다)와 프로세서 설정. 스냅샷에 있는 것만 잰다.
IDENTITY_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
                  "merges.txt", "tokenizer.model", "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
                  "processor_config.json", "video_preprocessor_config.json")
#: 출처의 토크나이저 신원 가운데 착수 조건이 지금 환경과 맞대는 칸. 적재한 프로세서에서만 나는 값(토크나이저 클래스 · 어휘 수 ·
#: 이미지 프로세서 설정의 해시)은 기록만 한다 — 이 칸들(파일 바이트 · 판)이 그 값을 정하고, 맞대는 자리는 모델 가중치도 프로세서도
#: 올리지 않는다.
IDENTITY_KEYS = ("model_id", "revision", "files", "transformers", "tokenizers")


def source_sha256() -> str:
    """계측 코드(이 모듈)의 LF 정규화 sha256 — 출처 파일에 싣는다."""
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _cached_file(model_id: str, revision: str | None, name: str) -> Path | None:
    """모델 저장소 스냅샷의 파일 — 로컬 캐시에서만, 학습의 적재(`from_pretrained`)와 같은 해석으로 푼다. 없으면 None."""
    from transformers.utils import cached_file

    p = cached_file(model_id, name, revision=revision, local_files_only=True, _raise_exceptions_for_missing_entries=False)
    return None if p is None else Path(p)


def _package_version(name: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def tokenizer_identity(model_id: str, revision: str | None) -> dict[str, Any]:
    """지금 환경의 토크나이저 신원 — 모델 저장소 · 판 · 스냅샷 파일(`IDENTITY_FILES`)의 sha256 · transformers · tokenizers 판.

    로컬 캐시의 작은 파일만 읽는다(모델 가중치 0). 토크나이저 설정을 찾지 못하면 `FileNotFoundError` — 맞대지 못한 표는 쓰지 않는다."""
    files = {}
    for name in IDENTITY_FILES:
        p = _cached_file(model_id, revision, name)
        if p is not None:
            files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if "tokenizer_config.json" not in files:
        raise FileNotFoundError(f"로컬 캐시에 {model_id}@{revision} 의 토크나이저 설정(tokenizer_config.json)이 없다")
    return {"model_id": model_id, "revision": revision, "files": files,
            "transformers": _package_version("transformers"), "tokenizers": _package_version("tokenizers")}


def identity_problems(recorded: Any, now: Mapping[str, Any]) -> list[str]:
    """출처에 적힌 토크나이저 신원과 지금 환경의 신원(`IDENTITY_KEYS`)의 어긋남. 빈 목록이면 같다. 값과 자료형을 함께 본다."""
    if not isinstance(recorded, Mapping):
        return ["출처 파일에 토크나이저 신원이 없다"]
    out = []
    for key in IDENTITY_KEYS:
        a, b = recorded.get(key), now.get(key)
        if key == "files" and isinstance(a, Mapping) and isinstance(b, Mapping):
            diff = sorted((set(a) ^ set(b)) | {n for n in set(a) & set(b) if a[n] != b[n]})
            if diff:
                out.append(f"토크나이저 · 프로세서 파일 {diff} 가 출처(계측 때)와 지금 환경에서 다르다")
        elif type(a) is not type(b) or a != b:
            out.append(f"토크나이저 신원 {key} — 출처 {a!r} ≠ 지금 환경 {b!r}")
    return out


def count_rows(rows: Sequence[Mapping[str, Any]], tokenize: Callable[[list[str]], list[list[int]]], *,
               recipe: str) -> list[tuple[str, str, int]]:
    """학습 행마다 (image_id, client, 감독 토큰). `tokenize` 는 문자열 목록 → id 목록의 목록(특수 토큰 없음)이다."""
    from vlm.coords import CoordCfg, ImageGeom
    from vlm.pilot_vlm import build_target

    cfg = CoordCfg(coord_space="ABS_ORIG")
    out: list[tuple[str, str, int]] = []
    for i in range(0, len(rows), CHUNK):
        part = rows[i:i + CHUNK]
        targets = [build_target(r, ImageGeom(orig_w=int(r["width_px"]), orig_h=int(r["height_px"])), cfg, recipe=recipe)
                   for r in part]
        ids = tokenize(targets)
        if len(ids) != len(part):
            raise ValueError("토크나이저가 받은 타깃 수와 다른 수를 냈다")
        out.extend((str(r["image_id"]), str(r["client"]), len(t) + 2) for r, t in zip(part, ids))
    return out


def table_bytes(counts: Iterable[tuple[str, str, int]]) -> bytes:
    """표의 바이트 — 머리 줄 + 행, UTF-8 · LF · BOM 없음."""
    lines = ["image_id,client,supervised_tokens"] + [f"{i},{c},{int(n)}" for i, c, n in counts]
    return ("\n".join(lines) + "\n").encode("utf-8")


def provenance(*, recipe: str, counts: Sequence[tuple[str, str, int]], table: bytes, pairs_path: str,
               pairs_sha256: str, pairs_digest: str, tokenizer_identity: Mapping[str, Any]) -> dict[str, Any]:
    from vlm.pilot_vlm import target_contract_sha256

    per_client: dict[str, dict[str, int]] = {}
    for _i, c, n in counts:
        d = per_client.setdefault(c, {"rows": 0, "supervised_tokens": 0})
        d["rows"] += 1
        d["supervised_tokens"] += int(n)
    return {"kind": "per_row_supervised_tokens", "schema": PROVENANCE_SCHEMA, "recipe": recipe,
            "contract_sha256": target_contract_sha256(recipe), "method": METHOD, "split": "train",
            "pairs": {"path": pairs_path, "sha256": pairs_sha256}, "pairs_digest": pairs_digest,
            "counter": {"module": "vlm.token_table", "sha256_lf": source_sha256()},
            "tokenizer": dict(tokenizer_identity), "n_rows": len(counts),
            "per_client": {c: per_client[c] for c in sorted(per_client)},
            "table_sha256": hashlib.sha256(table).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser(description="레시피별 행별 감독 토큰 표를 낸다(CPU · 토크나이저만)")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", type=Path, required=True, help="표 파일(있으면 거부) — 곁에 출처 파일을 쓴다")
    ap.add_argument("--config", type=Path, default=None, help="통합형 설정 파일(기본: configs/base.yaml)")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from vlm.budget import PROVENANCE_SUFFIX
    from vlm.pilot_vlm import RECIPE_V1, TARGET_CONTRACTS, load_pairs, processor_config_sha256
    from vlm.uni_config import load_uni_config, pairs_snapshot_problems

    if a.recipe not in TARGET_CONTRACTS or a.recipe == RECIPE_V1:
        print(f"새 레시피만 낸다 — 1 차 레시피의 표는 봉인 폴더의 것을 읽는다: {a.recipe!r}", file=sys.stderr)
        return 2
    prov_path = a.out.with_name(a.out.name + PROVENANCE_SUFFIX)
    if a.out.exists() or prov_path.exists():
        print(f"표 또는 출처 파일이 이미 있다 — 덮지 않는다: {a.out}", file=sys.stderr)
        return 2
    cfg = load_uni_config(a.config)
    bad = pairs_snapshot_problems(cfg.pairs_path, cfg.pairs_digest)
    if bad:
        print("페어 스냅샷이 설정과 다르다: " + " · ".join(bad), file=sys.stderr)
        return 2
    ident = tokenizer_identity(cfg.model_id, cfg.model_revision)        # 착수 조건이 맞대는 신원 — 셈과 같은 캐시 · 판
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.model_revision, local_files_only=True,
                                         **dict(cfg.processor_kwargs or {}))
    tok = proc.tokenizer
    rows = load_pairs("train", None, pairs_path=cfg.pairs_path)
    counts = count_rows(rows, lambda ts: tok(ts, add_special_tokens=False)["input_ids"], recipe=a.recipe)
    raw = table_bytes(counts)
    ident.update({"class": type(tok).__name__, "vocab_size": len(tok), "processor_config_sha256": processor_config_sha256(proc)})
    rec = provenance(recipe=a.recipe, counts=counts, table=raw, pairs_path=Path(cfg.pairs_path).as_posix(),
                     pairs_sha256=hashlib.sha256(Path(cfg.pairs_path).read_bytes()).hexdigest(),
                     pairs_digest=str(cfg.pairs_digest), tokenizer_identity=ident)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "xb") as fh:
        fh.write(raw)
    with open(prov_path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print("wrote", a.out.as_posix(), rec["table_sha256"], rec["n_rows"], json.dumps(rec["per_client"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
