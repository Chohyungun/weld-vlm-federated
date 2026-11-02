"""본실험 등록(④)을 짓는다 — `vlm.rehearsal_run.stage_register` 의 본실험 갈래(지시 1016l, D 04 §2).

리허설 · 진단 등록과 같은 칸을 같은 함수로 채운다(`evaluation.prereg_unified` 의 등록 객체 · `vlm.export_run` 의 디코딩 · 채움 방향 ·
`stage_register` 와 같은 꼴의 구현 식별자 · 에코 관문의 좌표 기록). 다른 점은 셋이다.

- **모델을 올리지 않는다.** 학습 설정 원문은 본 칸 다섯(로컬 셋 · 중앙 · 연합)의 어댑터 meta 의 원문이고, 다섯이 모두 같아야 한다 —
  export 가 모델 묶음마다 그 원문의 해시를 등록과 맞댄다(`evaluation.actuals.check_actuals`).
- **생성 쪽 실측 일곱 칸은 실측 함수(`vlm.export_measure.measure_env`)의 출력이다** — 서버 실측 파일(`measured_again` 아래)로 받거나
  그 자리에서 잰다. export 가 같은 함수로 다시 재서 등록과 맞댄다.
- **채점 쪽 칸은 출처가 셋이다.**
  - 평가 등록 설정에서 기계적으로 읽는 칸: 봉인한 주 표본(목록 두 해시 · 장수 · 묶음 수) · 가중 · 묶음 단위 · 대표 지표.
  - 구현이 정한 구간 규약: `evaluation.recovery_interval.REGISTERED_CI` · α 0.05.
  - 평가 쪽이 값을 내는 칸: 대조선 · 재표집 · 모집단 식별 · 카나리아의 나머지 — 채점 쪽 입력 파일(`scoring`)이다.
  - 두 출처가 같은 칸에 다른 값을 내면 거부한다.

손으로 옮기는 값 · 추정값을 두지 않는다 — 빈 칸이 하나라도 있으면 거부한다(`missing(side="all")`). 채점 쪽 결속은 채점 진입점과 같은
검사(`strict_binding_problems` · `registered_mismatches` · `rule5.registered_inputs`)로 미리 본다.

**영수증은 쓰지 않는다.** 영수증은 등록이 본줄기에 병합된 **다음** 커밋이다(`configs/registration/README.md`). 병합 뒤
`write_main_receipt` 가 그 커밋의 바이트로 쓴다 — 그 커밋에 영수증이 이미 있으면 거부한다.

| 등록 칸 | 출처 |
|---|---|
| 목적 · 분할 · 동결본 | `main` · `eval` · 설정의 `snapshot_digest`(동결 매니페스트의 검증 digest 와 같아야 한다) |
| 생성 목록 두 해시 | 평가 등록 설정의 `sealed[main_size]` 와 같은 바이트(크기 1,000 봉인) |
| 에코 목록 두 해시 | `--echo-list`(val 분할 — `check_list_split(echo=True)`), 정의 안은 `echo_list_ids` |
| 프롬프트 · 템플릿 · 좌표 · 모델 판 · 배치 · 생성 한도 | 설정(본실험 키) — 프롬프트 해시는 학습 · export 와 같은 `load_prompt` |
| 디코딩 · 채움 방향 | `vlm.export_run.DECODING` · `PADDING_SIDE` |
| 생성 접두 · 프로세서 여섯 | 실측 일곱 칸 |
| 초기 어댑터 digest | 시드 폴더의 `fl/uni_fed/initial.npz`(`vlm.pin.init_probe`) — 착수 고정의 값과 같아야 한다 |
| 학습 설정 원문 · 해시 · 학습량 | 본 칸 다섯의 어댑터 meta 원문(모두 같다) |
| 구현 식별자 | `export_impl_ids` — export 의 실측 · 생성기 적재의 실제 구현과 안쪽 구현 |
| 시드표 | 설정의 시드표에서 그 번호의 값 |
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import fields
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = [
    "CELLS",
    "ECHO_LIST_RAW_SHA256_V1",
    "ECHO_LIST_SET_SHA256_V1",
    "ECHO_LIST_SOURCE_COMMIT",
    "ECHO_SPLIT",
    "LIST_SUFFIXES",
    "REG_DIR",
    "MainRegistrationRefused",
    "build_main_registration",
    "echo_list_ids",
    "export_impl_ids",
    "read_measurement",
    "train_config_from_metas",
    "validate_committed_main_registration",
    "write_main_receipt",
    "write_main_registration",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
REG_DIR = "configs/registration"
LIST_SUFFIXES = {"gen": ".generation.list", "echo": ".echo.list"}
ECHO_SPLIT = "val"
CELLS = ("uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_central", "uni_fed")
SCORING_AREAS = ("metric", "baseline", "ci", "population", "canary")
SCORING_TOP = ("planned_seed_count", "adoption_anchor")
FIXTURES_REL = "vlm/coords_fixtures/golden_fixtures.json"
_INT_KEYS = ("processor_min_pixels", "processor_max_pixels", "patch_size", "merge_size")
_HEX_KEYS = ("processor_config_sha256", "gen_prefix_sha256")
V6_BLOCKS = ("narrative", "clause_axis")
ECHO_LIST_SOURCE_COMMIT = "f6c7af9996c25735a58083c7482f9cffe90dd922"
ECHO_LIST_RAW_SHA256_V1 = "fa4c5ac7ca43248a03903dc7607561b8592112a8865ab49d79d09d5150a9629d"
ECHO_LIST_SET_SHA256_V1 = "2ea9f4cb562a72278cd165a099f492f2147c6f508f60f099ed7dd29852c3c304"
"""D6 — 2 차 등록의 에코 목록은 1 차 등록 커밋의 `configs/registration/main-20261017-1.echo.list` 원본 raw 그대로다(판 6 연결 2 판
§4-3). raw 가 다르면 `echo_bytes`. 집합 sha 는 사람이 적은 값이 아니라 `validate_eval_list` 가 그 raw 에서 다시 낸다."""


class MainRegistrationRefused(ValueError):
    """본실험 등록을 짓지 않는다 — `code` 는 사유 코드다. 사유에는 id 를 싣지 않는다."""

    def __init__(self, code: str, message: str, detail: Any = None):
        self.code, self.detail = code, detail
        super().__init__(message)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def read_measurement(raw: bytes) -> dict[str, Any]:
    """실측 일곱 칸 — 서버 실측 파일이면 `measured_again` 아래, 실측 함수의 출력 그대로면 맨 위. 칸이 정확히 일곱이어야 한다."""
    from vlm.export_measure import MEASURED_KEYS

    R = MainRegistrationRefused
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R("measurement_form", f"실측 파일을 읽지 못했다: {type(exc).__name__}") from None
    m = obj.get("measured_again", obj) if isinstance(obj, dict) else None
    if not isinstance(m, dict) or set(m) != set(MEASURED_KEYS):
        got = set(m) if isinstance(m, dict) else set()
        raise R("measurement_keys", f"실측 칸이 일곱이 아니다 — 빠짐 {sorted(set(MEASURED_KEYS) - got)} · "
                                    f"모름 {sorted(got - set(MEASURED_KEYS))}")
    bad = [k for k in _HEX_KEYS if not _hex64(m[k])]
    bad += [k for k in _INT_KEYS if type(m[k]) is not int or m[k] <= 0]
    if not (isinstance(m["transformers_version"], str) and m["transformers_version"]):
        bad.append("transformers_version")
    if bad:
        raise R("measurement_value", f"실측 칸의 값 꼴이 틀렸다: {bad}")
    return {k: m[k] for k in MEASURED_KEYS}


def train_config_from_metas(metas: Mapping[str, Mapping[str, Any]], *, seed_index: int, seed_value: int,
                            init_digest: str) -> tuple[str, dict[str, Any]]:
    """본 칸 다섯의 어댑터 meta → (학습 설정 원문, 그 객체). 다섯이 모두 있고 본실험 · 그 시드 · 그 초기 어댑터이며 원문이 같아야 한다."""
    R = MainRegistrationRefused
    if set(metas) != set(CELLS):
        raise R("cells", f"본 칸 다섯의 어댑터 meta 가 다 있지 않다 — 빠짐 {sorted(set(CELLS) - set(metas))}")
    texts: dict[str, str] = {}
    for tag in CELLS:
        meta = metas[tag]
        ident = meta.get("identity") or {}
        if (ident.get("purpose"), ident.get("tag"), ident.get("seed_index")) != ("main", tag, seed_index):
            raise R("adapter_identity", f"{tag} 어댑터 meta 의 목적 · 태그 · 시드 번째가 본실험의 이 칸이 아니다")
        if type(ident.get("seed_value")) is not int or ident.get("seed_value") != seed_value:
            raise R("adapter_seed", f"{tag} 어댑터 meta 의 시드 값이 설정의 시드표와 다르다")
        if meta.get("init_adapter_digest") != init_digest:
            raise R("adapter_init", f"{tag} 어댑터 meta 의 초기 어댑터 digest 가 시드 폴더의 initial.npz 와 다르다")
        tc = meta.get("train_config")
        if not isinstance(tc, str) or not tc:
            raise R("train_config_missing", f"{tag} 어댑터 meta 에 학습 설정 원문이 없다")
        texts[tag] = tc
    if len(set(texts.values())) != 1:
        raise R("train_config_differs", f"본 칸의 학습 설정 원문이 같지 않다 — 서로 다른 원문 {len(set(texts.values()))}개")
    text = texts[CELLS[0]]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict) or obj.get("init_adapter_digest") != init_digest:
        raise R("train_config_form", "학습 설정 원문이 객체가 아니거나 그 초기 어댑터 digest 가 initial.npz 와 다르다")
    return text, obj


def echo_list_ids(pair_rows: Iterable[Mapping[str, Any]], removed: Iterable[str], *, split: str = ECHO_SPLIT) -> list[str]:
    """에코 목록의 정의 안 — ECC 대상과 **같은 규칙**을 val 분할에 건다(총괄 결정에 올린다).

    규칙의 정본은 `evaluation.early_coord_check.target_ids`(규칙 파일 4 판 · 미니스펙 10번 13-1 의 m-4): 봉인 페어의 그 분할 행
    가운데 골격 결함이 하나 이상인 장에서 R0 가 뺀 장을 뺀 것, 정렬. 그 함수는 분할이 상수(`SPLIT = "train"`)라 val 에 부를 수
    없다 — 같은 식을 분할 인자로 둔다. 시험이 `split="train"` 에서 그 함수와 같은 값을 보인다.
    """
    out = set()
    for r in pair_rows:
        if r.get("split") == split and (r.get("skeleton") or {}).get("defects"):
            out.add(str(r["image_id"]))
    return sorted(out - set(removed))


def export_impl_ids(purpose: str = "main") -> tuple[str, ...]:
    """export 가 곁 파일에 싣는 구현 식별자의 열쇠 — 실측 · 생성기 적재의 실제 구현과 그 안쪽 구현(`stage_register` 와 같은 꼴).

    등록에 싣는 값은 `impl_key` = `"{seam}={module}:{qualname}@{source_path}"` 다섯이다 — **코드 바이트(blob)는 묶지 않는다**.
    구현이 커밋된 코드인지(작업 트리 blob = HEAD blob)는 export 곁 파일의 식별자 사전을 `evaluation.actuals.impl_problems` 가 따로 본다.
    그래서 '등록과 export 를 같은 코드 커밋에서' 는 운영 규칙이다(03 §5 의 순서) — 코드가 지키는 것은 열쇠(이름 · 경로)의 대조
    (`check_actuals`)까지다.

    본실험은 대역을 받지 않는다 — 바깥 둘(실측 · 생성기 적재) 가운데 승인 목록 밖이 있거나 안쪽 셋이 대역이면
    `MainRegistrationRefused("standin_impl")` 다(G 05 m-5). 대역의 열쇠가 등록에 실려 등록 이름을 버리는 길을 막는다."""
    from evaluation.actuals import impl_key
    from vlm import export_run as XR
    from vlm.export_generator import inner_impl_ids, load_generator
    from vlm.export_measure import measure_env
    from vlm.seams import SeamRejected, impl_id

    loaders = XR.APPROVED_GENERATOR_LOADERS["model"] + XR.APPROVED_GENERATOR_LOADERS["echo"]
    outer_ids = [impl_id(fn, seam=seam, approved=any(fn is a for a in approved))
                 for fn, seam, approved in ((measure_env, "export_measure", XR.APPROVED_MEASURES),
                                            (load_generator, "export_generator", loaders))]
    bad = [d["seam"] for d in outer_ids if not d["approved"]]
    if bad:
        raise MainRegistrationRefused("standin_impl", f"승인 목록 밖의 바깥 구현이다 — 본실험 등록에 싣지 않는다: {bad}")
    try:
        inner = inner_impl_ids(purpose=purpose, standin_allowed=False)
    except SeamRejected as exc:
        raise MainRegistrationRefused("standin_impl", f"안쪽 구현이 대역이다 — 본실험 등록에 싣지 않는다: {exc}") from None
    return tuple(sorted({impl_key(d) for d in outer_ids} | {impl_key(v) for v in inner.values()}))


def _apply_scoring(reg: Any, scoring: Mapping[str, Any]) -> None:
    """채점 쪽 입력 파일의 칸을 등록 객체에 넣는다 — 모르는 칸 · 좌표 기록 · 이미 다른 값이 찬 칸이면 거부한다.

    판 6 의 서술 · 조항 축 블록은 작성기(평가 쪽 `narrative_block` · `clause_axis_block`)가 실제 파일에서 짓는다 — 입력 파일로 자유
    덮어쓰기를 받지 않는다(판 6 연결 2 판 §4-1). 같은 칸이 입력에 있으면 자동 작성 블록과 맞대 다르면 `scoring_conflict` 다."""
    from dataclasses import asdict

    from evaluation import prereg_unified as P

    R = MainRegistrationRefused
    if not isinstance(scoring, Mapping):
        raise R("scoring_form", "채점 쪽 입력이 객체가 아니다")
    v6 = reg.version == P.REGISTRATION_VERSION_V6
    unknown = sorted(set(scoring) - set(SCORING_AREAS) - set(SCORING_TOP) - (set(V6_BLOCKS) if v6 else set()))
    if unknown:
        raise R("scoring_keys", f"채점 쪽 입력의 모르는 칸: {unknown}")
    for area in V6_BLOCKS:
        if area not in scoring:
            continue
        block = getattr(reg, area)
        given = scoring[area]
        if block is None or not isinstance(given, Mapping):
            raise R("scoring_conflict", f"{area}: 판 6 블록은 작성기가 실제 파일에서 짓는다 — 입력으로 받지 않는다")
        mine = json.loads(json.dumps(asdict(block), ensure_ascii=False, default=list))
        if dict(given) != mine:
            raise R("scoring_conflict", f"{area}: 입력의 블록이 작성기가 실제 파일에서 지은 블록과 다르다")
    kinds = P._kinds()
    for area in SCORING_AREAS:
        block = scoring.get(area) or {}
        if not isinstance(block, Mapping):
            raise R("scoring_form", f"채점 쪽 입력의 {area} 가 객체가 아니다")
        spec = getattr(reg, area)
        bad = sorted(set(block) - {f.name for f in fields(spec)})
        if bad:
            raise R("scoring_keys", f"채점 쪽 입력 {area} 의 모르는 칸: {bad}")
        for k, v in block.items():
            kind = kinds[f"{area}.{k}"]
            if kind == "coord_cfg":
                raise R("scoring_keys", f"{area}.{k} 는 등록 짓기가 평가 쪽 함수로 채운다 — 입력으로 받지 않는다")
            if kind == "str_tuple" and isinstance(v, list):
                v = tuple(v)
            cur = getattr(spec, k)
            if cur is not None and cur != v:
                raise R("scoring_conflict", f"{area}.{k}: 평가 등록 설정 · 구현에서 읽은 값과 입력의 값이 다르다")
            setattr(spec, k, v)
    for k in SCORING_TOP:
        if k in scoring:
            setattr(reg, k, scoring[k])


def _json_object(raw: bytes | None) -> dict | None:
    if raw is None:
        return None
    try:
        obj = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _v6_inputs(*, recipe: str, repo: Path | None, pin: Mapping[str, Any], pin_raw: bytes | None,
               inputs_contract_raw: bytes | None, profile: Any, metas: Mapping[str, Mapping[str, Any]],
               echo_raw: bytes) -> tuple[bool, Any, list[str]]:
    """새 체제(판 6)의 입력 — (판 6 인가, 프로필, 다섯 메타의 start_gate.pin_sha256). 구판은 새 입력을 받지 않고(`regime_version`),
    신규는 누락(`v6_form`) · PIN 바이트(`pin_bytes`) · 계약 결속(`pin_contract`) · 계약 실물(`contract_form` · `hash_rule` ·
    `code_bytes`) · D6 에코 raw(`echo_bytes`)를 작성 전에 본다(판 6 연결 2 판 §4-1 · §4-3)."""
    from evaluation import narrative_registration as NR
    from evaluation.eval_list import validate_eval_list
    from evaluation.prereg_unified import REGISTRATION_VERSION_V6
    from vlm.export_preflight import registration_version_of_recipe

    R = MainRegistrationRefused
    new_regime = registration_version_of_recipe(recipe) == REGISTRATION_VERSION_V6
    given = [k for k, v in (("repo", repo), ("pin_raw", pin_raw), ("inputs_contract_raw", inputs_contract_raw),
                            ("profile", profile)) if v is not None]
    if not new_regime:
        if given:
            raise R("regime_version", f"옛 레시피 {recipe!r} 의 등록은 판 5 다 — 새 판 6 입력 {given} 을 받지 않는다")
        return False, None, []
    missing = [k for k in ("repo", "pin_raw", "inputs_contract_raw", "profile") if k not in given]
    if missing:
        raise R("v6_form", f"새 체제(판 6)의 등록 입력이 빠졌다: {missing} — 구판 인자 경로로 판 6 을 짓지 않는다")
    try:
        prof = NR.resolve_profile(profile)
    except NR.ProfileRefused as exc:
        raise R(exc.code, str(exc)) from None
    if prof.profile_id != NR.PROFILE_REGIME2:
        raise R("regime_version", f"레시피 {recipe!r} 의 판 6 은 {NR.PROFILE_REGIME2} 프로필로만 짓는다 — {prof.profile_id!r}")
    if _json_object(pin_raw) != dict(pin):
        raise R("pin_bytes", "착수 고정 바이트가 넘긴 착수 고정 객체와 다르다")
    pin_sha = _sha(pin_raw)
    adapters = [((metas.get(t) or {}).get("start_gate") or {}).get("pin_sha256") for t in CELLS]
    bad = [t for t, a in zip(CELLS, adapters, strict=True) if a != pin_sha]
    if bad:
        raise R("pin_bytes", f"어댑터 meta 의 start_gate.pin_sha256 이 착수 고정 바이트의 sha 와 다르다: {bad}")
    entry = ((pin.get("common") or {}).get("registration") or {}).get(NR.INPUTS_REGISTRATION_KEY)
    if not isinstance(entry, Mapping) or entry.get("sha256") != _sha(inputs_contract_raw):
        raise R("pin_contract", "착수 고정의 학습 등록이 묶은 계약 sha 가 넘긴 계약 바이트와 다르다 — 학습 당시의 계약이 아니다")
    code = NR.contract_code_problems(inputs_contract_raw, repo=Path(repo))
    if code:
        raise R(code[0].split(":", 1)[0], " · ".join(code))
    on_disk = Path(repo) / NR.INPUTS_CONTRACT_REL
    if not on_disk.is_file() or on_disk.read_bytes() != bytes(inputs_contract_raw):
        raise R("v6_contract_sha", f"넘긴 계약 바이트가 저장소의 {NR.INPUTS_CONTRACT_REL} 과 다르다")
    if _sha(echo_raw) != ECHO_LIST_RAW_SHA256_V1:
        raise R("echo_bytes", "2 차 등록의 에코 목록은 1 차 등록 커밋의 에코 목록 원본 raw 그대로여야 한다(D6)")
    if validate_eval_list(echo_raw, where="echo_list").set_sha256 != ECHO_LIST_SET_SHA256_V1:
        raise R("echo_bytes", "에코 목록 raw 에서 다시 낸 집합 sha 가 1 차 등록의 집합 sha 와 다르다(D6)")
    return True, prof, [str(a) for a in adapters]


def build_main_registration(*, cfg: Any, seed_index: int, gen_raw: bytes, echo_raw: bytes, eval_doc: Mapping[str, Any],
                            manifest_meta: Any, snapshot_digest: str, measured: Mapping[str, Any], init_digest: str,
                            pin: Mapping[str, Any], metas: Mapping[str, Mapping[str, Any]], scoring: Mapping[str, Any],
                            impl_ids: Iterable[str], prompt_path: Path, now: datetime, repo: Path | None = None,
                            pin_raw: bytes | None = None, inputs_contract_raw: bytes | None = None,
                            profile: Any = None) -> tuple[Any, bytes, str]:
    """(등록 객체, 등록 파일 바이트, 학습 설정 원문). 입력은 모두 파일 · 실측 함수에서 온 값이다. 어긋나면 `MainRegistrationRefused`.

    구판(1 차 레시피 `json_only`)은 기존 인자 그대로 판 5 를 짓는다. 새 체제(`json_line_then_basis`)는 keyword `repo` · `pin_raw` ·
    `inputs_contract_raw` · `profile`(`regime2_v4`)이 모두 있어야 하고 판 6 을 짓는다 — 서술 · 조항 축 블록은 평가 쪽 작성기가 `repo` 의
    실제 파일에서 짓고, 다섯 메타의 `start_gate.pin_sha256` · 계약 실물 · 착수 고정의 계약 결속 · D6 에코 raw · 판 6 신원 · 계약 ↔ PIN
    (`v6_contract_problems`) · 프로필 조합(`profile_for`)을 작성 앞뒤에서 본다(판 6 연결 2 판 §4-1).
    """
    from evaluation.coord_check import coord_fixture_digest, unified_coord_cfg_record
    from evaluation.entry_gate import EntryRejected, check_list_split
    from evaluation.eval_list import validate_eval_list
    from evaluation.eval_registration import strict_binding_problems
    from evaluation.prereg_unified import (
        REGISTRATION_VERSION_V6,
        RegistrationIncomplete,
        UnifiedRegistration,
        dump_registration,
    )
    from evaluation.recovery_interval import REGISTERED_CI, registered_mismatches
    from evaluation.reject_v14 import BundleRejected
    from evaluation.rule5 import registered_inputs
    from evaluation.sample_weights import WEIGHTING_SAMPLE
    from vlm import export_run as XR
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import load_prompt

    R = MainRegistrationRefused
    if type(seed_index) is not int or not 1 <= seed_index <= len(cfg.seeds):
        raise R("seed_index", f"시드 번호 {seed_index!r} 가 설정의 시드표 1..{len(cfg.seeds)} 밖이다")
    seed_value = int(cfg.seeds[seed_index - 1])
    if snapshot_digest != cfg.snapshot_digest:
        raise R("snapshot", "동결 매니페스트의 검증 digest 가 설정의 snapshot_digest 와 다르다")
    empty = [k for k in ("model_id", "prompt_path", "coord_space", "chat_template_kwargs", "max_new_tokens", "batch_size")
             if getattr(cfg, k, None) in (None, "")]
    if empty:
        raise R("config_incomplete", f"설정의 본실험 칸이 비었다: {empty}")
    # 목록 — 생성 목록은 평가 등록 설정이 봉인한 주 표본, 에코 목록은 val
    try:
        gen = validate_eval_list(gen_raw, where="generation_list")
        echo = validate_eval_list(echo_raw, where="echo_list")
    except BundleRejected as exc:
        raise R("list_form", str(exc)) from None
    pop_doc = eval_doc.get("population") if isinstance(eval_doc, Mapping) else None
    sample = pop_doc.get("sample") if isinstance(pop_doc, Mapping) else None
    sample = sample if isinstance(sample, Mapping) else {}
    if sample.get("weighting") != WEIGHTING_SAMPLE:
        raise R("eval_registration", f"평가 등록 설정의 주 분석 가중이 {WEIGHTING_SAMPLE!r} 가 아니다")
    table = sample.get("sealed")
    sealed = table.get(str(sample.get("main_size"))) if isinstance(table, Mapping) else None
    if not isinstance(sealed, Mapping):
        raise R("main_sample_not_sealed", f"평가 등록 설정에 주 표본 크기 {sample.get('main_size')!r} 의 봉인이 없다")
    if (gen.file_sha256, gen.set_sha256, len(gen)) != (sealed.get("file_sha256"), sealed.get("set_sha256"),
                                                       sealed.get("images")):
        raise R("generation_list_not_sealed", "생성 목록이 평가 등록 설정의 주 표본 봉인(두 해시 · 장수)이 아니다")
    try:
        check_list_split(gen.ids, manifest_meta, "main", where="생성 목록")
        check_list_split(echo.ids, manifest_meta, "main", where="에코 목록", echo=True)
    except EntryRejected as exc:
        raise R("list_split", str(exc)) from None
    # 초기 어댑터 · 착수 고정 · 학습 설정 원문
    if not _hex64(init_digest) or (pin.get("init_adapter") or {}).get("init_adapter_digest") != init_digest:
        raise R("init_vs_pin", "시드 폴더의 initial.npz digest 가 착수 고정의 초기 어댑터 digest 와 다르다")
    tc_text, tc = train_config_from_metas(metas, seed_index=seed_index, seed_value=seed_value, init_digest=init_digest)
    budget = (tc.get("total_epochs"), tc.get("num_rounds"), tc.get("local_epochs"))
    if budget != (cfg.total_epochs, cfg.num_rounds, cfg.local_epochs):
        raise R("budget", f"학습 설정 원문의 학습량 {budget} 이 설정의 (N, R, E) 와 다르다")
    _text, prompt_sha = load_prompt(Path(prompt_path))
    chash = coord_cfg_hash(CoordCfg(coord_space=cfg.coord_space))
    for key, want in (("prompt_sha256", prompt_sha), ("processor_config_sha256", measured["processor_config_sha256"]),
                      ("coord_cfg_hash", chash), ("model_id", cfg.model_id), ("model_revision", cfg.model_revision)):
        if tc.get(key) != want:
            raise R("train_config_vs_config", f"학습 설정 원문의 {key} 가 설정 · 실측과 다르다")
    # 타깃 계약 — 착수 고정에 묶인 학습 등록의 레시피와 맞댄다(계약이 하나뿐이던 때는 보지 않던 칸이다).
    from vlm.pilot_vlm import target_contract_sha256
    from vlm.registration import TRAIN_RECIPES

    recipe = ((pin.get("common") or {}).get("registration") or {}).get("recipe")
    if recipe not in TRAIN_RECIPES:
        raise R("train_recipe", f"착수 고정의 학습 등록에 아는 레시피가 없다: {recipe!r}")
    if tc.get("target_contract_sha256") != target_contract_sha256(recipe):
        raise R("train_config_vs_config", f"학습 설정 원문의 target_contract_sha256 이 등록 레시피 {recipe!r} 의 계약과 다르다")
    # 체제 짝 — 그 레시피의 프롬프트로 학습했고 그 체제의 생성 한도로 평가한다(지시 1022c 덧붙임 A).
    from vlm.registration import regime_problems

    bad = regime_problems(recipe, prompt_sha256=tc.get("prompt_sha256"), gen_limit=cfg.max_new_tokens)
    if bad:
        raise R("train_regime", " · ".join(bad))
    new_regime, prof, adapter_pins = _v6_inputs(recipe=recipe, repo=repo, pin=pin, pin_raw=pin_raw,
                                                inputs_contract_raw=inputs_contract_raw, profile=profile, metas=metas,
                                                echo_raw=echo_raw)

    reg = UnifiedRegistration()
    if new_regime:
        from evaluation import narrative_registration as NR

        reg.version = REGISTRATION_VERSION_V6
        try:
            reg.narrative = NR.narrative_block(Path(repo), profile=prof)
            reg.clause_axis = NR.clause_axis_block(Path(repo), profile=prof)
        except NR.ProfileRefused as exc:
            raise R(exc.code, str(exc)) from None
    g = reg.generation
    g.purpose, g.list_split, g.snapshot_digest = "main", "eval", cfg.snapshot_digest
    g.eval_list_file_sha256, g.eval_list_set_sha256 = gen.file_sha256, gen.set_sha256
    g.echo_list_file_sha256, g.echo_list_set_sha256 = echo.file_sha256, echo.set_sha256
    g.prompt_sha256, g.chat_template_kwargs = prompt_sha, dict(cfg.chat_template_kwargs)
    g.gen_prefix_sha256 = measured["gen_prefix_sha256"]
    g.max_new_tokens = int(cfg.max_new_tokens)
    g.decoding, g.batch_size, g.padding_side = dict(XR.DECODING), int(cfg.batch_size), XR.PADDING_SIDE
    for k in ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels", "patch_size", "merge_size",
              "transformers_version"):
        setattr(g, k, measured[k])
    g.coord_space, g.coord_cfg_hash = cfg.coord_space, chash
    g.base_model_id, g.base_model_revision = cfg.model_id, cfg.model_revision
    g.init_adapter_digest, g.start_checkpoint_sha256 = init_digest, None
    g.train_config_sha256 = _sha(tc_text.encode("utf-8"))
    g.budget_n, g.budget_r, g.budget_e = budget
    g.impl_ids = tuple(sorted(impl_ids))
    # 채점 쪽 — 평가 등록 설정에서 읽는 칸 · 구현의 구간 규약 · 에코 관문의 좌표 기록 · 시드표
    rep = eval_doc.get("representative") if isinstance(eval_doc.get("representative"), Mapping) else {}
    m = reg.metric
    m.metric_id, m.iou_threshold = rep.get("metric_id"), rep.get("iou_threshold")
    m.class_codes = tuple(str(c) for c in rep.get("class_codes") or ()) or None
    m.coupling_rule, m.sensitivity_variant_id = rep.get("coupling_rule"), rep.get("sensitivity_variant")
    m.macro_support_rule = rep.get("macro_support_rule")
    p = reg.population
    p.weighting, p.cluster_unit = WEIGHTING_SAMPLE, sample.get("cluster_unit")
    p.eval_list_file_sha256, p.eval_list_set_sha256 = gen.file_sha256, gen.set_sha256
    p.n_images, p.n_clusters = len(gen), sealed.get("groups")
    for k, v in REGISTERED_CI.items():
        setattr(reg.ci, k, v)
    reg.ci.alpha, reg.ci.cluster_unit = 0.05, sample.get("cluster_unit")
    fd = coord_fixture_digest((REPO_ROOT / FIXTURES_REL).read_bytes())
    reg.canary.coord_cfg = unified_coord_cfg_record(
        chash, fixture_digest=fd, registered_on=now.date().isoformat(),
        coords_source=(REPO_ROOT / "vlm/coords.py").read_bytes(),
        hash_record=(REPO_ROOT / "vlm/coords_fixtures/HASH_RECORD.yaml").read_bytes())
    reg.canary.coord_fixture_sha256 = fd
    reg.seeds = {str(seed_index): seed_value}
    _apply_scoring(reg, scoring)
    # 완결 · 결속 — 채점 진입점과 같은 검사
    bad = reg.invalid(side="all")
    if bad:
        raise R("registration_invalid", f"등록 칸의 값이 틀렸다 — {len(bad)}건: " + " · ".join(bad[:5]))
    gaps = reg.missing(side="all")
    if gaps:
        raise R("registration_incomplete", f"빈 칸 {len(gaps)}개 — 채점 쪽 입력이 채울 칸: {gaps}", detail=gaps)
    problems = list(strict_binding_problems(reg, dict(eval_doc))) + list(registered_mismatches(reg.ci))
    try:
        registered_inputs(reg)
    except ValueError as exc:
        problems.append(f"rule5: {exc}")
    if problems:
        raise R("eval_binding", f"평가 등록 설정 · 구현과 맞지 않는다 — {len(problems)}건: " + " · ".join(problems[:5]))
    if new_regime:
        from evaluation import narrative_registration as NR

        v6 = NR.identity_problems(reg, repo=Path(repo), profile=prof, prompt_path=Path(prompt_path))
        v6 += NR.v6_contract_problems(reg, inputs_contract_raw, pin_raw=pin_raw, adapter_pin_sha256=adapter_pins)
        if v6:
            raise R(v6[0].split(":", 1)[0], f"판 6 의 신원 · 계약 · 착수 고정 대조 — {len(v6)}건: " + " · ".join(v6[:5]))
        try:
            got = NR.profile_for(reg, train_entry=(pin.get("common") or {}).get("registration"))
        except NR.ProfileRefused as exc:
            raise R(exc.code, str(exc)) from None
        if got.profile_id != prof.profile_id:
            raise R("profile_mismatch", f"지은 등록의 프로필 {got.profile_id!r} 이 요청한 {prof.profile_id!r} 이 아니다")
    try:
        reg.require_complete(side="all")
    except RegistrationIncomplete as exc:
        raise R("registration_incomplete", str(exc)) from None
    return reg, dump_registration(reg), tc_text


def write_main_registration(out_dir: Path, *, stem: str, raw: bytes, tc_text: str, gen_raw: bytes,
                            echo_raw: bytes, eval_raw: bytes | None = None, sources_raw: bytes | None = None) -> dict[str, Path]:
    """커밋할 파일 — 구판(판 5) 넷: `{stem}.json` · `{stem}.train_config.json`(끝 LF 없음) · `{stem}.generation.list` · `{stem}.echo.list`.
    새 체제(판 6) 여섯: 넷 + `{stem}.eval.yaml`(고정 후보 바이트 그대로) + `{stem}.v6_sources.json`(D3 출처 운반). 이름은
    `configs/registration/README.md` 의 꼴이다. 여섯을 모두 검사한 뒤 배타 생성한다 — 하나라도 있으면 첫 쓰기 전에 `name_taken`, 검사
    실패면 새 파일 0. 쓰는 중 I/O 실패는 부분 후보를 남기고 `write_failed`(쓴 파일 목록을 detail 에) — 같은 이름 재시도는 name_taken
    으로 막힌다. OS 원자성을 주장하지 않는다. **영수증은 쓰지 않는다** — 다음 커밋의 일이다."""
    from evaluation import eval_registration as ER
    from evaluation.prereg_unified import (
        REGISTRATION_VERSION_V6,
        RegistrationFileError,
        load_registration,
    )
    from vlm.export_preflight import TRAIN_CONFIG_SUFFIX

    R = MainRegistrationRefused
    try:
        ER.registration_stem(f"{REG_DIR}/{stem}.json")
    except ER.EvalPathRefused as exc:
        raise R("name", f"본실험 등록 이름이 main-YYYYMMDD-n 꼴이 아니다: {stem!r} — {exc}") from None
    try:
        reg = load_registration(raw)
    except RegistrationFileError as exc:
        raise R("registration_form", str(exc)) from None
    v6 = reg.version == REGISTRATION_VERSION_V6
    if v6 and (eval_raw is None or sources_raw is None):
        raise R("v6_form", "판 6 등록은 평가 설정(.eval.yaml)과 출처 운반 파일(.v6_sources.json)을 함께 쓴다 — 둘 다 있어야 한다")
    if not v6 and (eval_raw is not None or sources_raw is not None):
        raise R("regime_version", "판 5 등록에는 곁 파일 둘을 쓰지 않는다")
    d = Path(out_dir)
    files = {"registration": (d / f"{stem}.json", raw), "train_config": (d / f"{stem}{TRAIN_CONFIG_SUFFIX}", tc_text.encode("utf-8")),
             "generation_list": (d / f"{stem}{LIST_SUFFIXES['gen']}", gen_raw),
             "echo_list": (d / f"{stem}{LIST_SUFFIXES['echo']}", echo_raw)}
    if v6:
        try:
            doc = ER.load_eval_registration(eval_raw)
        except ER.EvalRegistrationError as exc:
            raise R("eval_binding", f"평가 설정 바이트를 읽지 못했다: {exc}") from None
        problems = ER.strict_binding_problems(reg, doc)
        if problems:
            raise R("eval_binding", f"평가 설정과 등록이 맞지 않는다 — {len(problems)}건: " + " · ".join(problems[:3]))
        src = _json_object(sources_raw)
        if src is None or src.get("registration_sha256") != _sha(raw) or src.get("eval_sha256") != _sha(eval_raw):
            raise R("v6_sources_mismatch", "출처 운반 파일이 이 등록 · 평가 설정 바이트를 묶지 않았다")
        files["eval_registration"] = (d / f"{stem}{ER.SIDECAR_EVAL_SUFFIX}", eval_raw)
        files["v6_sources"] = (d / f"{stem}{ER.SIDECAR_SOURCES_SUFFIX}", sources_raw)
    d.mkdir(parents=True, exist_ok=True)
    taken = [p.name for p, _ in files.values() if p.exists()] + [p.name for p in d.glob(f"{stem}.receipt.json")]
    if taken:
        raise R("name_taken", f"같은 이름의 파일이 이미 있다 — 새 이름으로: {taken}")
    written: list[str] = []
    for p, b in files.values():
        try:
            with open(p, "xb") as fh:
                fh.write(b)
        except OSError as exc:
            raise R("write_failed", f"쓰는 중 실패했다 — 부분 후보 {written} 를 남긴다, 같은 이름으로 다시 쓰지 않는다: {exc}",
                    detail=list(written)) from None
        written.append(p.name)
    return {k: p for k, (p, _) in files.items()}


def validate_committed_main_registration(*, registration_rel: str, commit: str, checkout: Path):
    """커밋된 등록의 첫 정식 소비자(판 6 연결 2 판 §2-2) — 등록 · 학습 설정 원문 · 목록 둘 · 평가 설정 · 출처를 **그 커밋의 blob** 으로
    다시 읽어 맞댄다. 메모리 객체로 검사를 대신하지 않는다. 돌려주는 값은 평가 쪽 `BoundEval`(경로 · 바이트 · 프로필).

    본다: 등록 경로의 닫힌 꼴 · 커밋이 풀림 · 등록이 완결이고 목적 main · 학습 설정 원문 · 목록 해시가 등록 칸과 같음 · 평가 설정 결속
    (`load_bound_eval` — 판 5 는 옛 경로, 신규는 같은 줄기의 `.eval.yaml` · `.v6_sources.json`) · 신규의 에코 raw(D6). 영수증은 이
    함수 통과 전에 발행할 수 없다(`write_main_receipt` 가 먼저 부른다). 본줄기 조상 검사는 영수증 쪽에 둔다."""
    from evaluation import eval_registration as ER
    from evaluation import narrative_registration as NR
    from evaluation.entry_gate import EntryRejected, blob_at
    from evaluation.eval_list import validate_eval_list
    from evaluation.prereg_unified import (
        RegistrationFileError,
        RegistrationIncomplete,
        RegistrationInvalid,
        load_registration,
    )
    from evaluation.reject_v14 import BundleRejected
    from vlm.export_preflight import echo_train_config_rel

    R = MainRegistrationRefused
    checkout = Path(checkout)
    try:
        stem = ER.registration_stem(registration_rel)
    except ER.EvalPathRefused as exc:
        raise R("receipt_name", f"본실험 등록 경로가 {REG_DIR}/main-YYYYMMDD-n.json 이 아니다: {registration_rel} — {exc}") from None
    full = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                          capture_output=True, text=True, check=False)
    if full.returncode != 0:
        raise R("commit", f"커밋 {commit} 을 풀 수 없다")
    sha = full.stdout.strip()
    try:
        raw = blob_at(checkout, sha, registration_rel)
        tc = blob_at(checkout, sha, echo_train_config_rel(registration_rel))
        lists = {k: blob_at(checkout, sha, f"{REG_DIR}/{stem}{s}") for k, s in LIST_SUFFIXES.items()}
    except EntryRejected as exc:
        raise R("receipt_blob", str(exc)) from None
    try:
        reg = load_registration(raw)
        reg.require_complete(side="all")
    except (RegistrationFileError, RegistrationIncomplete, RegistrationInvalid) as exc:
        raise R("registration_form", str(exc)) from None
    g = reg.generation
    if g.purpose != "main":
        raise R("registration_form", "커밋한 등록의 목적이 main 이 아니다")
    if _sha(tc) != g.train_config_sha256:
        raise R("train_config_vs_registration", "커밋한 학습 설정 원문의 해시가 등록의 train_config_sha256 과 다르다")
    for k, (f_sha, s_sha) in {"gen": (g.eval_list_file_sha256, g.eval_list_set_sha256),
                              "echo": (g.echo_list_file_sha256, g.echo_list_set_sha256)}.items():
        try:
            el = validate_eval_list(lists[k], where=k)
        except BundleRejected as exc:
            raise R("list_vs_registration", str(exc)) from None
        if (el.file_sha256, el.set_sha256) != (f_sha, s_sha):
            raise R("list_vs_registration", f"커밋한 {LIST_SUFFIXES[k]} 의 두 해시가 등록의 칸과 다르다")
    try:
        bound = ER.load_bound_eval(repo=checkout, anchor_rev=sha, registration_rel=registration_rel, registration=reg)
    except ER.EvalBindingRefused as exc:
        raise R(exc.code, f"그 커밋의 평가 설정 결속 — {len(exc.problems)}건: " + " · ".join(exc.problems[:3])) from None
    if bound.profile_id == NR.PROFILE_REGIME2 and _sha(lists["echo"]) != ECHO_LIST_RAW_SHA256_V1:
        raise R("echo_bytes", "커밋한 2 차 등록의 에코 목록이 1 차 등록 커밋의 원본 raw 가 아니다(D6)")
    return bound


def write_main_receipt(*, registration_rel: str, commit: str, checkout: Path, now: datetime | None = None) -> Path:
    """본실험 영수증 — 등록이 병합된 본줄기 커밋의 **바이트**로 쓴다(README: 영수증은 그 다음 커밋이다).

    본다: 커밋이 `refs/heads/main` 의 조상 · 그 커밋에 등록 · 학습 설정 원문 · 목록 둘이 README 의 이름으로 있고, 학습 설정과
    목록의 해시가 등록의 칸과 같다 · 등록이 완결(생성 · 채점)이고 목적이 main · 그 커밋의 평가 등록 설정과 결속이 맞다 ·
    그 커밋에 영수증이 아직 없다. 영수증은
    체크아웃의 `configs/registration/{stem}.receipt.json` 에 배타 생성한다. 돌려주는 값은 영수증 경로다."""
    from evaluation import eval_registration as ER
    from evaluation.entry_gate import EntryRejected, blob_at, is_ancestor
    from evaluation.prereg_unified import load_registration

    R = MainRegistrationRefused
    checkout = Path(checkout)
    rel = PurePosixPath(registration_rel)
    try:
        stem = ER.registration_stem(registration_rel)
    except ER.EvalPathRefused as exc:
        raise R("receipt_name", f"본실험 등록 경로가 {REG_DIR}/main-YYYYMMDD-n.json 이 아니다: {registration_rel} — {exc}") from None
    full = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                          capture_output=True, text=True, check=False)
    if full.returncode != 0 or not is_ancestor(checkout, full.stdout.strip()):
        raise R("receipt_not_ancestor", f"커밋 {commit} 이 본줄기의 조상이 아니다")
    sha = full.stdout.strip()
    receipt_rel = f"{REG_DIR}/{stem}.receipt.json"
    if subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", f"{sha}:{receipt_rel}"],
                      capture_output=True, check=False).returncode == 0:
        raise R("receipt_same_commit", "그 커밋에 영수증이 이미 있다 — 영수증은 등록이 병합된 다음 커밋이다(README)")
    # 커밋 후 검수 — 등록 · 원문 · 목록 · 평가 설정 · 출처를 그 커밋의 blob 으로(첫 정식 소비자). 통과 전에는 영수증이 없다.
    validate_committed_main_registration(registration_rel=rel.as_posix(), commit=sha, checkout=checkout)
    try:
        raw = blob_at(checkout, sha, rel.as_posix())
    except EntryRejected as exc:
        raise R("receipt_blob", str(exc)) from None
    reg = load_registration(raw)
    now = now or datetime.now().astimezone()
    receipt = {"generation_sha256": reg.generation_sha256(), "scoring_sha256": reg.scoring_sha256(),
               "registered_at": now.isoformat(timespec="microseconds"), "main_commit": sha, "kind": "main",
               "registration_path": rel.as_posix(), "registration_file_sha256": _sha(raw)}
    rp = checkout / receipt_rel
    with open(rp, "xb") as fh:
        fh.write((json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return rp
