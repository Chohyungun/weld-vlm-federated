"""평가 등록 설정 판독기 — `configs/eval_registration.yaml` 을 코드 상수 · `configs/base.yaml` · 등록 객체와 맞댄다.

## 왜 있나

대표 지표 이름 · 표본 시드 · 봉인 해시 · 민감도 변형 · 결합 규칙은 코드 상수와 설정 파일과 문서(실험설계 8-5)에 세 벌로 적힌다.
세 벌이 어긋나도 아무도 알리지 않았다(총괄 검토 2026-10-13 · 1013o 의 4). 학습 쪽은 `vlm/registration.py` 가 `train_registration.yaml`
을 코드 상수와 맞대는데 평가 쪽에는 그 짝이 없었다. 이 모듈이 그 짝이다 — 설정의 값이 코드가 실제로 쓰는 값인지, 본실험 등록
객체의 모집단 칸이 봉인한 표본과 같은지 본다.

## 무엇을 맞대나

| 설정 칸 | 코드의 정본 |
|---|---|
| `representative.metric_id` · `population.sample.weighting` · `population.sample.side_file_header` | `evaluation.sample_weights` 의 상수 |
| `representative.sensitivity_variant` | `evaluation.metrics.box_f1.SENSITIVITY_REGISTERED` |
| `representative.coupling_rule` | `evaluation.adapters_v14.COUPLING_RULE_V2` |
| `representative.class_codes` | `evaluation.conformance_v14.SCORING_CODES`(사상표에서 온다) |
| `representative.macro_support_rule` | `evaluation.prereg_unified.VOCAB` |
| `population.sample.seed` · `sealed` 의 키 집합 | `data.eval_sample.SEED` · `SIZES`(표본 추출기) + `EXTRA_SEALED_SIZES` |
| `population.snapshot_digest` · `generation.max_new_tokens` | `configs/base.yaml` 의 `fixed_before_main_runs` |

`check_population_binding` 은 본실험 등록 객체(`UnifiedRegistration`)의 모집단 칸이 `stratified_g_over_n` 이면 그 목록 해시 · 장수가
`sealed[main_size]` 와 같아야 한다고 본다 — 봉인한 목록이 아닌 것으로 본채점이 돌지 않게.

    python -X utf8 -B -m evaluation.eval_registration [--file configs/eval_registration.yaml] [--base configs/base.yaml]

문제가 하나라도 있으면 줄마다 적고 종료 코드 1. 설정 파일이 없으면 그것도 문제다 — "없음" 을 "맞음" 으로 읽지 않는다.
**`--base` 도 같다** — CLI 는 base 설정을 늘 맞댄다. 파일이 없거나 객체(사상)가 아니거나 `fixed_before_main_runs` 가 없으면
대조를 생략하지 않고 종료 1 이다(외부 검토 137-2 M-04 — 앞 판은 없는 파일을 `None` 으로 넘겨 대조 없이 통과할 수 있었다).
라이브러리 함수 `registration_problems` 의 `base_doc=None` 은 **부르는 쪽이 고른 생략**이다 — 값이 주어졌는데 객체가 아니면 문제다.

## 2 차 설정의 결정적 곁 파일(판 6 연결 2 판 — 263 §2)

정식 등록 `configs/registration/{purpose}-{YYYYMMDD}-{n}.json` 과 **같은 커밋**의 같은 줄기 `.eval.yaml`(`eval_sidecar_rel`)이 2 차 등록의
평가 설정이다. 네 소비자(커밋 후 검수 · 영수증 · export · 엄격 채점)는 `load_bound_eval` 하나로 **anchor 커밋의 blob** 만 읽는다 — HEAD · 작업
트리 · `candidates/` · 명령줄 설정으로 대신하지 않는다(신규 체제에 곁 파일이 없으면 `eval_sidecar_missing`). 2 차 설정은 1 차 설정 바이트
(`SOURCE_EVAL_SHA256_V1`)에서 `build_regime2_eval` 로 도출된 것이어야 하고(`regime2_eval_problems` — `eval_binding` · `aux_names`), 출처
운반 파일(`{stem}.v6_sources.json`)이 그 바이트를 묶는다(`eval_bytes`). 판 5 와 보존 기준 커밋의 역사적 판 6 은 옛 `CONFIG_REL` 경로 그대로다.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_REL = "configs/eval_registration.yaml"
BASE_REL = "configs/base.yaml"
CANDIDATE_REL = "docs/dev_log/2026-10-16-본실험실행/eval_lists_D/eval_registration_후보_1000_D.yaml"
"""평가 쪽이 값을 산출 파일에서 채운 후보 — 데이터 쪽이 `CONFIG_REL` 로 옮긴다(바이트 그대로). 앞 판(봉인 2000 · 3000 · 4000, 주 크기 3000)은
`docs/dev_log/2026-10-07-본실험코퍼스/28_eval_registration_후보_D.yaml` 이다. 이 판은 크기 1000 을 봉인하고 주 크기로 둔다."""
KIND = "eval_registration"
IOU_THRESHOLD = "1/2"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXTRA_SEALED_SIZES = (1000,)
"""추출기의 기본 크기(`data.eval_sample.SIZES`) 밖에 봉인한 크기 — 같은 시드 · 같은 규칙(`sample_eval(…, sizes=[…])`)으로 뽑은 같은
묶음 순열의 앞이다(§2-6, 기본 크기들의 부분집합). 1000: 본실험 평가 표본(의사결정로그 2026-10-17 — 1,059 장 · 101 묶음)."""
REG_DIR = "configs/registration"
TRAIN_REG_REL = "configs/train_registration.yaml"
SIDECAR_EVAL_SUFFIX = ".eval.yaml"
SIDECAR_SOURCES_SUFFIX = ".v6_sources.json"
SOURCE_EVAL_SHA256_V1 = "dd65c35c149f1f4b7d10950c5930fb7a3ef5a2e506dec5425ea4ef9a5afcb755"
"""1 차 평가 설정(`CONFIG_REL`)의 바이트 — 2 차 설정은 anchor 의 이 바이트에서 `build_regime2_eval` 로 도출된 것이어야 한다(263 §2-3)."""
HISTORY_BASE_COMMIT = "06422e0890365a85ab8a0d40907089c0782d57b0"
"""보존 기준 커밋 — 이 커밋의 같은 상대 경로와 raw 가 같은 판 6 등록만 옛 `CONFIG_REL` 경로의 역사적 예외다(263 §2-1). 이후 발행은 옛 핀을
붙여 곁 파일을 생략할 수 없다."""
REGIME2_RECIPE = "json_line_then_basis"
AUXILIARY_V6: tuple[str, ...] = ("narrative_forbidden_rate", "narrative_reservation_rate", "citation_match_rate",
                                 "citation_precision", "ungrounded_rate")
"""D4 — 2 차 설정이 1 차 보조 이름 뒤에 **이 순서로 한 번씩** 붙이는 다섯 이름(263 §5). 누락 · 중복 · 치환은 `aux_names`."""
ALLOWED_CHANGES = frozenset({
    "generation.max_new_tokens", "generation.recipe", "generation.prompt_file", "generation.prompt_sha256",
    "generation.pairs_digest", "generation.registration_version", "auxiliary",
})
"""1 차 → 2 차 설정에서 바뀌어도 되는 칸 — 생성 여섯 칸과 보조 이름 목록뿐. 대표 지표 · 모집단 · 봉인 · 시드는 계승한다."""
_STEM = re.compile(r"^(main)-(\d{8})-([1-9][0-9]*)$")




class EvalRegistrationError(ValueError):
    """설정 파일을 읽을 수 없다(꼴)."""


class EvalPathRefused(EvalRegistrationError):
    """정식 등록 경로의 닫힌 꼴 밖 — 절대 경로 · 상위 이동 · 역슬래시 · 다른 디렉터리 · 잘못된 줄기 이름."""

    code = "eval_path"


class EvalBindingRefused(EvalRegistrationError):
    """anchor 의 평가 설정 결속 거부 — `code` 가 안정된 사유 코드, `problems` 가 설명 줄."""

    def __init__(self, code: str, problems):
        self.code = code
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__(f"[{code}] " + " · ".join(self.problems))


@dataclass(frozen=True)
class BoundEval:
    """anchor 커밋에서 읽은 평가 설정 — 경로 · 바이트 · sha · 판독한 문서 · anchor · 프로필(판 5 는 None)."""

    relative_path: str
    raw: bytes
    sha256: str
    doc: Mapping
    anchor_rev: str
    profile_id: str | None


def load_eval_registration(raw: bytes) -> dict:
    try:
        doc = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvalRegistrationError(f"평가 등록 설정을 읽지 못한다: {type(exc).__name__}") from None
    if not isinstance(doc, dict) or doc.get("kind") != KIND:
        raise EvalRegistrationError(f"평가 등록 설정의 kind 가 {KIND!r} 가 아니다")
    return doc


def load_base(raw: bytes) -> dict:
    """`configs/base.yaml` 의 바이트. 객체(사상)가 아니면 거부한다 — CLI 는 이 함수로만 읽는다."""
    try:
        doc = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvalRegistrationError(f"base 설정을 읽지 못한다: {type(exc).__name__}") from None
    if not isinstance(doc, dict):
        raise EvalRegistrationError(f"base 설정이 객체(사상)가 아니다({type(doc).__name__}) — 대조를 생략하지 않는다")
    return doc


def _get(doc: dict, path: str):
    cur = doc
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def registration_problems(doc: dict, *, base_doc: dict | None = None) -> list[str]:
    """설정의 값이 코드 상수 · base.yaml 과 같은가. 문제의 목록 — 비면 맞는 것이다.

    `base_doc=None` 은 base 와 맞대지 않는다는 **부르는 쪽의 선택**이다(CLI 는 늘 맞댄다). 값이 주어졌는데 객체가 아니거나
    `fixed_before_main_runs` 가 객체가 아니면 맞댈 수 없다는 문제다 — 맞은 것으로 넘기지 않는다.
    """
    from data.eval_sample import SEED as SAMPLE_SEED
    from data.eval_sample import SIZES as SAMPLE_SIZES
    from evaluation import sample_weights as SW
    from evaluation.adapters_v14 import COUPLING_RULE_V2
    from evaluation.conformance_v14 import SCORING_CODES
    from evaluation.metrics.box_f1 import SENSITIVITY_REGISTERED
    from evaluation.prereg_unified import VOCAB

    out: list[str] = []

    def want(path: str, expected, label: str) -> None:
        got = _get(doc, path)
        if got != expected:
            out.append(f"{path}: 설정 {got!r} ≠ {label} {expected!r}")

    want("representative.metric_id", SW.REPRESENTATIVE_METRIC, "sample_weights.REPRESENTATIVE_METRIC")
    want("representative.iou_threshold", IOU_THRESHOLD, "유리수 문턱")
    codes = _get(doc, "representative.class_codes")
    if not isinstance(codes, list) or sorted(map(str, codes)) != sorted(SCORING_CODES):
        out.append(f"representative.class_codes: 설정 {codes!r} ≠ conformance_v14.SCORING_CODES {sorted(SCORING_CODES)!r}")
    if _get(doc, "representative.macro_support_rule") not in VOCAB["metric.macro_support_rule"]:
        out.append(f"representative.macro_support_rule: 등록 어휘 {VOCAB['metric.macro_support_rule']} 밖이다")
    want("representative.coupling_rule", COUPLING_RULE_V2, "adapters_v14.COUPLING_RULE_V2")
    want("representative.sensitivity_variant", SENSITIVITY_REGISTERED, "box_f1.SENSITIVITY_REGISTERED")
    want("population.sample.seed", SAMPLE_SEED, "data.eval_sample.SEED")
    want("population.sample.weighting", SW.WEIGHTING_SAMPLE, "sample_weights.WEIGHTING_SAMPLE")
    want("population.sample.side_file_header", SW.SIDE_HEADER, "sample_weights.SIDE_HEADER")
    sealed = _get(doc, "population.sample.sealed")
    sizes = {str(s) for s in (*SAMPLE_SIZES, *EXTRA_SEALED_SIZES)}
    if not isinstance(sealed, dict) or set(sealed) != sizes:
        out.append(f"population.sample.sealed: 키 집합 {sorted(sealed) if isinstance(sealed, dict) else sealed!r} ≠ data.eval_sample.SIZES + "
                   f"EXTRA_SEALED_SIZES {sorted(sizes)}")
    else:
        for k, v in sealed.items():
            if not isinstance(v, dict) or not all(_HEX64.match(str(v.get(h, ""))) for h in ("file_sha256", "set_sha256", "side_sha256")) \
                    or not all(type(v.get(n)) is int and v[n] > 0 for n in ("images", "groups")):
                out.append(f"population.sample.sealed[{k}]: 장수 · 묶음 · 세 해시(16진 64자리)가 다 있어야 한다")
    main_size = _get(doc, "population.sample.main_size")
    if type(main_size) is not int or main_size not in (*SAMPLE_SIZES, *EXTRA_SEALED_SIZES):
        out.append(f"population.sample.main_size: {main_size!r} 가 봉인한 크기 {sorted((*SAMPLE_SIZES, *EXTRA_SEALED_SIZES))} 가운데 하나가 아니다")
    d_min = _get(doc, "population.sample.full_addendum_d_min")
    if not isinstance(d_min, (int, float)) or isinstance(d_min, bool) or not math.isfinite(d_min) or d_min <= 0:
        out.append("population.sample.full_addendum_d_min: 유한한 양수여야 한다")
    n_full = _get(doc, "population.n_full")
    if type(n_full) is not int or n_full < 1:
        out.append("population.n_full: 양의 정수여야 한다")
    seeds = doc.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 1 for s in seeds):
        out.append("seeds: 1 이상의 정수 목록이어야 한다")
    gen = _get(doc, "generation.max_new_tokens")
    if type(gen) is not int or gen < 1:
        out.append("generation.max_new_tokens: 양의 정수여야 한다")
    if base_doc is not None:
        if not isinstance(base_doc, dict):
            out.append(f"base 설정이 객체(사상)가 아니다({type(base_doc).__name__}) — base.yaml 과 맞대지 못했다")
            return out
        fixed = base_doc.get("fixed_before_main_runs")
        if not isinstance(fixed, dict):
            out.append("base 설정에 fixed_before_main_runs 객체가 없다 — base.yaml 과 맞대지 못했다")
            return out
        if _get(doc, "population.snapshot_digest") != fixed.get("snapshot_digest"):
            out.append("population.snapshot_digest: base.yaml fixed_before_main_runs.snapshot_digest 와 다르다")
        base_gen = fixed.get("uni_max_new_tokens")
        if base_gen is None:
            out.append("generation.max_new_tokens 를 base.yaml 과 맞댈 수 없다 — fixed_before_main_runs.uni_max_new_tokens 가 비어 있다")
        elif base_gen != gen:
            out.append(f"generation.max_new_tokens: 설정 {gen!r} ≠ base.yaml uni_max_new_tokens {base_gen!r}")
    return out


def check_population_binding(registration, doc: dict) -> list[str]:
    """본실험 등록 객체의 모집단 칸 — 층화 가중이면 봉인한 주 표본(`sealed[main_size]`)과 같아야 한다."""
    from evaluation import sample_weights as SW

    pop = registration.population
    if pop.weighting != SW.WEIGHTING_SAMPLE:
        return []
    sealed = (_get(doc, "population.sample.sealed") or {}).get(str(_get(doc, "population.sample.main_size")))
    if not isinstance(sealed, dict):
        return ["population.sample.sealed 에 main_size 의 항목이 없다"]
    out = []
    for field, key in (("eval_list_file_sha256", "file_sha256"), ("eval_list_set_sha256", "set_sha256"), ("n_images", "images")):
        if getattr(pop, field) != sealed.get(key):
            out.append(f"population.{field}: 등록 {getattr(pop, field)!r} ≠ 봉인한 주 표본의 {key} {sealed.get(key)!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(ROOT / CONFIG_REL))
    ap.add_argument("--base", default=str(ROOT / BASE_REL))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    f = Path(a.file)
    if not f.is_file():
        print(f"평가 등록 설정이 없다: {a.file} — 후보는 {CANDIDATE_REL}")
        return 1
    try:
        doc = load_eval_registration(f.read_bytes())
    except EvalRegistrationError as exc:
        print(str(exc))
        return 1
    b = Path(a.base)
    if not b.is_file():
        print(f"base 설정이 없다: {a.base} — 대조를 생략하지 않는다")
        return 1
    try:
        base = load_base(b.read_bytes())
    except EvalRegistrationError as exc:
        print(str(exc))
        return 1
    problems = registration_problems(doc, base_doc=base)
    for p in problems:
        print(p)
    print("맞다" if not problems else f"문제 {len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())


def strict_binding_problems(registration, doc: dict) -> list[str]:
    """본채점(엄격 · 표본 모드)의 결속 — `check_population_binding` 에 더해 시드 · 묶음 단위 · 묶음 수 · 대표 지표 칸을 맞댄다.

    표본 모드는 등록 객체와 평가 등록 설정 **두 파일**이 함께 정한다. 등록의 칸이 설정과 다르면 어느 쪽이 정본인지 진입점이 고르지
    않고 거부한다. 문제의 목록 — 비면 맞다.
    """
    from evaluation import sample_weights as SW
    from evaluation.sample_bootstrap import CLUSTER_UNIT

    out = list(check_population_binding(registration, doc))
    pop, ci, metric = registration.population, registration.ci, registration.metric
    if pop.weighting != SW.WEIGHTING_SAMPLE:
        out.append(f"population.weighting: 표본 모드는 {SW.WEIGHTING_SAMPLE!r} 다 — 등록 {pop.weighting!r}")
        return out
    sealed = (_get(doc, "population.sample.sealed") or {}).get(str(_get(doc, "population.sample.main_size"))) or {}
    if pop.n_clusters != sealed.get("groups"):
        out.append(f"population.n_clusters: 등록 {pop.n_clusters!r} ≠ 봉인한 주 표본의 groups {sealed.get('groups')!r}")
    unit = _get(doc, "population.sample.cluster_unit")
    if unit != CLUSTER_UNIT:
        out.append(f"population.sample.cluster_unit: 설정 {unit!r} ≠ 재표집이 실제로 쓰는 단위 {CLUSTER_UNIT!r}")
    for name, got in (("population.cluster_unit", pop.cluster_unit), ("ci.cluster_unit", ci.cluster_unit)):
        if got != unit:
            out.append(f"{name}: 등록 {got!r} ≠ 평가 등록 설정의 cluster_unit {unit!r} — 표본의 추첨 단위가 재표집 단위다")
    want_seeds = doc.get("seeds")
    got_seeds = sorted(int(k) for k in (registration.seeds or {}) if str(k).isdigit())
    if not isinstance(want_seeds, list) or sorted(want_seeds) != got_seeds:
        out.append(f"seeds: 등록의 시드 번호 {got_seeds} ≠ 평가 등록 설정의 seeds {want_seeds!r}")
    for cell, got, path in (("metric.metric_id", metric.metric_id, "representative.metric_id"),
                            ("metric.iou_threshold", metric.iou_threshold, "representative.iou_threshold"),
                            ("metric.coupling_rule", metric.coupling_rule, "representative.coupling_rule"),
                            ("metric.sensitivity_variant_id", metric.sensitivity_variant_id,
                             "representative.sensitivity_variant"),
                            ("metric.macro_support_rule", metric.macro_support_rule, "representative.macro_support_rule")):
        if got != _get(doc, path):
            out.append(f"{cell}: 등록 {got!r} ≠ 평가 등록 설정의 {path} {_get(doc, path)!r}")
    codes = _get(doc, "representative.class_codes")
    if sorted(map(str, metric.class_codes or ())) != sorted(map(str, codes or ())):
        out.append(f"metric.class_codes: 등록 {metric.class_codes!r} ≠ 평가 등록 설정의 class_codes {codes!r}")
    gen_limit = _get(doc, "generation.max_new_tokens")
    if registration.generation.max_new_tokens != gen_limit:
        out.append(f"generation.max_new_tokens: 등록 {registration.generation.max_new_tokens!r} ≠ 평가 등록 설정의 "
                   f"생성 한도 {gen_limit!r} — 등록한 한도와 다른 생성을 채점하지 않는다")
    return out


def narrative_layout_problems(registration, train_contract=None) -> list[str]:
    """판 6 등록의 서술 배치 칸이 학습 타깃 계약 v2 의 정본 문자열과 **같은가**(04 §7-5 ④ — `json_then_text` 폐기, 대응표 없음).

    `train_contract` 는 학습 쪽 계약(C 의 `layout` · `separator` 를 든 사전)이다. 주면 그 값이 정본이고, 이 모듈의 상수와도 같아야 한다 —
    두 쪽이 따로 적은 문자열이 갈리면 그것부터 문제다. 판 5 등록에는 서술 블록이 없으므로 볼 것이 없다. 문제의 목록 — 비면 맞다.
    """
    from evaluation.prereg_unified import (
        NARRATIVE_LAYOUT,
        NARRATIVE_SEPARATOR,
        REGISTRATION_VERSION_V6,
    )

    if registration.version != REGISTRATION_VERSION_V6:
        return []
    n = registration.narrative
    if n is None:
        return ["narrative: 판 6 등록인데 서술 블록이 없다"]
    want = {"layout": NARRATIVE_LAYOUT, "separator": NARRATIVE_SEPARATOR}
    out: list[str] = []
    if train_contract is not None:
        for key, mine in want.items():
            theirs = train_contract.get(key)
            if theirs != mine:
                out.append(f"학습 계약의 {key} {theirs!r} ≠ 평가 쪽 정본 {mine!r} — 두 쪽 문자열이 갈렸다")
    for key, mine in want.items():
        if getattr(n, key) != mine:
            out.append(f"narrative.{key}: 등록 {getattr(n, key)!r} ≠ 계약 v2 의 {mine!r}")
    return out


# ════════════════════════════════════════════════════════════════════ 2 차 설정의 결정적 곁 파일(263 §2)

def registration_stem(registration_rel: str) -> str:
    """정식 등록 경로 `configs/registration/{purpose}-{YYYYMMDD}-{n}.json` 의 줄기 — 닫힌 꼴 밖은 `EvalPathRefused`(`eval_path`).

    목적은 `main` 뿐이고, 날짜는 달력 날짜, n 은 선행 0 없는 양의 정수다. `candidates/` · 학습 설정 원문 · 영수증 · 출처 파일은 등록이 아니다.
    """
    p = registration_rel
    if not isinstance(p, str) or not p or "\\" in p or p.startswith("/") or re.match(r"^[A-Za-z]:", p) \
            or any(part in ("", ".", "..") for part in p.split("/")):
        raise EvalPathRefused("저장소 상대 posix 경로가 아니다(절대 경로 · 상위 이동 · 역슬래시 불가)")
    pp = PurePosixPath(p)
    if pp.parent.as_posix() != REG_DIR:
        raise EvalPathRefused(f"{REG_DIR} 바로 아래의 등록이 아니다 — candidates 는 완결 등록의 곁 파일 자리가 아니다")
    if pp.suffix != ".json" or pp.name.endswith((".train_config.json", ".receipt.json", SIDECAR_SOURCES_SUFFIX)):
        raise EvalPathRefused("등록 파일(`{stem}.json`)이 아니다")
    m = _STEM.match(pp.stem)
    if m is None:
        raise EvalPathRefused("이름이 main-YYYYMMDD-n 꼴이 아니다")
    d = m.group(2)
    try:
        datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    except ValueError:
        raise EvalPathRefused("날짜가 달력 날짜가 아니다") from None
    return pp.stem


def eval_sidecar_rel(registration_rel: str) -> str:
    """등록과 같은 커밋 · 같은 줄기의 결정적 평가 설정 경로 — 자유로운 설정 경로 인자를 받지 않는다(D2)."""
    return f"{REG_DIR}/{registration_stem(registration_rel)}{SIDECAR_EVAL_SUFFIX}"


def v6_sources_rel(registration_rel: str) -> str:
    """같은 줄기의 D3 출처 운반 파일 경로."""
    return f"{REG_DIR}/{registration_stem(registration_rel)}{SIDECAR_SOURCES_SUFFIX}"


def changed_fields(old: dict, new: dict, prefix: str = "") -> list[str]:
    """값의 변화와 칸 추가 · 삭제를 함께 센다. 목록은 순서까지 비교한다."""
    result = []
    for key in sorted(old.keys() | new.keys()):
        path = f"{prefix}.{key}" if prefix else key
        if key not in old or key not in new:
            result.append(path)
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            result.extend(changed_fields(old[key], new[key], path))
        elif type(old[key]) is not type(new[key]) or old[key] != new[key]:
            result.append(path)
    return result


def build_regime2_eval(source_doc: dict, *, base_doc: dict, train_doc: Mapping, prompt_raw: bytes) -> dict:
    """1 차 평가 설정에서 2 차 설정을 **도출**한다 — 순수 함수(263 §2-3). 후보 작성(`scripts/prepare_eval_regime2.py`)과 네 소비자의
    대조(`regime2_eval_problems`)가 같은 함수를 쓴다.

    계승: 대표 지표 · 모집단 · 봉인 표본 · 시드 · 기존 보조 이름. 채움: 생성 여섯 칸(학습 등록의 레시피 · 한도, base 의 프롬프트 자리 ·
    페어 digest, 프롬프트 바이트의 sha, 등록 판 6)과 보조 다섯 이름(`AUXILIARY_V6` — 기존 이름 뒤에 그 순서로). 입력이 2 차 체제가 아니거나
    허용 범위 밖이 바뀌면 `EvalRegistrationError`.
    """
    from evaluation.prereg_unified import REGISTRATION_VERSION_V6
    from vlm.registration import REGIME_PROMPTS, regime_problems

    problems = registration_problems(source_doc)
    if source_doc.get("kind") != KIND or source_doc.get("version") != 1:
        problems.append("1차 평가 설정 스키마가 아니다")
    fixed = base_doc.get("fixed_before_main_runs", {}) if isinstance(base_doc, Mapping) else {}
    fixed = fixed if isinstance(fixed, Mapping) else {}
    recipe = train_doc.get("recipe") if isinstance(train_doc, Mapping) else None
    limit = train_doc.get("eval_gen_limit") if isinstance(train_doc, Mapping) else None
    prompt = fixed.get("uni_prompt_path")
    pairs = (fixed.get("uni_pairs") or {}).get("digest") if isinstance(fixed.get("uni_pairs"), Mapping) else None
    prompt_sha = hashlib.sha256(bytes(prompt_raw)).hexdigest()
    if recipe != REGIME2_RECIPE:
        problems.append("2차 레시피가 아니다")
    if prompt != REGIME_PROMPTS.get(recipe):
        problems.append("학습 등록 레시피와 base 프롬프트 경로가 다르다")
    if limit != fixed.get("uni_max_new_tokens") or type(limit) is not int:
        problems.append("학습 등록 한도와 base 한도가 다르다")
    problems.extend(regime_problems(recipe, prompt_sha256=prompt_sha, gen_limit=limit))
    if not isinstance(pairs, str) or not _HEX64.match(pairs):
        problems.append("base의 봉인 페어 digest가 없다")
    aux = source_doc.get("auxiliary")
    if not isinstance(aux, list) or any(name in aux for name in AUXILIARY_V6):
        problems.append(f"aux_names: 1차 설정의 auxiliary 가 목록이 아니거나 2차 이름 {list(AUXILIARY_V6)} 을 이미 담고 있다")
    if problems:
        raise EvalRegistrationError("; ".join(problems))
    candidate = copy.deepcopy(source_doc)
    candidate["generation"].update({
        "max_new_tokens": limit, "recipe": recipe, "prompt_file": prompt,
        "prompt_sha256": prompt_sha, "pairs_digest": pairs,
        "registration_version": REGISTRATION_VERSION_V6,
    })
    candidate["auxiliary"] = [*aux, *AUXILIARY_V6]
    problems = registration_problems(candidate, base_doc=base_doc)
    changes = set(changed_fields(source_doc, candidate))
    if changes - ALLOWED_CHANGES:
        problems.append(f"허용 범위 밖 변경: {sorted(changes - ALLOWED_CHANGES)}")
    if problems:
        raise EvalRegistrationError("; ".join(problems))
    return candidate


def regime2_eval_problems(doc, *, source_doc: dict, base_doc: dict, train_doc: Mapping, prompt_raw: bytes) -> list[str]:
    """2 차 설정 `doc` 이 1 차 설정 · base · 학습 등록 · 프롬프트에서 도출한 설정과 같은가 — 후보 작성과 네 소비자가 공유하는 순수 대조.

    줄은 `eval_binding: …`(도출 입력의 결속 · 칸의 차이) 또는 `aux_names: …`(보조 이름 목록). 비면 맞다. M1 규약을 맞춰서 구제하지 않는다.
    """
    try:
        expected = build_regime2_eval(source_doc, base_doc=base_doc, train_doc=train_doc, prompt_raw=prompt_raw)
    except (EvalRegistrationError, KeyError, TypeError, AttributeError) as exc:
        return [f"eval_binding: 입력 결속 — {exc}"]
    if not isinstance(doc, dict):
        return ["eval_binding: 2차 설정이 객체(사상)가 아니다"]
    out: list[str] = []
    if doc.get("auxiliary") != expected["auxiliary"]:
        out.append(f"aux_names: 보조 이름 목록이 1차 여섯 뒤에 2차 다섯({', '.join(AUXILIARY_V6)})을 그 순서로 붙인 정본이 아니다 — "
                   f"받은 {doc.get('auxiliary')!r}")
    for path in changed_fields(expected, doc):
        if path == "auxiliary":
            continue
        out.append(f"eval_binding: 2차 설정이 1차 설정 · anchor 의 base · 학습 등록 · 프롬프트에서 도출한 설정과 다르다: {path}")
    return out


def _history_blob(repo: Path, history_rev: str, rel: str) -> bytes | None:
    from evaluation.entry_gate import EntryRejected, blob_at

    try:
        return blob_at(repo, history_rev, rel)
    except EntryRejected:
        return None


def load_bound_eval(*, repo: Path, anchor_rev: str, registration_rel: str, registration,
                    history_rev: str | None = None) -> BoundEval:
    """네 소비자의 공통 판독(263 §2-2) — anchor 커밋의 blob 만 읽어 등록과 맞댄 평가 설정을 돌려준다. 거부는 `EvalBindingRefused`
    (`code` — `eval_path` · `eval_sidecar_missing` · `eval_binding` · `eval_bytes` · `aux_names` · `profile_mismatch` · `v6_sources_*`).

    1. 등록을 anchor 의 blob 으로 다시 읽어 넘긴 객체의 정규 바이트와 맞댄다 — 메모리 객체로 검사를 대신하지 않는다.
    2. 판 5, 또는 보존 기준 커밋(`history_rev` — 비면 `HISTORY_BASE_COMMIT`, 합성 시험은 임시 저장소의 커밋을 준다)의 같은 상대 경로와
       raw 가 같은 역사적 판 6 → 옛 `CONFIG_REL`(그 anchor 의 것)과 기존 결속.
       그 밖의 판 6 → `profile_for`(같은 anchor 의 학습 등록) 가 `regime2_v4` 여야 하고(옛 핀을 붙인 새 발행은 거부) 결정적 곁 파일
       `eval_sidecar_rel` 만 읽는다 — 없으면 `eval_sidecar_missing`(HEAD · 작업 트리 · candidates · CLI 로 대체하지 않는다).
    3. 신규는 같은 anchor 의 1 차 설정(바이트가 `SOURCE_EVAL_SHA256_V1`) · base · 학습 등록 · 프롬프트에서 도출한 설정과 맞대고
       (`regime2_eval_problems`), 코드 상수 · base(`registration_problems`) · 등록 결속(`strict_binding_problems` — 시드 · 표본 · 대표
       지표 · 한도)을 지나며, 출처 운반 파일(`load_v6_sources`)이 등록 · 설정 바이트와 아홉 원본을 묶었는지 본다.
    """
    from evaluation import narrative_registration as NR
    from evaluation.entry_gate import EntryRejected, blob_at
    from evaluation.prereg_unified import REGISTRATION_VERSION_V6, dump_registration
    from vlm.registration import RegistrationRejected, registration_from_bytes

    root = Path(repo)
    stem_rel = registration_rel
    try:
        registration_stem(stem_rel)                                        # 닫힌 경로 꼴(`eval_path`) — 읽기 전에
    except EvalPathRefused as exc:
        raise EvalBindingRefused(EvalPathRefused.code, [str(exc)]) from None

    def read(rel: str, what: str) -> bytes:
        try:
            return blob_at(root, anchor_rev, rel)
        except EntryRejected as exc:
            raise EvalBindingRefused("eval_binding", [f"anchor 에 {what} {rel} 이 없다({exc.code})"]) from None

    reg_raw = read(stem_rel, "등록")
    if dump_registration(registration) != reg_raw:
        raise EvalBindingRefused("eval_binding", ["넘긴 등록 객체의 정규 바이트가 anchor 의 등록 바이트와 다르다"])
    profile = None
    legacy = registration.version != REGISTRATION_VERSION_V6
    if not legacy:
        history_rev = history_rev or HISTORY_BASE_COMMIT
        hist = _history_blob(root, history_rev, stem_rel)
        if hist is not None and hist == reg_raw:
            profile, legacy = NR.NARRATIVE_PROFILES[NR.PROFILE_LEGACY], True
        else:
            try:
                train_entry = registration_from_bytes(read(TRAIN_REG_REL, "학습 등록"))
            except RegistrationRejected as exc:
                raise EvalBindingRefused("eval_binding", [f"anchor 의 학습 등록을 읽지 못했다 — {exc}"]) from None
            try:
                profile = NR.profile_for(registration, train_entry=train_entry)
            except NR.ProfileRefused as exc:
                raise EvalBindingRefused(exc.code, exc.problems) from None
            if profile.profile_id != NR.PROFILE_REGIME2:
                raise EvalBindingRefused("profile_mismatch", [(f"보존 기준 {history_rev[:8]} 의 등록 바이트가 아닌 판 6 발행은 "
                                                               f"{NR.PROFILE_REGIME2} 여야 한다 — 옛 핀을 붙인 새 발행은 곁 파일 없이 서지 않는다")])
    base_doc = load_base(read(BASE_REL, "base 설정"))
    if legacy:
        rel = CONFIG_REL
        raw = read(rel, "평가 등록 설정")
        try:
            doc = load_eval_registration(raw)
        except EvalRegistrationError as exc:
            raise EvalBindingRefused("eval_binding", [str(exc)]) from None
        problems = registration_problems(doc, base_doc=base_doc)
        if not problems:                    # 꼴이 맞는 설정에서만 결속을 본다 — 틀린 꼴이 결속 검사에서 예외로 죽지 않게
            problems = strict_binding_problems(registration, doc)
        if problems:
            raise EvalBindingRefused("eval_binding", problems)
        return BoundEval(rel, raw, hashlib.sha256(raw).hexdigest(), doc, anchor_rev, None if profile is None else profile.profile_id)
    rel = eval_sidecar_rel(stem_rel)
    try:
        raw = blob_at(root, anchor_rev, rel)
    except EntryRejected:
        raise EvalBindingRefused("eval_sidecar_missing", [(f"anchor 에 결정적 평가 설정 {rel} 이 없다 — HEAD · 작업 트리 · candidates · "
                                                           f"명령줄 설정으로 대신하지 않는다")]) from None
    try:
        doc = load_eval_registration(raw)
    except EvalRegistrationError as exc:
        raise EvalBindingRefused("eval_binding", [str(exc)]) from None
    source_raw = read(CONFIG_REL, "1 차 평가 설정")
    if hashlib.sha256(source_raw).hexdigest() != SOURCE_EVAL_SHA256_V1:
        raise EvalBindingRefused("eval_bytes", ["anchor 의 1 차 평가 설정 바이트가 보존 지문(SOURCE_EVAL_SHA256_V1)과 다르다"])
    try:
        source_doc = load_eval_registration(source_raw)
        train_doc = registration_from_bytes(read(TRAIN_REG_REL, "학습 등록"))
    except (EvalRegistrationError, RegistrationRejected) as exc:
        raise EvalBindingRefused("eval_binding", [str(exc)]) from None
    fixed = base_doc.get("fixed_before_main_runs") if isinstance(base_doc.get("fixed_before_main_runs"), Mapping) else {}
    prompt_rel = fixed.get("uni_prompt_path")
    if not isinstance(prompt_rel, str) or not prompt_rel:
        raise EvalBindingRefused("eval_binding", ["anchor 의 base 설정에 uni_prompt_path 가 없다"])
    prompt_raw = read(prompt_rel, "프롬프트")
    problems = [f"eval_binding: {p}" for p in registration_problems(doc, base_doc=base_doc)]
    if not problems:                        # 꼴이 맞는 설정에서만 결속 · 도출 대조를 본다
        problems += [f"eval_binding: {p}" for p in strict_binding_problems(registration, doc)]
        problems += regime2_eval_problems(doc, source_doc=source_doc, base_doc=base_doc, train_doc=train_doc, prompt_raw=prompt_raw)
    if problems:
        raise EvalBindingRefused(problems[0].split(":", 1)[0], problems)
    try:
        NR.load_v6_sources(repo=root, anchor_rev=anchor_rev, registration_rel=stem_rel, registration_raw=reg_raw, eval_raw=raw)
    except NR.ProfileRefused as exc:
        raise EvalBindingRefused(exc.code, exc.problems) from None
    return BoundEval(rel, raw, hashlib.sha256(raw).hexdigest(), doc, anchor_rev, profile.profile_id)
