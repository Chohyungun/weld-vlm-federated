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
| `population.sample.seed` · `sealed` 의 키 집합 | `data.eval_sample.SEED` · `SIZES`(표본 추출기) |
| `population.snapshot_digest` · `generation.max_new_tokens` | `configs/base.yaml` 의 `fixed_before_main_runs` |

`check_population_binding` 은 본실험 등록 객체(`UnifiedRegistration`)의 모집단 칸이 `stratified_g_over_n` 이면 그 목록 해시 · 장수가
`sealed[main_size]` 와 같아야 한다고 본다 — 봉인한 목록이 아닌 것으로 본채점이 돌지 않게.

    python -X utf8 -B -m evaluation.eval_registration [--file configs/eval_registration.yaml] [--base configs/base.yaml]

문제가 하나라도 있으면 줄마다 적고 종료 코드 1. 설정 파일이 없으면 그것도 문제다 — "없음" 을 "맞음" 으로 읽지 않는다.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_REL = "configs/eval_registration.yaml"
BASE_REL = "configs/base.yaml"
CANDIDATE_REL = "docs/dev_log/2026-10-07-본실험코퍼스/28_eval_registration_후보_D.yaml"
"""평가 쪽이 값을 산출 파일에서 채운 후보 — 데이터 쪽이 `CONFIG_REL` 로 옮긴다(바이트 그대로)."""
KIND = "eval_registration"
IOU_THRESHOLD = "1/2"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EvalRegistrationError(ValueError):
    """설정 파일을 읽을 수 없다(꼴)."""


def load_eval_registration(raw: bytes) -> dict:
    try:
        doc = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EvalRegistrationError(f"평가 등록 설정을 읽지 못한다: {type(exc).__name__}") from None
    if not isinstance(doc, dict) or doc.get("kind") != KIND:
        raise EvalRegistrationError(f"평가 등록 설정의 kind 가 {KIND!r} 가 아니다")
    return doc


def _get(doc: dict, path: str):
    cur = doc
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def registration_problems(doc: dict, *, base_doc: dict | None = None) -> list[str]:
    """설정의 값이 코드 상수 · base.yaml 과 같은가. 문제의 목록 — 비면 맞는 것이다."""
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
    sizes = {str(s) for s in SAMPLE_SIZES}
    if not isinstance(sealed, dict) or set(sealed) != sizes:
        out.append(f"population.sample.sealed: 키 집합 {sorted(sealed) if isinstance(sealed, dict) else sealed!r} ≠ data.eval_sample.SIZES {sorted(sizes)}")
    else:
        for k, v in sealed.items():
            if not isinstance(v, dict) or not all(_HEX64.match(str(v.get(h, ""))) for h in ("file_sha256", "set_sha256", "side_sha256")) \
                    or not all(type(v.get(n)) is int and v[n] > 0 for n in ("images", "groups")):
                out.append(f"population.sample.sealed[{k}]: 장수 · 묶음 · 세 해시(16진 64자리)가 다 있어야 한다")
    main_size = _get(doc, "population.sample.main_size")
    if type(main_size) is not int or main_size not in SAMPLE_SIZES:
        out.append(f"population.sample.main_size: {main_size!r} 가 봉인한 크기 {list(SAMPLE_SIZES)} 가운데 하나가 아니다")
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
        fixed = base_doc.get("fixed_before_main_runs") or {}
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
    base = yaml.safe_load(Path(a.base).read_text(encoding="utf-8")) if Path(a.base).is_file() else None
    problems = registration_problems(doc, base_doc=base)
    for p in problems:
        print(p)
    print("맞다" if not problems else f"문제 {len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
