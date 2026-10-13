"""리허설 · 진단의 계획 파일을 읽는다 — 리허설 2판 §4-4 의 키 표와 규칙.

계획 파일(`configs/rehearsal_uni.yaml` · `configs/frame_diag_uni.yaml`)은 **기본 설정에 병합하지 않는다.** 진입점이 `--plan <경로>` 로만
읽는다. 원시 바이트의 sha256 이 `plan_sha256` 으로 등록 · 어댑터 meta · 곁 파일에 실린다.

| 규칙 | 거부 사유 코드 |
|---|---|
| YAML 객체가 아니다 · `kind` 가 목적과 다르다 · `version` 이 1 이 아니다 | `plan_form` · `plan_kind` · `plan_version` |
| 모르는 키 · 그 목적에 두지 않는 키 | `plan_unknown_key` · `plan_key_not_for_purpose` |
| 본실험 조건을 바꾸는 이름(모델 · 프롬프트 · 템플릿 · 좌표 · 디코딩 · 시드 값 · 스냅샷 · 페어 지문 · 예산) | `plan_forbidden_key` |
| 학습량 `amount` 가 `{n, r, e}` 정수가 아니거나 `n ≠ r × e` | `plan_amount` |
| 값의 꼴(분할 · 칸 · 중단 문자열 · 구판 넷) | `plan_value` |

**`null` 은 아직 없는 값이다**(계획 파일 머리말). 그 값이 필요한 단계는 `require` 로 부르고 `null` 이면 `plan_value_missing` 으로 시작하지 않는다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["FORBIDDEN_SEGMENTS", "NEGATIVES", "PLAN_KEYS", "Plan", "PlanRejected", "load_plan", "parse_plan"]

#: (잎 키, rehearsal 에 둔다, frame_diag 에 둔다) — 2판 §4-4 의 표. 사전이 값인 잎(`train.rows` · `amount` · `faults.*` ·
#: `timeouts_s` · `optional` · `exclude.near_dup_edges`)은 그 아래를 따로 본다.
PLAN_KEYS: tuple[tuple[str, bool, bool], ...] = (
    ("kind", True, True), ("version", True, True), ("sampling_seed", True, True), ("seed_index", True, True),
    ("cells", True, True),
    ("gen.split", True, True), ("gen.defect_only", True, True), ("gen.target_n", True, False),
    ("gen.target_boxes", False, True), ("gen.max_n", True, True), ("gen.min_groups_per_stratum", True, True),
    ("gen.max_target_tokens", True, True),
    ("echo.same_as_gen", True, True),
    ("train.split", True, True), ("train.rows", True, True), ("train.avoid_multiple_of_accum", True, True),
    ("train.central_rows", True, True), ("train.fed_rows", True, False),
    ("amount", True, True),
    ("exclude.near_dup_edges", True, True), ("exclude.drop_components_touching_eval", True, True),
    ("gen_limit.max_new_tokens", True, True), ("gen_limit.provisional", True, True),
    ("faults.train", True, False), ("faults.export", True, False),
    ("negatives", True, False), ("timeouts_s", True, True), ("optional", True, False),
    ("attempts_allowed", False, True),
)
#: 계획에 있으면 안 되는 마디 이름 — 본실험 조건을 계획이 바꾸는 길(2판 §4-4 의 "계획에 두지 않는 것").
FORBIDDEN_SEGMENTS = frozenset({
    "model", "adapter", "lora", "optimizer", "lr", "preprocess", "prompt", "template", "chat_template_kwargs",
    "coord_space", "decoding", "seed", "seeds", "experiment", "budget", "train_budget", "uni_train_budget",
    "fixed_before_main_runs", "snapshot_digest", "pairs_sha256", "pairs_digest", "expect",
})
TAGS = ("uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_central", "uni_fed")
NEGATIVES = ("policy_blank", "identity_key_missing", "identity_key_extra", "format_mismatch")
CLIENTS = ("C1", "C2", "C3")


class PlanRejected(ValueError):
    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


@dataclass(frozen=True)
class Plan:
    purpose: str
    data: Mapping[str, Any]
    raw: bytes
    sha256: str
    path: Path | None = None

    def get(self, dotted: str, default: Any = None) -> Any:
        d: Any = self.data
        for part in dotted.split("."):
            if not isinstance(d, Mapping) or part not in d:
                return default
            d = d[part]
        return d

    def require(self, dotted: str) -> Any:
        """값이 있어야 시작하는 키 — `null` 이거나 없으면 거부한다(계획 파일의 `null` 은 아직 없는 값이다)."""
        v = self.get(dotted)
        if v is None:
            raise PlanRejected("plan_value_missing", f"계획의 {dotted} 가 비어 있다(null) — 그 값이 필요한 단계는 시작하지 않는다")
        return v

    @property
    def amount(self) -> tuple[int, int, int]:
        a = self.data["amount"]
        return int(a["n"]), int(a["r"]), int(a["e"])


def _leaves(d: Mapping, known: set[str], prefix: str = "") -> list[str]:
    out = []
    for k, v in d.items():
        p = f"{prefix}{k}"
        if p in known:
            out.append(p)
        elif isinstance(v, Mapping) and any(x.startswith(p + ".") for x in known):
            out.extend(_leaves(v, known, p + "."))
        else:
            out.append(p)
    return out


def _all_segments(d: Any) -> set[str]:
    if isinstance(d, Mapping):
        return {str(k) for k in d} | {s for v in d.values() for s in _all_segments(v)}
    if isinstance(d, list):
        return {s for v in d for s in _all_segments(v)}
    return set()


def _int(v: Any, *, low: int = 0) -> bool:
    return type(v) is int and v >= low


def parse_plan(raw: bytes, *, purpose: str, path: Path | None = None) -> Plan:
    """바이트 → `Plan`. 규칙은 머리말의 표다."""
    import yaml

    from vlm.fault import parse_fault

    if purpose not in ("rehearsal", "frame_diag"):
        raise PlanRejected("plan_purpose", f"계획 파일은 rehearsal · frame_diag 만 받는다: {purpose!r}")
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PlanRejected("plan_form", f"계획 파일을 읽을 수 없다: {exc}") from None
    if not isinstance(data, Mapping):
        raise PlanRejected("plan_form", "계획 파일이 YAML 객체가 아니다")
    if data.get("kind") != purpose:
        raise PlanRejected("plan_kind", f"계획의 kind {data.get('kind')!r} 가 목적 {purpose!r} 와 다르다")
    if data.get("version") != 1:
        raise PlanRejected("plan_version", f"계획 꼴의 판이 1 이 아니다: {data.get('version')!r}")
    forbidden = sorted(_all_segments(data) & FORBIDDEN_SEGMENTS - {"seed_index", "sampling_seed"})
    if forbidden:
        raise PlanRejected("plan_forbidden_key", f"계획에 본실험 조건의 키가 있다: {forbidden}")
    col = 1 if purpose == "rehearsal" else 2
    allowed = {k for k, *flags in PLAN_KEYS if flags[col - 1]}
    known = {k for k, *_ in PLAN_KEYS}
    leaves = _leaves(data, known)
    unknown = sorted(p for p in leaves if p not in known)
    if unknown:
        raise PlanRejected("plan_unknown_key", f"모르는 키: {unknown}")
    misplaced = sorted(p for p in leaves if p not in allowed)
    if misplaced:
        raise PlanRejected("plan_key_not_for_purpose", f"{purpose} 계획에 두지 않는 키: {misplaced}")
    plan = Plan(purpose=purpose, data=data, raw=raw, sha256=hashlib.sha256(raw).hexdigest(), path=path)

    a = data.get("amount")
    if not isinstance(a, Mapping) or set(a) != {"n", "r", "e"} or not all(_int(a[k], low=1) for k in a) \
            or a["n"] != a["r"] * a["e"]:
        raise PlanRejected("plan_amount", f"학습량 amount 는 n == r × e 인 정수 셋이다: {a!r}")
    bad: list[str] = []
    if not _int(data.get("seed_index"), low=1):
        bad.append("seed_index")
    if not _int(data.get("sampling_seed")):
        bad.append("sampling_seed")
    cells = data.get("cells")
    if not isinstance(cells, list) or not cells or not set(cells) <= set(TAGS) or len(set(cells)) != len(cells):
        bad.append("cells")
    if plan.get("gen.split") != "val":
        bad.append("gen.split")
    if plan.get("train.split") != "train":
        bad.append("train.split")
    rows = plan.get("train.rows")
    if purpose == "rehearsal" and (not isinstance(rows, Mapping) or set(rows) != set(CLIENTS)
                                   or not all(_int(v, low=1) for v in rows.values())):
        bad.append("train.rows")
    if purpose == "rehearsal":
        if not _int(plan.get("gen.target_n"), low=1):
            bad.append("gen.target_n")
        if plan.get("train.central_rows") != "union" or plan.get("train.fed_rows") != "same_as_local":
            bad.append("train.central_rows · train.fed_rows")
    if purpose == "frame_diag":                                  # 진단 — 중앙 한 칸 · 중앙 풀에서 바로(총괄 결정 20) · 박스 수 목표
        if not isinstance(rows, Mapping) or set(rows) != {"central"} or not _int(rows.get("central"), low=1):
            bad.append("train.rows")
        if plan.get("train.central_rows") != "direct":
            bad.append("train.central_rows")
        if cells != ["uni_central"]:
            bad.append("cells")
        if not _int(plan.get("gen.target_boxes"), low=1):
            bad.append("gen.target_boxes")
        if not _int(data.get("attempts_allowed"), low=1):
            bad.append("attempts_allowed")
    if not _int(plan.get("gen.max_n"), low=1) or not _int(plan.get("gen.min_groups_per_stratum"), low=0):
        bad.append("gen.max_n · gen.min_groups_per_stratum")
    if plan.get("echo.same_as_gen") is not True:
        bad.append("echo.same_as_gen")
    edges = plan.get("exclude.near_dup_edges")
    if edges is not None and (not isinstance(edges, Mapping) or set(edges) != {"path", "sha256"}):
        bad.append("exclude.near_dup_edges")
    if not _int(plan.get("gen_limit.max_new_tokens"), low=1):
        bad.append("gen_limit.max_new_tokens")
    neg = data.get("negatives")
    if neg is not None and (not isinstance(neg, list) or sorted(neg) != sorted(NEGATIVES)):
        bad.append("negatives")
    for stage, by_tag in (data.get("faults") or {}).items():
        if stage not in ("train", "export") or not isinstance(by_tag, Mapping):
            bad.append(f"faults.{stage}")
            continue
        for tag, v in by_tag.items():
            if tag not in TAGS or tag == "uni_fed":          # 연합 칸은 중단을 넣지 않는다(2판 §1-1 — 서버 쪽 재개가 없다)
                bad.append(f"faults.{stage}.{tag}")
                continue
            items = [v] if stage == "train" else (v if isinstance(v, list) else [None])
            for it in items:
                raw_f = it if stage == "train" else (it or {}).get("fault") if isinstance(it, Mapping) else None
                if stage == "export" and isinstance(it, Mapping) and set(it) in ({"stop_after_lines"}, {"to_end"}):
                    continue
                try:
                    fs = parse_fault(raw_f) if isinstance(raw_f, str) else None
                except Exception:  # noqa: BLE001 - 꼴이 틀린 문자열은 아래에서 사유로 모은다
                    fs = None
                if fs is None or fs.stage != stage or fs.tag != tag:
                    bad.append(f"faults.{stage}.{tag}")
    if bad:
        raise PlanRejected("plan_value", f"값의 꼴이 틀리다: {sorted(set(bad))}")
    return plan


def load_plan(path: str | Path, *, purpose: str) -> Plan:
    p = Path(path)
    return parse_plan(p.read_bytes(), purpose=purpose, path=p)
