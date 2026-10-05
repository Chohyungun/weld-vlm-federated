"""곁 파일의 **실제 값** ↔ 등록의 생성 쪽 값 — 쓰는 쪽의 시작 전 점검과 읽는 쪽의 묶음 검증이 **같은 함수**를 부른다.
진입점 미니스펙 3판 1-8 · 07번 §14-1 자 · §13-3 다-11 · §30-4.

| 규칙 | 무엇 |
|---|---|
| 항목 | `registered_values(등록, mode)` 가 낸 사전의 키마다 곁 파일에 같은 키가 있어야 한다 |
| 자료형 | 자료형까지 같아야 같다 — `256` 과 `"256"` 은 다르다, 참·거짓은 수가 아니다. **사전 · 목록의 안쪽까지** 같은 규칙이다(`{"do_sample": 0}` 과 `False` 는 다르다). 등록의 튜플은 JSON 의 목록과 같은 것으로 본다 |
| 출발점 두 필드 | 둘 다 싣는다 — `null` 도 같음으로 본다 |
| 목록 해시 | 모델 묶음은 등록의 생성 목록 해시, 에코 묶음은 등록의 **에코 목록** 해시와 맞댄다(§31-3) |
| 목적마다 비어도 되는 칸 | `plan_sha256` 은 등록에 값이 있을 때만 곁 파일과 맞댄다(§31-5). `train_rows_digest` 는 **곁 파일에 없다** — 어댑터 meta 와 맞댄다(리허설 2판 반영판 §2-5 · §2-7, §34) |
| 구현 식별자 | 곁 파일의 `impl_ids` 는 쓰는 쪽의 꼴 그대로다 — 식별자 사전(`seam` · `approved` · `module` · `qualname` · `source_path` · `in_repo` · `blob_sha1` · `head_blob_sha1`)의 목록, 또는 이음새 이름 → 식별자 사전. 등록은 식별자마다 열쇠 `"{seam}={module}:{qualname}@{source_path}"` 를 적고, 열쇠의 집합을 맞댄다. **승인된 실제 구현**인지(`approved` · `in_repo` · 커밋된 코드)는 `impl_problems` 가 따로 본다 |
| 싣는 것 | 어긋난 **항목 이름**과 두 자료형의 이름. **값은 싣지 않는다** |

생성 쪽 칸마다 어디서 맞대는지(또는 왜 맞대지 않는지)를 `GENERATION_ROUTE` 가 든다 — 칸이 늘면 시험이 떨어진다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from evaluation.prereg_unified import START_FIELDS, UnifiedRegistration

SIDE_EQUAL_FIELDS: tuple[str, ...] = (
    "purpose", "list_split", "snapshot_digest",
    "prompt_sha256", "chat_template_kwargs", "gen_prefix_sha256", "max_new_tokens", "decoding",
    "batch_size", "padding_side", "processor_config_sha256", "processor_min_pixels", "processor_max_pixels",
    "patch_size", "merge_size", "coord_space", "coord_cfg_hash", "base_model_id", "base_model_revision",
    "train_config_sha256", "budget_n", "budget_r", "budget_e", "transformers_version", "impl_ids",
)
"""등록의 생성 쪽 칸 가운데 **곁 파일에 같은 이름으로 실리는 것**. 목록 해시와 출발점은 아래에서 따로 넣는다.
`impl_ids` 는 이름이 같지만 꼴이 다르다 — 열쇠의 집합으로 맞댄다(`impl_keys`)."""
OPTIONAL_WHEN_SET: tuple[str, ...] = ("plan_sha256",)
"""등록에 값이 있을 때만 곁 파일과 맞대는 칸."""
ADAPTER_META_FIELDS: tuple[str, ...] = ("train_rows_digest",)
"""곁 파일이 아니라 **어댑터 meta** 와 맞대는 칸(등록에 값이 있을 때, 모델 묶음만). 묶음 검증의 학습 단계가 본다."""

GENERATION_ROUTE: dict[str, str] = {
    **{k: "곁 파일 · 같은 이름" for k in SIDE_EQUAL_FIELDS},
    **{k: "곁 파일 · 출발점(null 끼리도 같다)" for k in START_FIELDS},
    "eval_list_file_sha256": "곁 파일 · 모델 묶음의 목록 해시", "eval_list_set_sha256": "곁 파일 · 모델 묶음의 목록 해시",
    "echo_list_file_sha256": "곁 파일 · 에코 묶음의 목록 해시(`eval_list_*` 이름으로)",
    "echo_list_set_sha256": "곁 파일 · 에코 묶음의 목록 해시(`eval_list_*` 이름으로)",
    "plan_sha256": "곁 파일 · 등록에 값이 있을 때",
    "train_rows_digest": "어댑터 meta · 등록에 값이 있을 때(모델 묶음)",
    "frame_diag_rules_sha256": "맞대지 않는다 — 진단 규칙은 생성 지문 안에서 고정되고 진단 판정기가 규칙 파일을 읽는다. 곁 파일에 없다",
    "frame_diag_rules_path": "맞대지 않는다 — 위와 같다",
}
"""`GenerationSpec` 의 칸 → 맞대는 자리. 칸이 생기면 여기 없어서 시험이 떨어진다(검수 16번 M-2)."""

IMPL_FIELDS: tuple[str, ...] = ("seam", "approved", "module", "qualname", "source_path", "in_repo",
                                "blob_sha1", "head_blob_sha1")
"""구현 식별자 사전의 열쇠 — 쓰는 쪽 `vlm.seams.impl_id` 의 꼴(리허설 2판 반영판 §1-4)."""


@dataclass(frozen=True)
class ActualMismatch:
    """어긋난 항목 하나. 값은 싣지 않는다."""

    item: str
    reason: str
    registered_type: str | None = None
    actual_type: str | None = None
    equal: bool | None = None
    """자료형을 빼고 보면 같은가 — `256` 과 `"256"` 처럼 **자료형만** 다른 경우를 가른다. 값은 아니다."""

    def as_detail(self) -> dict:
        out = {"item": self.item, "reason": self.reason}
        if self.registered_type is not None:
            out |= {"registered_type": self.registered_type, "sidecar_type": self.actual_type,
                    "equal": self.equal}
        return out


def registered_values(registration: UnifiedRegistration, *, mode: str) -> dict[str, object]:
    """곁 파일 키 → 등록값. `mode` 는 `model` · `echo` 다."""
    if mode not in ("model", "echo"):
        raise ValueError(f"모드는 model · echo 가운데 하나다 — 받은 값 {mode!r}")
    g = registration.generation
    out: dict[str, object] = {k: getattr(g, k) for k in SIDE_EQUAL_FIELDS}
    for k in START_FIELDS:
        out[k] = getattr(g, k)
    src = ("echo_list_file_sha256", "echo_list_set_sha256") if mode == "echo" \
        else ("eval_list_file_sha256", "eval_list_set_sha256")
    out["eval_list_file_sha256"], out["eval_list_set_sha256"] = getattr(g, src[0]), getattr(g, src[1])
    for k in OPTIONAL_WHEN_SET:
        if getattr(g, k) is not None:
            out[k] = getattr(g, k)
    return out


def _same(want, got) -> bool:
    """자료형까지, **안쪽까지** 같은가. 튜플과 목록은 같은 것으로 본다."""
    if isinstance(want, (tuple, list)) and isinstance(got, (tuple, list)):
        return len(want) == len(got) and all(_same(a, b) for a, b in zip(want, got, strict=True))
    if isinstance(want, Mapping) and isinstance(got, Mapping):
        return set(want) == set(got) and all(_same(want[k], got[k]) for k in want)
    return type(want) is type(got) and want == got


def _hex40_or_none(v) -> bool:
    return v is None or (isinstance(v, str) and len(v) == 40 and all(c in "0123456789abcdef" for c in v))


def impl_identifiers(value) -> list[dict] | None:
    """곁 파일 · 어댑터 meta 의 `impl_ids` → 이음새 이름 순의 식별자 사전 목록. 꼴이 아니면 `None`.

    받는 꼴 둘 — 식별자 사전의 목록(어댑터 meta), 이음새 이름 → 식별자 사전(export). 이음새는 겹치지 않는다.
    """
    if isinstance(value, Mapping):
        if not all(isinstance(v, Mapping) and v.get("seam") == k for k, v in value.items()):
            return None
        items = list(value.values())
    elif isinstance(value, list):
        items = value
    else:
        return None
    if not items:
        return None
    for d in items:
        if not (isinstance(d, Mapping) and set(d) == set(IMPL_FIELDS)
                and isinstance(d["seam"], str) and d["seam"]
                and type(d["approved"]) is bool and type(d["in_repo"]) is bool
                and all(d[k] is None or (isinstance(d[k], str) and d[k]) for k in ("module", "qualname", "source_path"))
                and _hex40_or_none(d["blob_sha1"]) and _hex40_or_none(d["head_blob_sha1"])):
            return None
    seams = [d["seam"] for d in items]
    if len(set(seams)) != len(seams):
        return None
    return sorted((dict(d) for d in items), key=lambda d: d["seam"])


def impl_key(d: Mapping) -> str:
    """식별자 하나의 등록 열쇠 — `"{seam}={module}:{qualname}@{source_path}"`."""
    return f"{d['seam']}={d['module']}:{d['qualname']}@{d['source_path']}"


def impl_keys(value) -> list[str] | None:
    ids = impl_identifiers(value)
    return None if ids is None else sorted(impl_key(d) for d in ids)


def impl_problems(value) -> list[str]:
    """**승인된 실제 구현**이 아닌 까닭. 빈 목록이면 승인된 구현이다 — 승인 표시 · 저장소 안 · 커밋된 코드(작업 트리 blob = HEAD blob)."""
    ids = impl_identifiers(value)
    if ids is None:
        return ["꼴이 아니다"]
    out = []
    for d in ids:
        if not d["approved"]:
            out.append(f"{d['seam']}: 승인 목록 밖의 구현")
        if not d["in_repo"]:
            out.append(f"{d['seam']}: 저장소 밖의 구현")
        if d["blob_sha1"] is None or d["blob_sha1"] != d["head_blob_sha1"]:
            out.append(f"{d['seam']}: 커밋되지 않은 코드")
    return out


def compare_values(registered: Mapping[str, object], actual: Mapping[str, object]) -> list[ActualMismatch]:
    """항목별로, 자료형까지. 어긋난 것을 **전부** 돌려준다(없으면 빈 목록)."""
    out: list[ActualMismatch] = []
    for key, want in registered.items():
        if key not in actual:
            out.append(ActualMismatch(key, "곁 파일에 없다"))
            continue
        got = actual[key]
        if key == "impl_ids":
            keys = impl_keys(got)
            if keys is None or not isinstance(want, (tuple, list)) or keys != sorted(want):
                out.append(ActualMismatch(key, "구현 식별자의 열쇠가 등록과 다르다" if keys is not None
                                          else "구현 식별자의 꼴이 아니다"))
            continue
        if not _same(want, got):
            out.append(ActualMismatch(key, "등록값과 다르다", type(want).__name__, type(got).__name__,
                                      bool(got == want)))
    return out


def check_actuals(registration: UnifiedRegistration, actual: Mapping[str, object], *,
                  mode: str) -> list[ActualMismatch]:
    """등록의 생성 쪽 값과 곁 파일(또는 쓰는 쪽이 모은 실제 값)을 맞댄다."""
    return compare_values(registered_values(registration, mode=mode), actual)
