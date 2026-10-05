"""통합형 본실험의 **등록 블록** — 값을 보기 전에 정한 것. 07번 미니스펙 §4 · §12-5 · §16-5 · §18-6 · §29.

`evaluation/prereg.py` 는 고치지 않는다. 그 파일의 상수와 `RECOVERY_CI_REGISTRATION` 은 사전실험 v3 산출물이
기대는 값이고, 09-16 에 "값을 보기 전 제안, 3세트 채점 뒤 등록" 으로 지위까지 적어 둔 기록이다. 새 블록은 여기 따로 선다.

## 지문이 둘인 이유

| 지문 | 덮는 것 | 누가 묶이나 |
|---|---|---|
| `generation_sha256` | 목적과 기준 분할 · 동결본 · 목록 · 프롬프트와 템플릿 · 디코딩 · 모델과 출발점 · 학습 설정 · 좌표 규약 · 라이브러리 판 | **내보내기**와 채점 |
| `scoring_sha256` | 지표 정의 · 결합 규칙 · 대조선 · CI 매개변수와 규칙 · 모집단 설계 · 기지답 벡터 | 채점·집계 |

나누지 않으면 **채점 규칙을 한 글자 고칠 때마다 재추론**을 요구하게 된다. 결과를 이미 본 뒤의 재추론은 증거가 되지 못하고
(greedy 는 같은 답을 낸다) 곁 파일을 손으로 고칠 유인만 만든다. 채점 등록의 개정은 허용하되 기록을 남긴다.

## 빈 값은 등록이 아니다 — 그리고 값이 있다고 등록인 것도 아니다

`missing()` 은 **`None` 만** 빈 값으로 본다. `False` 와 `0` 은 값이다 — "보정하지 않는다(False)" 는 결정이지 미정이 아니다.
예외는 **출발점 두 필드**다. 배경지식 단계를 유지하면 `start_checkpoint_sha256`, 폐지하면 `init_adapter_digest` 에
값이 들고 **나머지는 null** 이다(07번 §13-3 마-13). 둘 다 비면 둘 다 빈 항목이고, 하나만 차면 둘 다 빈 항목이 아니다.

`invalid()` 는 채워진 값이 **그 칸에 올 수 있는 값인가**를 본다 — 자료형(`256` 과 `"256"` 은 다르다, 참·거짓은 수가 아니다),
해시의 꼴(소문자 64자리), 허용 어휘, 범위, 칸 사이의 관계. 앞 판은 존재만 보아 정수 칸에 문자열을 넣은 등록이 완결로 지났다.
`require_complete()` 는 둘 다 본다.

## 영수증

"등록이 먼저였다" 를 커밋 시각으로 증명하지 않는다. 이 저장소는 이력을 두 번 다시 썼다. 대신 **내용 해시**에 건다 —
등록이 본줄기에 병합된 다음 커밋으로 영수증을 적고, 내보내기와 본채점이 그 영수증이 가리키는 값으로만 돈다.
영수증은 **종류**(`main` · `rehearsal` · `frame_diag`)와 **등록 파일의 경로 · 그 바이트의 sha256** 을 싣는다.
등록 객체는 작업 트리가 아니라 영수증의 커밋에서 그 경로의 바이트를 읽어 만든다(`load_registration`) —
읽는 일(`git show`)은 호출부가 한다. 바이트의 해시가 영수증과 다르면 등록을 만들지 않는다(`registration_for_receipt`).

## 목적 셋 — 목적마다 요구하는 칸이 다르다 (07번 §31-5 · 진입점 미니스펙 3판 8-4)

| 목적 | 기준 분할 | 생성 쪽 | 채점 쪽 |
|---|---|---|---|
| `main` | `eval` | 전부. 진단 규칙 두 칸 · `train_rows_digest` · `plan_sha256` 은 **null** 이어야 한다 | 전부 |
| `rehearsal` | `val` | 전부. 진단 규칙 두 칸은 null. `train_rows_digest` · `plan_sha256` 은 null 이어도 된다 | 에코 관문 두 칸(`require_probe_scoring`) |
| `frame_diag` | `val` | 전부 — 진단 규칙 두 칸 · `train_rows_digest` · `plan_sha256` 까지 | 에코 관문 두 칸 |

진단 규칙은 **생성 쪽**에 있다(§31-1) — 채점 등록을 개정해도 진단 규칙은 바뀌지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import get_type_hints

from evaluation.recovery_interval import REGISTERED_CI as _REGISTERED_CI

REGISTRATION_VERSION = "4"
"""판 4(2026-10-01): 구간 규약의 채택(의사결정로그 10-01 · 결정표 11행). `ci` 의 규약 칸은 **채택값만** 받는다 —
`CI_ADOPTED`. 칸 여덟을 더했다(끝점 · 반폭의 상태 어휘 · 동률 규칙 · 분자 0 의 판정 · `n` 의 정의 · 점추정의 분모 실패 ·
적용 범위 · 규약 id). 07번 §33.
판 3(2026-10-01): 목적 `frame_diag` · 생성 쪽 칸 일곱(에코 목록 해시 둘 · `impl_ids` · `train_rows_digest` ·
`plan_sha256` · 진단 규칙의 경로와 digest) · 채점 쪽 칸 둘(`canary.coord_fixture_sha256` · `canary.frame_diag_sha256`) ·
목적별 완결 규칙 · 영수증의 `registration_file_sha256`. 07번 §31 · §32.
판 2 까지는 영수증을 낸 적이 없다 — 지문이 쓰이기 전에 꼴을 바꿨다."""


class RegistrationIncomplete(RuntimeError):
    """값이 비어 있다. **빈 값으로는 채점하지 않는다** — 어느 항목인지 이름을 들고 있다.

    다른 거부와 같은 `[코드]` 꼴로 말한다 — 쓰는 쪽과 읽는 쪽의 거부를 코드로 맞대게(결정 04 의 4절 13)."""

    code = "REGISTRATION_INCOMPLETE"

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(f"[{self.code}] 등록이 비어 있다: " + ", ".join(self.missing))


class RegistrationInvalid(RuntimeError):
    """값이 있으나 그 칸에 올 수 없다. **항목 이름과 사유를 전부** 들고 있다. `[코드]` 꼴로 말한다(결정 04 의 4절 13)."""

    code = "REGISTRATION_INVALID"

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(f"[{self.code}] 등록 값이 유효하지 않다: " + " · ".join(self.problems))


class ReceiptMissing(RuntimeError):
    """영수증이 없거나, 꼴이 틀리거나, 지문이 어긋난다."""


@dataclass
class MetricSpec:
    """대표 지표의 정의. 여기 있는 것이 바뀌면 채점 지문이 바뀐다."""

    metric_id: str | None = None
    iou_threshold: str | None = None
    """`"1/2"` 처럼 **유리수 문자열**로 적는다 — 0.5 를 부동소수로 적으면 경계 규칙과 어긋난다."""
    matching_rule_id: str | None = None
    class_codes: tuple[str, ...] | None = None
    label_map_version: str | None = None
    coupling_rule: str | None = None
    sensitivity_variant_id: str | None = None
    discard_policy_id: str | None = None
    macro_support_rule: str | None = None
    """평균 대상을 정하는 규칙. 등록값은 `gold_support`(정답 지지량으로 정한다)."""


@dataclass
class BaselineSpec:
    """대조선. **규칙만이 아니라 그 규칙이 낸 예측과 값까지** 박는다 — 구현이 바뀌면 드러나게."""

    rule_id: str | None = None
    strata_k: int | None = None
    threshold_t: float | None = None
    fit_split: str | None = None
    """`"train_then_refit_trainval"` — train 으로 적합하고 val 로 t 를 고른 뒤 **train+val 로 재적합**한다(§15-7)."""
    prediction_sha256: str | None = None
    full_population_m1: float | None = None
    companion_rule_ids: tuple[str, ...] | None = None


@dataclass
class CiSpec:
    """CI 와 판정 규칙.

    **구간 규약 칸은 2026-10-01 채택값만 받는다**(판 4 · `CI_ADOPTED`). 값은 방식 이름이 아니라 식과 어휘다 —
    값의 정본은 `evaluation.recovery_interval.REGISTERED_CI` 이고, 그 모듈이 등록의 칸을 읽어 같지 않으면 채점하지 않는다.
    판 3 까지는 칸이 비어 있었다(고르기 전). `min_defined_draws` 는 채택 행이 수를 정하지 않았다 — 여전히 등록이 정한다.
    """

    n_resamples: int | None = None
    rng_seed: int | None = None
    alpha: float | None = None
    interval: str | None = None
    cluster_unit: str | None = None
    """`"component"` — 묶음이 아니라 **연결 성분**이다(§18-4)."""
    rule1_sd_definitions: tuple[str, ...] | None = None
    rule1_multiplier_by_seed_count: dict[str, int] | None = None
    """시드 수 → 배수. 3시드는 3, **2시드는 6**(§15-8)."""
    rule2_max_half_width: float | None = None
    rule3_denominator_excludes_zero: bool | None = None
    rule4_max_undefined_fraction: float | None = None
    rule4_counts: tuple[str, ...] | None = None
    """규칙 ④ 가 세는 사유. `denominator` 는 빼고 `support`·`indeterminate` 만(§15-6)."""
    rule5_statistic: str | None = None
    undefined_denominator_policy: str | None = None
    """분모가 0 이하인 추첨의 처리(규약안 축 A). 판 4 는 **채택값만** 받는다(`CI_ADOPTED`). 판 3 의 선택지 셋
    `UNDEFINED_DENOMINATOR_POLICIES` 는 고르기 전의 기록이고 검사에 쓰지 않는다(07번 §33-7)."""
    percentile_method: str | None = None
    """백분위 방식(규약안 축 C). 판 4 는 **채택값만** 받는다 — 방식 이름이 아니라 순위 식(`CI_ADOPTED`).
    판 3 의 선택지 `PERCENTILE_METHODS` 는 기록이다."""
    infinity_handling: str | None = None
    """±무한을 백분위에서 다루는 법(규약안 축 B). 판 4 는 **채택값만** 받는다(`CI_ADOPTED`). 판 3 의 `INFINITY_HANDLINGS` 는 기록이다."""
    interval_absence_representation: str | None = None
    """끝점·반폭이 수가 아닐 때의 표현(규약안 축 D). 판 4 는 **채택값만** 받는다(`CI_ADOPTED`). 판 3 의 `ABSENCE_REPRESENTATIONS` 는 기록이다."""
    min_defined_draws: int | None = None
    """유효 추첨 수의 하한(§22-3). 규칙 ⑤ 의 등록 판정이 이 값과 `n_resamples` 를 **둘 다** 요구한다."""
    endpoint_states: tuple[str, ...] | None = None
    """끝점의 상태 어휘 — `finite` · `pos_inf` · `neg_inf` · `absent`(판 4)."""
    half_width_states: tuple[str, ...] | None = None
    """반폭의 상태 어휘 — `finite` · `infinite` · `undefined`(같은 부호의 무한끼리) · `absent`(판 4)."""
    numerator_zero_rule: str | None = None
    """동률 — 분모 실패 추첨에서 분자가 0 이면 판정 불가(판 4). §15-6 의 `F − L̄ ≥ 0 이면 +∞` 를 대신한다(§33)."""
    numerator_zero_test: str | None = None
    """"분자가 0" 의 판정 — 부동소수 정확 비교(판 4)."""
    percentile_n_definition: str | None = None
    """백분위의 `n` — 집계에 들어가는 추첨 수(유한과 ±무한)(판 4)."""
    point_denominator_failure: str | None = None
    """점추정의 분모 실패 — 내지 않는다(`absent` 와 사유)(판 4)."""
    interval_scope: str | None = None
    """이 규칙을 쓰는 구간 — 통합형 산출물에 실리는 모든 백분위 구간(판 4)."""
    convention_id: str | None = None
    """구간마다 싣는 규약 id(판 4)."""


@dataclass
class PopulationSpec:
    """평가 모집단. **근사 중복 제외가 여기를 흔든다**(§18)."""

    population_id: str | None = None
    exclusion_list_id: str | None = None
    exclusion_list_sha256: str | None = None
    duplicate_pairs_sha256: str | None = None
    cluster_unit: str | None = None
    eval_list_file_sha256: str | None = None
    eval_list_set_sha256: str | None = None
    """**채점 모집단**의 목록 해시. 생성 쪽 목록과 같거나 그 부분집합이다(§13-2 가) — `check_list_binding`."""
    n_images: int | None = None
    n_clusters: int | None = None
    weighting: str | None = None
    """`"none"` 이 주 보고다(§18-2). 가중 추정값은 병기이고 그 설계는 `subsample_design` 에 적는다."""
    subsample_design: dict | None = None
    class_support: dict | None = None
    totals_error_table: dict | None = None
    """목록별 총계 오차표. 총계 보정을 하지 않는 대신 그 크기를 공개하는 조건이다(§16-3)."""


@dataclass
class GenerationSpec:
    """내보내기가 묶이는 것. **시드 목록은 여기 없다** — 채점 쪽이다(§13-2 가)."""

    purpose: str | None = None
    """`main`(본실험) · `rehearsal`(리허설) · `frame_diag`(착수 전 프레임 진단). 영수증의 종류와 같아야 한다."""
    list_split: str | None = None
    """생성 목록이 뽑힌 기준 분할. `main` 은 `eval`, 나머지 둘은 `val` 이다 — 셋이 한 줄로 서야 한다."""
    snapshot_digest: str | None = None
    """생성 목록의 기준이 된 동결본의 digest. 목록이 우연히 같아도 목적·분할·동결본이 다르면 지문이 갈린다.
    승인된 동결본인지는 여기서 보지 않는다 — 진입점이 `configs/base.yaml` 의 닻과 맞댄다(§32-3)."""
    eval_list_file_sha256: str | None = None
    eval_list_set_sha256: str | None = None
    echo_list_file_sha256: str | None = None
    """에코 목록의 파일 해시. 세 목적 모두 필수다 — 본실험도 본 export 전에 에코를 한 번 돈다(§31-3)."""
    echo_list_set_sha256: str | None = None
    prompt_sha256: str | None = None
    chat_template_kwargs: dict | None = None
    gen_prefix_sha256: str | None = None
    max_new_tokens: int | None = None
    decoding: dict | None = None
    batch_size: int | None = None
    padding_side: str | None = None
    processor_config_sha256: str | None = None
    processor_min_pixels: int | None = None
    processor_max_pixels: int | None = None
    patch_size: int | None = None
    merge_size: int | None = None
    coord_space: str | None = None
    coord_cfg_hash: str | None = None
    base_model_id: str | None = None
    base_model_revision: str | None = None
    start_checkpoint_sha256: str | None = None
    """배경지식 단계를 **유지**할 때의 출발 체크포인트. 그때 `init_adapter_digest` 는 null 이다."""
    init_adapter_digest: str | None = None
    """배경지식 단계를 **폐지**할 때의 초기 어댑터 다이제스트. 그때 `start_checkpoint_sha256` 은 null 이다.
    꼴(해시인가 수의 목록인가)은 쓰는 쪽이 정해 알린다 — 지금은 비지 않은 문자열인지만 본다."""
    train_config_sha256: str | None = None
    budget_n: int | None = None
    budget_r: int | None = None
    budget_e: int | None = None
    transformers_version: str | None = None
    impl_ids: tuple[str, ...] | None = None
    """승인된 실제 구현의 **열쇠** 목록 — 식별자마다 `"{seam}={module}:{qualname}@{source_path}"`(`actuals.impl_key`).
    곁 파일의 `impl_ids`(쓰는 쪽의 식별자 사전, 리허설 2판 반영판 §1-4)에서 같은 열쇠를 내 집합으로 맞댄다 — 대역 실행을 가른다
    (§30-4 · §31-5 · §34)."""
    train_rows_digest: str | None = None
    """이 등록으로 실제로 학습한 행의 `image_id` 수열의 sha256. `frame_diag` 필수 · `rehearsal` 선택 · `main` null(§31-5)."""
    plan_sha256: str | None = None
    """계획 파일의 원시 바이트 sha256. 목적별 규칙은 `train_rows_digest` 와 같다."""
    frame_diag_rules_sha256: str | None = None
    """진단 규칙 파일 바이트의 sha256. `frame_diag` 에서만 값이다 — 규칙을 바꾸면 생성 지문이 바뀐다(§31-1)."""
    frame_diag_rules_path: str | None = None
    """진단 규칙 파일의 저장소 상대 경로. digest 옆에 둔다 — digest 로 폴더를 훑지 않는다(§32-9)."""


@dataclass
class CanarySpec:
    """좌표 카나리아. **규약이 바뀌면 다시 본다**(§16-5)."""

    echo_criterion: str | None = None
    echo_max_coord_diff: float | None = None
    """규약이 왕복 무손실이면 0.0. 손실이 있으면 등록한 허용오차."""
    axis_ratio_threshold: float | None = None
    axis_ratio_min_pairs: int | None = None
    literal_match_tolerance: float | None = None
    conformance_vector_version: str | None = None
    conformance_vector_sha256: str | None = None
    known_answer_sha256: str | None = None
    """기지답 벡터(고정 픽스처의 계수표 sha). 본채점이 먼저 재현한다. **에코 관문의 칸이 아니다** —
    그 자리는 `coord_fixture_sha256` 이다(§31-6 이 한 칸의 두 뜻을 갈랐다)."""
    coord_cfg: CoordCfgRecord | None = None
    """좌표 설정 해시. **하나가 아니라 목록이다** — 아래 설명을 본다."""
    coord_fixture_sha256: str | None = None
    """골든 픽스처 지문(`coord_fixture_digest` 의 값). 에코 관문이 이 칸을 본다(§31-6)."""
    frame_diag_sha256: str | None = None
    """통과한 진단 산출물 파일의 sha256. 본실험 등록에만 쓴다 — 진단 통과의 효력이 이 칸으로 선다(§31-7 · §32-1)."""


@dataclass
class CoordCfgRecord:
    """좌표 설정 해시가 **이미 갈린** 것을 병기로 등록한다.

    동결 당시 값과 지금 값이 다르다. 이 저장소의 규칙은 **재고정이 아니라 병기**다 —
    조용히 다시 못 박으면 근거 사슬이 끊긴 사실 자체가 사라진다.

    네 가지를 함께 든다.

    | | 무엇 |
    |---|---|
    | 가 | 동결 당시 값과 그 날짜 |
    | 나 | 변경 뒤 값 |
    | 다 | 바뀐 것이 **주석뿐**이라는 근거 |
    | 라 | AST 와 골든 픽스처로 **동작이 같음**을 확인한 사실 |

    **동결 export 는 옛 값을 기록한다.** 그 파일들을 고치지 않는다 — 그것이 당시의 사실이다.
    그래서 관문은 "등록된 값 **가운데 하나**와 맞는가" 로 본다(`accepted`).
    """

    accepted: tuple[str, ...]
    """받아들이는 값 전부. 동결 당시 값과 현재 값이 **둘 다** 들어간다."""
    frozen_value: str
    frozen_at: str
    """동결 시각. 날짜가 없으면 어느 쪽이 먼저인지 알 수 없다."""
    current_value: str
    change_scope: str
    """무엇이 바뀌었나. `comments_only` 가 아니면 병기로 끝낼 사안이 아니다."""
    equivalence_evidence: tuple[str, ...]
    """동작 동일성의 근거. AST 동일성과 골든 픽스처 지문 둘 다 든다."""
    frozen_exports_keep_old: bool = True
    """동결 export 가 옛 값을 기록한다는 사실. 고치지 않는다."""

    def __post_init__(self) -> None:
        if not self.accepted:
            raise ValueError("받아들일 값이 비어 있다")
        for v in (self.frozen_value, self.current_value):
            if v not in self.accepted:
                raise ValueError(f"{v!r} 가 accepted 에 없다 — 병기는 둘을 다 담는 것이다")
        if not self.equivalence_evidence:
            raise ValueError("동작 동일성 근거가 없다 — 병기의 조건이다")

    def as_dict(self) -> dict:
        return {"accepted": list(self.accepted),
                "frozen_value": self.frozen_value, "frozen_at": self.frozen_at,
                "current_value": self.current_value,
                "change_scope": self.change_scope,
                "equivalence_evidence": list(self.equivalence_evidence),
                "frozen_exports_keep_old": self.frozen_exports_keep_old,
                "note": ("재고정하지 않고 병기한다. 동결 export 는 옛 값을 기록하므로 "
                         "관문은 등록된 값 가운데 하나와 맞는지로 본다")}


class EchoGateFailed(RuntimeError):
    """에코 관문이 막았다. **어느 쪽이 안 맞았는지** 들고 있다."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("에코 관문 불통과: " + " · ".join(self.reasons))


def check_echo_gate(observed_cfg_hash: str | None, observed_fixture_digest: str | None,
                    canary: CanarySpec) -> None:
    """에코 묶음의 좌표 관문. **둘 다 맞아야 통과한다.**

    ① 관측한 `coord_cfg_hash` 가 등록된 값 가운데 하나와 같다.
    ② 골든 픽스처 지문이 등록한 `canary.coord_fixture_sha256` 과 같다. 07번 §31-6 이 `known_answer_sha256` 의 두 뜻을
       갈랐다 — 그 칸은 M1 의 기지답 벡터로 돌아갔고 에코 관문은 이 칸을 본다.

    한쪽만 맞는 것을 통과시키면 관문이 사라진다. 해시만 보면 설정이 같아도 동작이 갈린 경우를
    놓치고, 픽스처만 보면 다른 설정으로 돈 산출물이 들어온다. 그래서 논리곱이다.

    Raises:
        EchoGateFailed: 둘 중 하나라도 어긋나면. 사유를 **전부** 모아서 낸다.
        ValueError: 등록이 비어 있으면 — 관문을 열어 두지 않는다.
    """
    if canary.coord_cfg is None or canary.coord_fixture_sha256 is None:
        raise ValueError("에코 관문을 걸 등록이 없다 — coord_cfg 와 coord_fixture_sha256 이 있어야 한다")
    reasons: list[str] = []
    if observed_cfg_hash not in canary.coord_cfg.accepted:
        reasons.append(f"coord_cfg_hash {observed_cfg_hash!r} 가 등록된 값 "
                       f"{list(canary.coord_cfg.accepted)} 가운데 없다")
    if observed_fixture_digest != canary.coord_fixture_sha256:
        reasons.append("골든 픽스처 지문이 등록한 coord_fixture_sha256 과 다르다")
    if reasons:
        raise EchoGateFailed(reasons)


DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "metric.matching_rule_id": ("metric.iou_threshold", "canary.known_answer_sha256",
                                "canary.conformance_vector_sha256", "baseline.full_population_m1"),
    "generation.coord_space": ("canary.echo_criterion", "canary.echo_max_coord_diff",
                               "canary.literal_match_tolerance", "canary.axis_ratio_threshold"),
    "generation.coord_cfg_hash": ("canary.echo_criterion",),
    "generation.processor_config_sha256": ("canary.echo_criterion", "canary.known_answer_sha256"),
    "generation.patch_size": ("canary.echo_criterion",),
    "generation.merge_size": ("canary.echo_criterion",),
    "generation.prompt_sha256": ("generation.gen_prefix_sha256", "generation.max_new_tokens"),
    "generation.chat_template_kwargs": ("generation.gen_prefix_sha256", "generation.max_new_tokens"),
    "population.exclusion_list_sha256": ("population.eval_list_file_sha256",
                                         "population.eval_list_set_sha256",
                                         "population.n_images", "population.n_clusters",
                                         "baseline.full_population_m1", "population.class_support",
                                         "population.totals_error_table"),
    "population.duplicate_pairs_sha256": ("population.n_clusters",),
    "population.population_id": ("population.exclusion_list_id", "population.exclusion_list_sha256",
                                 "population.eval_list_file_sha256", "population.eval_list_set_sha256",
                                 "population.n_images", "population.n_clusters",
                                 "population.cluster_unit", "population.class_support",
                                 "population.totals_error_table", "baseline.prediction_sha256",
                                 "baseline.full_population_m1", "population.subsample_design"),
}
"""**무엇이 바뀌면 무엇을 다시 봐야 하는가.** 왼쪽을 고치고 오른쪽을 그대로 두면 등록이 스스로와 어긋난다.

**이름은 영역까지 적는다**(`population.eval_list_file_sha256`). 같은 끝 이름의 칸이 생성 쪽과 채점 쪽에 한 벌씩
있어서, 끝 이름만 적으면 어느 쪽을 다시 봐야 하는지 모른다(앞 판이 그랬다).

좌표 규약이 바뀌면 왕복 무손실이 아닐 수 있어 에코 기준이 "전량 TP" 로 설 수 없다 — 기준과 허용오차를 규약별로 다시 등록한다.
제외 목록이 바뀌면 모집단에 묶인 상수가 전부 다시 나와야 한다(§18-7).

**매칭 규약은 수 규약까지 포함한다.** `matching_rule_id` 는 후보 판정식만이 아니라 그 판정이
어느 수 위에서 이뤄지는지(`box_f1.COORD_NUMERIC_CONVENTION`, 현재 `binary64`)까지 가리킨다.
십진으로 바꾸면 같은 판정식이어도 **다른 규약**이므로 규칙 id 를 함께 바꾼다 — 그래야 기지답
픽스처와 대조선 값이 딸려 나온다. 값이 하나뿐인 지금은 걸릴 일이 없지만, 걸릴 날에 같은 지문으로
통과하는 것을 막는 자리다.

모집단 자체가 갈리는 경우(결정표 3c 의 10-01 시한 규칙 — 재분할한 새 eval 이냐 현행 전량이냐)는
`population_id` 가 출발점이다. **대조선까지 끌고 간다** — 다만 무엇이 바뀌는지는 두 경우가 다르다.

- **평가 모집단만 바뀐다**(기존 목록의 부분집합): 대조선은 **다시 채점**만 한다. 적합은 그대로다 —
  적합에는 **등록한 train/val 만** 쓴다 — eval 은 **대조선의 적합·임계 선택에 쓰지 않는다**
  (정해진 eval 을 채점하는 단계까지 막는 말이 아니다). 바뀌는 것은
  `full_population_m1`(그 모집단에서의 대조선 값)이고 `prediction_sha256` 은 대조선이 낸 예측이라
  모집단이 부분집합이면 그대로일 수 있다.
- **train/val 까지 바뀐다**(재분할): 대조선을 **새 train/val 로 다시 적합**한다. 그러면
  `prediction_sha256` 과 `full_population_m1` 이 **둘 다** 바뀐다. 이 경우는 학습 입력이 바뀐 것이라
  어댑터·예측·생성 지문도 함께 갈린다 — **다른 실행**이다(07번 §19-2).
"""

FRAME_DEFINING_FIELDS: tuple[str, ...] = (
    "prompt_sha256", "chat_template_kwargs", "gen_prefix_sha256", "processor_config_sha256",
    "processor_min_pixels", "processor_max_pixels", "patch_size", "merge_size", "coord_space",
    "coord_cfg_hash", "base_model_id", "base_model_revision", "init_adapter_digest", "transformers_version",
)
"""프레임을 정하는 생성 쪽 칸 — 쓰는 쪽이 적은 목록 그대로다(07번 §31-7). 하나라도 바뀌면 진단이 본 프레임은
본실험의 프레임이 아니므로 `canary.frame_diag_sha256` 을 다시 정해야 한다. `train_config` 의 나머지는 등록의 칸이 아니다."""
for _name in FRAME_DEFINING_FIELDS:
    _key = f"generation.{_name}"
    DEPENDENCIES[_key] = DEPENDENCIES.get(_key, ()) + ("canary.frame_diag_sha256",)
del _name, _key


# ── 허용 어휘 ───────────────────────────────────────────────────────────────
# 자유 문자열이던 칸 가운데 **계약이 선택지를 적은 것**만 어휘를 둔다. 어휘는 고른 값이 아니라 고를 수 있는 값이다.

PURPOSES = ("main", "rehearsal", "frame_diag")
SPLIT_OF_PURPOSE = {"main": "eval", "rehearsal": "val", "frame_diag": "val"}
"""목적 → 기준 분할. 영수증의 종류 · 등록의 목적 · 목록의 분할 셋이 한 줄로 서지 않으면 어느 경로든 거부한다."""
RECEIPT_KINDS = PURPOSES

PURPOSE_NULL: dict[str, frozenset[str]] = {
    "main": frozenset({"generation.frame_diag_rules_sha256", "generation.frame_diag_rules_path",
                       "generation.train_rows_digest", "generation.plan_sha256"}),
    "rehearsal": frozenset({"generation.frame_diag_rules_sha256", "generation.frame_diag_rules_path"}),
    "frame_diag": frozenset(),
}
"""목적마다 **null 이어야 하는** 생성 쪽 칸. 값이 있으면 무효다. 본실험의 학습 행은 칸마다 달라 어댑터 meta 에만 있다(§31-5)."""
PURPOSE_OPTIONAL: dict[str, frozenset[str]] = {
    "main": frozenset(),
    "rehearsal": frozenset({"generation.train_rows_digest", "generation.plan_sha256"}),
    "frame_diag": frozenset(),
}
"""목적마다 **비어 있어도 되는** 생성 쪽 칸. 값이면 곁 파일 · 어댑터 meta 와 같아야 한다 — 대조는 읽는 쪽의 일이다."""
PROBE_SCORING_REQUIRED = ("canary.coord_cfg", "canary.coord_fixture_sha256")
"""리허설 · 진단이 채점 쪽에서 요구하는 칸 — 에코 관문 둘. 나머지 빈 칸은 이름을 산출에 싣는다."""
_RULES_PATH = re.compile(r"^configs/registration/frame_diag_rules-[0-9]{8}-[1-9][0-9]*\.json$")
"""진단 규칙 파일의 자리와 이름(§31-1 의 1)."""

CI_ADOPTED: dict[str, object] = {f"ci.{k}": v for k, v in _REGISTERED_CI.items()}
"""`ci` 의 규약 칸 → 2026-10-01 채택값(판 4). 이 값이 아니면 무효다. 아래 네 어휘는 **고르기 전의 선택지 기록**이다 —
검사에 쓰지 않는다."""

UNDEFINED_DENOMINATOR_POLICIES = ("drop_undefined", "signed_infinity", "no_interval_if_any")
"""규약안 축 A 의 A1 · A2 · A3. `drop_undefined` 는 사전실험 구간이 쓴 규약의 이름이다(`recovery_ci.recovery_from`)."""
INFINITY_HANDLINGS = ("drop_nonfinite", "include_as_extreme", "clamp_to_outermost_finite")
"""규약안 축 B 의 B1 · B2 · B3."""
PERCENTILE_METHODS = ("linear", "nearest", "lower", "higher", "lower_lo_higher_hi",
                      "inverted_cdf", "closest_observation", "hazen", "weibull", "median_unbiased")
"""규약안 축 C 의 C1 · C2 · C3-가·나 · C3-다 · C4-가(둘) · C4-나(셋). `lower_lo_higher_hi` 는 하한 내림 · 상한 올림이다."""
ABSENCE_REPRESENTATIONS = ("null", "bare_nonfinite", "state_field")
"""규약안 축 D 의 D-가 · D-나 · D-다."""

VOCAB: dict[str, tuple[str, ...]] = {
    "metric.coupling_rule": ("decoupled_v2",),
    "metric.macro_support_rule": ("gold_support",),
    "baseline.fit_split": ("train_then_refit_trainval",),
    "ci.interval": ("percentile",),
    "ci.cluster_unit": ("component", "group"),
    "population.cluster_unit": ("component", "group"),
    "population.weighting": ("none",),
    "generation.purpose": PURPOSES,
    "generation.list_split": tuple(dict.fromkeys(SPLIT_OF_PURPOSE.values())),
    "generation.coord_space": ("ABS_ORIG", "ABS_RESIZED", "NORM_1000"),
    "generation.padding_side": ("left", "right"),
}
"""칸 → 허용 어휘. 여기 없는 문자열 칸은 **비지 않은 문자열**인지만 본다 — 계약이 선택지를 적지 않은 칸이다."""

HEX64_FIELDS: frozenset[str] = frozenset({
    "baseline.prediction_sha256",
    "population.exclusion_list_sha256", "population.duplicate_pairs_sha256",
    "population.eval_list_file_sha256", "population.eval_list_set_sha256",
    "generation.snapshot_digest", "generation.eval_list_file_sha256", "generation.eval_list_set_sha256",
    "generation.prompt_sha256", "generation.gen_prefix_sha256", "generation.processor_config_sha256",
    "generation.coord_cfg_hash", "generation.start_checkpoint_sha256", "generation.train_config_sha256",
    "generation.echo_list_file_sha256", "generation.echo_list_set_sha256",
    "generation.train_rows_digest", "generation.plan_sha256", "generation.frame_diag_rules_sha256",
    "canary.conformance_vector_sha256", "canary.known_answer_sha256",
    "canary.coord_fixture_sha256", "canary.frame_diag_sha256",
})
"""소문자 16진 64자리여야 하는 칸. 한 글자짜리도 받으면 "해시를 적었다" 가 아무 뜻이 없다."""

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_RATIONAL = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*$")
_SEED_NO = re.compile(r"^[1-9][0-9]*$")
_IMPL_KEY = re.compile(r"^[^=\s]+=[^:\s]+:[^@\s]+@\S+$")

START_FIELDS = ("start_checkpoint_sha256", "init_adapter_digest")
"""출발점 두 필드. **정확히 하나가 값이고 나머지는 null** 이다(07번 §13-3 마-13 · 판정 06 의 3)."""

_AT_LEAST: dict[str, int] = {
    "baseline.strata_k": 1, "ci.n_resamples": 1, "ci.rng_seed": 0, "ci.min_defined_draws": 1,
    "population.n_images": 1, "population.n_clusters": 1,
    "generation.max_new_tokens": 1, "generation.batch_size": 0,
    "generation.processor_min_pixels": 1, "generation.processor_max_pixels": 1,
    "generation.patch_size": 1, "generation.merge_size": 1,
    "generation.budget_n": 1, "generation.budget_r": 1, "generation.budget_e": 1,
    "canary.axis_ratio_min_pairs": 1, "planned_seed_count": 1,
}
"""정수 칸의 하한. `batch_size` 는 0 을 받는다 — "배치를 쓰지 않는다" 는 결정이다(모듈 머리말)."""

_UNIT_OPEN = ("ci.alpha",)
"""(0, 1) 열린 구간."""
_UNIT_CLOSED = ("ci.rule4_max_undefined_fraction", "baseline.full_population_m1")
"""[0, 1] 닫힌 구간."""
_NON_NEGATIVE = ("canary.echo_max_coord_diff", "canary.literal_match_tolerance")
_POSITIVE = ("ci.rule2_max_half_width", "canary.axis_ratio_threshold")

_NON_EMPTY_DICT = ("generation.decoding",)
"""빈 사전이면 값이 아닌 칸. 디코딩 설정이 비어 있으면 greedy 1회를 적을 수 없다."""

_SPECS = {"metric": MetricSpec, "baseline": BaselineSpec, "ci": CiSpec,
          "population": PopulationSpec, "generation": GenerationSpec, "canary": CanarySpec}
_GENERATION_AREAS = ("generation",)
_SCORING_AREAS = ("metric", "baseline", "ci", "population", "canary")
_SIDES = ("generation", "scoring", "all")


def _kind(hint) -> str:
    """자료형 표기 → 검사 종류. 표기는 이 모듈의 칸들이 쓰는 것뿐이다."""
    text = str(hint)
    for key, name in (("CoordCfgRecord", "coord_cfg"), ("tuple[str", "str_tuple"),
                      ("dict[str, int]", "str_int_dict"), ("dict", "dict"), ("bool", "bool"),
                      ("int", "int"), ("float", "float"), ("str", "str")):
        if key in text:
            return name
    raise TypeError(f"검사 종류를 모르는 자료형: {text}")


@lru_cache(maxsize=1)
def _kinds() -> dict[str, str]:
    out: dict[str, str] = {}
    for area, cls in _SPECS.items():
        hints = get_type_hints(cls, globalns={**globals(), "CoordCfgRecord": CoordCfgRecord})
        for f in fields(cls):
            out[f"{area}.{f.name}"] = _kind(hints[f.name])
    return out


def _check_value(name: str, kind: str, v) -> str | None:
    """한 칸의 값이 그 칸에 올 수 있는가. 올 수 없으면 사유."""
    if kind == "bool":
        return None if type(v) is bool else "참·거짓이어야 한다"
    if kind == "int":
        if type(v) is not int:
            return "정수여야 한다(참·거짓·실수·문자열은 받지 않는다)"
        low = _AT_LEAST.get(name)
        return None if low is None or v >= low else f"{low} 이상이어야 한다"
    if kind == "float":
        if type(v) not in (int, float) or not math.isfinite(v):
            return "유한한 수여야 한다(참·거짓·문자열은 받지 않는다)"
        if name in _UNIT_OPEN and not 0 < v < 1:
            return "0 과 1 사이(양 끝 제외)여야 한다"
        if name in _UNIT_CLOSED and not 0 <= v <= 1:
            return "0 이상 1 이하여야 한다"
        if name in _NON_NEGATIVE and v < 0:
            return "0 이상이어야 한다"
        if name in _POSITIVE and v <= 0:
            return "0 보다 커야 한다"
        return None
    if kind == "str":
        if not isinstance(v, str) or not v:
            return "비지 않은 문자열이어야 한다"
        if name in CI_ADOPTED and v != CI_ADOPTED[name]:
            return f"2026-10-01 채택값과 다르다 — {CI_ADOPTED[name]!r}"
        if name in VOCAB and v not in VOCAB[name]:
            return f"허용 어휘 밖이다 — {list(VOCAB[name])}"
        if name in HEX64_FIELDS and not _HEX64.match(v):
            return "sha256(소문자 16진 64자리)이 아니다"
        if name == "metric.iou_threshold":
            if not _RATIONAL.match(v):
                return "유리수 문자열(`p/q`)이 아니다"
            p, q = (int(x) for x in v.split("/"))
            if not 0 < p <= q:
                return "0 보다 크고 1 이하여야 한다"
        if name == "generation.frame_diag_rules_path" and not _RULES_PATH.match(v):
            return "`configs/registration/frame_diag_rules-{YYYYMMDD}-{n}.json` 꼴이 아니다"
        return None
    if kind == "str_tuple":
        if not isinstance(v, (tuple, list)) or not v:
            return "비지 않은 문자열 목록이어야 한다"
        if not all(isinstance(x, str) and x for x in v):
            return "목록의 원소가 비지 않은 문자열이 아니다"
        if len(set(v)) != len(v):
            return "목록에 중복이 있다"
        if name == "generation.impl_ids" and not all(_IMPL_KEY.match(x) for x in v):
            return "구현 식별자의 열쇠(`{seam}={module}:{qualname}@{source_path}`)가 아니다"
        if name in CI_ADOPTED and tuple(v) != CI_ADOPTED[name]:
            return f"2026-10-01 채택값과 다르다 — {list(CI_ADOPTED[name])!r}"
        return None
    if kind == "str_int_dict":
        if not isinstance(v, dict) or not v:
            return "비지 않은 사전이어야 한다"
        if not all(isinstance(k, str) and _SEED_NO.match(k) for k in v):
            return "키가 1 이상의 정수를 적은 문자열이 아니다"
        if not all(type(x) is int and x >= 1 for x in v.values()):
            return "값이 1 이상의 정수가 아니다"
        return None
    if kind == "dict":
        if not isinstance(v, dict):
            return "사전이어야 한다"
        if name in _NON_EMPTY_DICT and not v:
            return "빈 사전은 값이 아니다"
        return None
    if kind == "coord_cfg":
        return None if isinstance(v, CoordCfgRecord) else "CoordCfgRecord 가 아니다"
    raise TypeError(kind)


@dataclass
class UnifiedRegistration:
    """통합형 본실험의 등록 블록 하나."""

    version: str = REGISTRATION_VERSION
    metric: MetricSpec = field(default_factory=MetricSpec)
    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    ci: CiSpec = field(default_factory=CiSpec)
    population: PopulationSpec = field(default_factory=PopulationSpec)
    generation: GenerationSpec = field(default_factory=GenerationSpec)
    canary: CanarySpec = field(default_factory=CanarySpec)
    seeds: dict[str, int] | None = None
    """시드 번호(문자열) → 시드 값. 채점 쪽이다. 값이 겹치면 무효다(§22-2)."""
    planned_seed_count: int | None = None
    adoption_anchor: str | None = None
    """의사결정로그에서 이 등록을 가리키는 **미리 정한 앵커 문자열**. 서로 가리키면 닫히지 않는다."""

    # ── 지문 ────────────────────────────────────────────────────────────────
    def _canonical(self, part) -> str:
        return json.dumps(part, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def generation_payload(self) -> dict:
        return {"version": self.version, "generation": asdict(self.generation)}

    def scoring_payload(self) -> dict:
        return {"version": self.version, "metric": asdict(self.metric),
                "baseline": asdict(self.baseline), "ci": asdict(self.ci),
                "population": asdict(self.population), "canary": asdict(self.canary),
                "seeds": self.seeds, "planned_seed_count": self.planned_seed_count,
                "adoption_anchor": self.adoption_anchor}

    def generation_sha256(self) -> str:
        return hashlib.sha256(self._canonical(self.generation_payload()).encode("utf-8")).hexdigest()

    def scoring_sha256(self) -> str:
        return hashlib.sha256(self._canonical(self.scoring_payload()).encode("utf-8")).hexdigest()

    # ── 완결성 ──────────────────────────────────────────────────────────────
    def missing(self, *, side: str = "all") -> list[str]:
        """빈 항목의 이름. **`None` 만 빈 값이다** — `False`·`0`·빈 문자열은 값이다.

        출발점 두 필드는 짝으로 본다 — 하나가 값이면 둘 다 빈 항목이 아니다.
        """
        _check_side(side)
        out: list[str] = []

        def walk(prefix: str, obj) -> None:
            for key, value in obj.items():
                name = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict) and key in _SPECS:
                    walk(name, value)
                elif value is None:
                    out.append(name)

        payloads = {"generation": self.generation_payload, "scoring": self.scoring_payload}
        parts = payloads if side == "all" else {side: payloads[side]}
        for maker in parts.values():
            payload = dict(maker())
            payload.pop("version", None)
            walk("", payload)
        if "generation" in parts and any(getattr(self.generation, f) is not None for f in START_FIELDS):
            out = [n for n in out if n not in {f"generation.{f}" for f in START_FIELDS}]
        purpose = self.generation.purpose
        if "generation" in parts and purpose is not None:
            # 목적이 정해져야 어느 칸이 null 이어야 하는지 안다. 목적이 비면 목적 자체가 빈 항목으로 나오고
            # 조건부 칸도 그대로 빈 항목으로 남긴다 — 무엇을 채워야 하는지 모르는 채로 줄이지 않는다.
            # 목적이 어휘 밖이면 조건부 칸을 세지 않는다 — 사유는 목적 쪽(`invalid`)이고, 빈 항목이 그것을 가리면 안 된다.
            skip = (PURPOSE_NULL[purpose] | PURPOSE_OPTIONAL[purpose] if purpose in PURPOSE_NULL
                    else frozenset().union(*PURPOSE_NULL.values(), *PURPOSE_OPTIONAL.values()))
            out = [n for n in out if n not in skip]
        return sorted(set(out))

    def invalid(self, *, side: str = "all") -> list[str]:
        """채워진 값 가운데 **그 칸에 올 수 없는 것**. `항목: 사유` 의 목록이다. 빈 칸은 보지 않는다(`missing` 의 일)."""
        _check_side(side)
        areas = {"generation": _GENERATION_AREAS, "scoring": _SCORING_AREAS,
                 "all": _GENERATION_AREAS + _SCORING_AREAS}[side]
        kinds = _kinds()
        out: list[str] = []
        for area in areas:
            spec = getattr(self, area)
            for f in fields(spec):
                v = getattr(spec, f.name)
                if v is None:
                    continue
                why = _check_value(f"{area}.{f.name}", kinds[f"{area}.{f.name}"], v)
                if why:
                    out.append(f"{area}.{f.name}: {why}")
        if side in ("scoring", "all"):
            out += self._invalid_scoring_top()
        if side in ("generation", "all"):
            out += self._invalid_generation_relations()
        if side in ("scoring", "all"):
            out += self._invalid_scoring_relations()
        return out

    def _invalid_scoring_top(self) -> list[str]:
        out: list[str] = []
        s = self.seeds
        if s is not None:
            if not isinstance(s, dict) or not s:
                out.append("seeds: 비지 않은 사전이어야 한다")
            elif not all(isinstance(k, str) and _SEED_NO.match(k) for k in s):
                out.append("seeds: 키가 1 이상의 정수를 적은 문자열이 아니다")
            elif not all(type(v) is int and v >= 0 for v in s.values()):
                out.append("seeds: 시드 값이 0 이상의 정수가 아니다(참·거짓은 받지 않는다)")
            elif len(set(s.values())) != len(s):
                out.append("seeds: 시드 값이 겹친다 — 번호가 달라도 같은 학습이다(§22-2)")
        if self.planned_seed_count is not None:
            why = _check_value("planned_seed_count", "int", self.planned_seed_count)
            if why:
                out.append(f"planned_seed_count: {why}")
        if self.adoption_anchor is not None:
            why = _check_value("adoption_anchor", "str", self.adoption_anchor)
            if why:
                out.append(f"adoption_anchor: {why}")
        return out

    def _invalid_generation_relations(self) -> list[str]:
        g = self.generation
        out: list[str] = []
        if all(getattr(g, f) is not None for f in START_FIELDS):
            out.append("generation.start_checkpoint_sha256·init_adapter_digest: 출발점 두 필드가 둘 다 값이다 — "
                       "정확히 하나만 값이고 나머지는 null 이다")
        if g.purpose in SPLIT_OF_PURPOSE and g.list_split is not None \
                and g.list_split != SPLIT_OF_PURPOSE[g.purpose]:
            out.append(f"generation.list_split: 목적 {g.purpose!r} 의 기준 분할은 "
                       f"{SPLIT_OF_PURPOSE[g.purpose]!r} 이다 — 받은 값 {g.list_split!r}")
        for name in sorted(PURPOSE_NULL.get(g.purpose, ())):
            if getattr(g, name.split(".", 1)[1]) is not None:
                out.append(f"{name}: 목적 {g.purpose!r} 에서는 null 이어야 한다(07번 §31-5)")
        if (g.frame_diag_rules_sha256 is None) != (g.frame_diag_rules_path is None):
            out.append("generation.frame_diag_rules_path·frame_diag_rules_sha256: 진단 규칙의 경로와 digest 는 "
                       "둘 다 값이거나 둘 다 null 이다")
        if g.echo_list_file_sha256 is not None and g.echo_list_file_sha256 == g.eval_list_file_sha256 \
                and g.echo_list_set_sha256 is not None and g.eval_list_set_sha256 is not None \
                and g.echo_list_set_sha256 != g.eval_list_set_sha256:
            out.append("generation.echo_list_set_sha256: 생성 목록과 파일 해시가 같은데 집합 해시가 다르다")
        if type(g.processor_min_pixels) is int and type(g.processor_max_pixels) is int \
                and g.processor_min_pixels > g.processor_max_pixels:
            out.append("generation.processor_min_pixels: 최대보다 크다")
        out += _list_pair_problem("generation", g.eval_list_file_sha256, g.eval_list_set_sha256, None, None)
        return out

    def _invalid_scoring_relations(self) -> list[str]:
        out: list[str] = []
        c = self.ci
        if type(c.min_defined_draws) is int and type(c.n_resamples) is int \
                and c.min_defined_draws > c.n_resamples:
            out.append("ci.min_defined_draws: 재표집 수보다 크다")
        if c.percentile_method == CI_ADOPTED["ci.percentile_method"] and type(c.alpha) in (int, float) \
                and 0 < c.alpha < 1 and c.alpha != 0.05:
            out.append("ci.alpha: 순위 식의 25/1000 은 α = 0.05 의 α/2 다 — 식과 α 가 어긋난다")
        g, p = self.generation, self.population
        out += _list_pair_problem("population", g.eval_list_file_sha256, g.eval_list_set_sha256,
                                  p.eval_list_file_sha256, p.eval_list_set_sha256)
        return out

    def require_complete(self, *, side: str = "all") -> None:
        """빈 칸이 없고 채워진 값이 모두 유효해야 지난다."""
        gaps = self.missing(side=side)
        if gaps:
            raise RegistrationIncomplete(gaps)
        problems = self.invalid(side=side)
        if problems:
            raise RegistrationInvalid(problems)

    def dependency_gaps(self, changed: list[str]) -> list[str]:
        """바꾼 항목이 끌고 가는 것들. 함께 다시 정하지 않으면 등록이 스스로와 어긋난다.

        이름은 영역까지 적는다(`generation.coord_space`). 등록에 없는 이름을 주면 거부한다 — 오타가 빈 목록으로 지나지 않게.
        """
        known = set(_kinds()) | {"seeds", "planned_seed_count", "adoption_anchor"}
        unknown = sorted(set(changed) - known)
        if unknown:
            raise ValueError(f"등록에 없는 항목: {unknown} — 이름은 영역까지 적는다(예: generation.coord_space)")
        out: set[str] = set()
        for key in changed:
            out.update(DEPENDENCIES.get(key, ()))
        return sorted(out)


def _check_side(side: str) -> None:
    if side not in _SIDES:
        raise ValueError(f"영역은 {_SIDES} 가운데 하나다 — 받은 값 {side!r}")


def _list_pair_problem(where: str, gen_file, gen_set, pop_file, pop_set) -> list[str]:
    """목록 해시 두 벌의 관계. **파일 해시가 같으면 집합 해시도 같다** — 같은 바이트는 같은 집합이다.

    채점 쪽 목록은 생성 쪽 목록과 같거나 그 부분집합이다(§13-2 가). 부분집합인지는 해시로 볼 수 없어
    목록을 받아 `check_list_binding` 이 본다. 여기서는 해시만으로 모순인 경우를 막는다.
    """
    if pop_file is None:
        return []
    if gen_file is not None and gen_set is not None and pop_set is not None \
            and pop_file == gen_file and pop_set != gen_set:
        return [f"{where}.eval_list_set_sha256: 생성 쪽과 파일 해시가 같은데 집합 해시가 다르다"]
    return []


class ListBindingError(RuntimeError):
    """등록의 목록 해시 두 벌이 실제 목록과 맞지 않거나, 채점 목록이 생성 목록의 부분집합이 아니다."""


def check_list_binding(registration: UnifiedRegistration, generation_list, population_list) -> None:
    """등록의 **목록 해시 두 벌**을 실제 목록과 맞대고, 채점 목록이 생성 목록의 부분집합인지 본다.

    두 목록은 `evaluation.eval_list.validate_eval_list` 가 만든 객체다(해시를 id 에서 다시 계산해 둔 것).
    생성 쪽과 채점 쪽이 같은 목록이면 두 벌이 같고, 부분집합 재채점이면 채점 쪽이 생성 쪽의 부분집합이다.

    Raises:
        ListBindingError: 어느 한 쌍이라도 어긋나면. 사유를 전부 모아서 낸다.
    """
    g, p = registration.generation, registration.population
    reasons: list[str] = []
    if (g.eval_list_file_sha256, g.eval_list_set_sha256) != \
            (generation_list.file_sha256, generation_list.set_sha256):
        reasons.append("생성 쪽 등록의 목록 해시가 생성 목록과 다르다")
    if (p.eval_list_file_sha256, p.eval_list_set_sha256) != \
            (population_list.file_sha256, population_list.set_sha256):
        reasons.append("채점 쪽 등록의 목록 해시가 채점 목록과 다르다")
    extra = population_list.id_set - generation_list.id_set
    if extra:
        reasons.append(f"채점 목록이 생성 목록의 부분집합이 아니다 — 생성 목록 밖 {len(extra)}개")
    if reasons:
        raise ListBindingError(" · ".join(reasons))


# ── 등록 파일 ───────────────────────────────────────────────────────────────

def dump_registration(registration: UnifiedRegistration) -> bytes:
    """등록을 **정규 바이트**로 적는다 — 키 정렬 · UTF-8 · 한 줄 + LF. `load_registration` 이 받는 유일한 꼴이다."""
    body = {"version": registration.version,
            **{area: asdict(getattr(registration, area)) for area in _SPECS},
            "seeds": registration.seeds, "planned_seed_count": registration.planned_seed_count,
            "adoption_anchor": registration.adoption_anchor}
    return (json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n").encode("utf-8")


class RegistrationFileError(ValueError):
    """등록 파일의 바이트가 등록의 정규형이 아니다."""


def load_registration(raw: bytes) -> UnifiedRegistration:
    """등록 파일의 바이트에서 등록 객체를 만든다. **정규형만 받는다.**

    읽은 객체를 다시 적은 바이트가 받은 바이트와 같아야 한다. 그래서 같은 바이트면 같은 등록이고,
    손으로 고친 파일(키 순서·공백·줄끝)은 여기서 멈춘다. 모르는 키나 빠진 키도 멈춘다 — 오타가 난 키가
    조용히 사라지면 그 항목이 빈 채로 지문에 들어간다.
    """
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise RegistrationFileError("BOM 이나 CR 이 있다")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistrationFileError(f"JSON 이 아니다 ({type(exc).__name__})") from None
    if not isinstance(body, dict):
        raise RegistrationFileError("객체가 아니다")
    top = {"version", *_SPECS, "seeds", "planned_seed_count", "adoption_anchor"}
    if set(body) != top:
        raise RegistrationFileError(f"최상위 키가 다르다 — 모르는 키 {sorted(set(body) - top)}, "
                                    f"빠진 키 {sorted(top - set(body))}")
    if body["version"] != REGISTRATION_VERSION:
        raise RegistrationFileError(f"등록 판이 {body['version']!r} 다 — 이 코드는 {REGISTRATION_VERSION!r} 를 읽는다")
    kinds = _kinds()
    parts = {}
    for area, cls in _SPECS.items():
        got = body[area]
        names = {f.name for f in fields(cls)}
        if not isinstance(got, dict) or set(got) != names:
            raise RegistrationFileError(f"{area}: 칸이 등록 판과 다르다")
        kw = {}
        for name, v in got.items():
            kind = kinds[f"{area}.{name}"]
            if kind == "str_tuple" and isinstance(v, list):
                v = tuple(v)
            elif kind == "coord_cfg" and isinstance(v, dict):
                v = CoordCfgRecord(**{k: tuple(x) if isinstance(x, list) else x for k, x in v.items()})
            kw[name] = v
        parts[area] = cls(**kw)
    reg = UnifiedRegistration(version=body["version"], seeds=body["seeds"],
                              planned_seed_count=body["planned_seed_count"],
                              adoption_anchor=body["adoption_anchor"], **parts)
    if dump_registration(reg) != raw:
        raise RegistrationFileError("정규형이 아니다 — dump_registration 이 적은 바이트만 받는다")
    return reg


# ── 영수증 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Receipt:
    """등록이 먼저였다는 증거. **등록 병합 다음 커밋으로 적는다.**

    `kind` 는 영수증의 종류(`main` · `rehearsal` · `frame_diag`)이고 등록의 목적과 같아야 한다 — 커밋 자리에 접두
    문자열을 붙여 가르지 않는다. `registration_path` 는 `main_commit` 에서 등록 파일을 읽을 저장소 상대 경로다.
    `registration_file_sha256` 은 그 파일 **바이트**의 sha256 이다 — 영수증이 가리키는 등록과 지금 읽은 등록이
    같은 바이트인지 지문을 계산하기 전에 본다(07번 §30-10).
    """

    generation_sha256: str
    scoring_sha256: str
    registered_at: str
    main_commit: str
    kind: str
    registration_path: str
    registration_file_sha256: str
    device: str = ""

    def __post_init__(self) -> None:
        bad: list[str] = []
        for name in ("generation_sha256", "scoring_sha256", "registration_file_sha256"):
            if not (isinstance(getattr(self, name), str) and _HEX64.match(getattr(self, name))):
                bad.append(f"{name} 가 sha256 이 아니다")
        if not (isinstance(self.main_commit, str) and _HEX40.match(self.main_commit)):
            bad.append("main_commit 이 커밋 SHA(소문자 40자리)가 아니다")
        if self.kind not in RECEIPT_KINDS:
            bad.append(f"kind 가 {list(RECEIPT_KINDS)} 가운데 하나가 아니다")
        path = self.registration_path
        if not (isinstance(path, str) and path) or "\\" in path or path.startswith("/") \
                or ":" in path or ".." in PurePosixPath(path).parts:
            bad.append("registration_path 가 저장소 상대 POSIX 경로가 아니다")
        if bad:
            raise ReceiptMissing("영수증의 꼴이 틀렸다: " + " · ".join(bad))
        _ = self.time      # naive 시각이면 여기서 막힌다

    @property
    def time(self) -> datetime:
        try:
            t = datetime.fromisoformat(self.registered_at)
        except (TypeError, ValueError):
            raise ReceiptMissing("영수증 시각이 ISO 형식이 아니다") from None
        if t.tzinfo is None:
            raise ReceiptMissing("영수증 시각에 오프셋이 없다 — naive 시각은 받지 않는다")
        return t


RECEIPT_KEYS = ("generation_sha256", "scoring_sha256", "registered_at", "main_commit",
                "kind", "registration_path", "registration_file_sha256")


def read_receipt(path: Path) -> Receipt:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return Receipt(**{k: data[k] for k in RECEIPT_KEYS}, device=data.get("device", ""))
    except KeyError as exc:
        raise ReceiptMissing(f"영수증에 {exc.args[0]} 이 없다") from None


def require_receipt(registration: UnifiedRegistration, receipt: Receipt, *,
                    side: str = "generation") -> str:
    """영수증이 **지금 등록**을 가리키는지 보고 그 영역의 지문을 돌려준다.

    `side="generation"` 은 내보내기가 부른다 — 채점 쪽이 뒤에 개정돼도 예측이 무효가 되지 않는다(§12-5).
    **본채점은 `generation` 과 `scoring` 을 둘 다 부른다.** `all` 은 받지 않는다 — 앞 판은 `all` 로 부르면
    완결은 둘 다 보고 지문은 채점 쪽만 맞댔다.

    Raises:
        RegistrationIncomplete · RegistrationInvalid: 그 영역이 비었거나 무효면.
        ReceiptMissing: 영수증의 종류가 등록의 목적과 다르거나 지문이 어긋나면.
    """
    if side not in ("generation", "scoring"):
        raise ValueError(f"영역은 generation 또는 scoring 이다 — 받은 값 {side!r}. 둘 다 보려면 두 번 부른다")
    registration.require_complete(side=side)
    purpose = registration.generation.purpose
    if purpose is not None and purpose != receipt.kind:
        raise ReceiptMissing(f"영수증의 종류 {receipt.kind!r} 가 등록의 목적 {purpose!r} 와 다르다")
    want = (registration.generation_sha256() if side == "generation"
            else registration.scoring_sha256())
    got = receipt.generation_sha256 if side == "generation" else receipt.scoring_sha256
    if want != got:
        raise ReceiptMissing(
            f"영수증의 {side} 지문이 지금 등록과 다르다 — 영수증이 가리키는 값으로만 돈다"
        )
    return want


def registration_for_receipt(raw: bytes, receipt: Receipt, *, side: str) -> tuple[UnifiedRegistration, str]:
    """영수증의 커밋에서 읽은 **등록 파일의 바이트**로 등록 객체를 만들고 영수증과 맞댄다.

    바이트를 읽는 일(`git show <main_commit>:<registration_path>`)은 호출부가 한다 — 작업 트리의 파일을 읽으면
    "영수증이 가리키는 등록" 과 "지금 쓰는 등록" 이 갈릴 수 있다. **바이트의 해시를 먼저 본다** — 영수증의
    `registration_file_sha256` 과 다르면 등록 객체를 만들지 않는다.
    """
    require_file_hash(raw, receipt)
    reg = load_registration(raw)
    return reg, require_receipt(reg, receipt, side=side)


def require_file_hash(raw: bytes, receipt: Receipt) -> None:
    """등록 파일 바이트의 sha256 이 영수증의 `registration_file_sha256` 과 같은가."""
    if hashlib.sha256(raw).hexdigest() != receipt.registration_file_sha256:
        raise ReceiptMissing("등록 파일 바이트의 sha256 이 영수증의 registration_file_sha256 과 다르다 — "
                             "영수증이 가리키는 등록이 아니다")


def require_probe_scoring(registration: UnifiedRegistration, receipt: Receipt) -> list[str]:
    """리허설 · 진단의 **채점 쪽** — 에코 관문 두 칸만 요구하고, 나머지 빈 칸의 이름을 돌려준다.

    진단에 채점 쪽 전체를 요구하지 않는다 — 채점 쪽에는 구간 규약 칸(보류-4)이 있어 팀이 고르기 전까지 비어 있고,
    진단 규칙은 생성 쪽에 있다(07번 §31-8 의 11). 그래도 **채점 지문은 영수증과 같아야 한다** — 에코 관문 두 칸이
    등록된 그 값인지가 거기서 선다. 채워진 칸이 무효면 멈춘다(빈 칸과 틀린 칸은 다르다).

    Raises:
        ValueError: 목적이 `rehearsal` · `frame_diag` 가 아니면 — 본채점은 `require_receipt` 를 두 영역으로 부른다.
        ReceiptMissing: 종류가 목적과 다르거나 채점 지문이 영수증과 다르면.
        RegistrationIncomplete: 에코 관문 두 칸 가운데 빈 것이 있으면.
        RegistrationInvalid: 채점 쪽의 채워진 값이 무효면.
    """
    purpose = registration.generation.purpose
    if purpose not in ("rehearsal", "frame_diag"):
        raise ValueError(f"리허설 · 진단의 채점 쪽 검사다 — 목적 {purpose!r}. 본채점은 require_receipt 를 두 영역으로 부른다")
    if purpose != receipt.kind:
        raise ReceiptMissing(f"영수증의 종류 {receipt.kind!r} 가 등록의 목적 {purpose!r} 와 다르다")
    if registration.scoring_sha256() != receipt.scoring_sha256:
        raise ReceiptMissing("영수증의 scoring 지문이 지금 등록과 다르다 — 영수증이 가리키는 값으로만 돈다")
    gaps = registration.missing(side="scoring")
    lacking = [n for n in PROBE_SCORING_REQUIRED if n in gaps]
    if lacking:
        raise RegistrationIncomplete(lacking)
    problems = registration.invalid(side="scoring")
    if problems:
        raise RegistrationInvalid(problems)
    return gaps
