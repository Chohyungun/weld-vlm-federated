"""1.4 묶음 검증 — 이 파일들로 수치를 내도 되는가. 07번 미니스펙 §13-3 · §14-1 · §16-1 · §16-2.

묶음 하나는 `generations.jsonl` 과 곁 파일 셋(시작 도장 · 곁 파일 · 시도 기록), 선택으로 `tokens.jsonl` 이다.
**채점을 시작하기 전에** 이 파일들이 약속한 그 파일인지, 약속한 조건으로 만들어졌는지를 본다.

## 왜 이렇게 촘촘한가

`_s2` 파일을 `_s1` 로 복사해도 통과하는 계약은 쓸 수 없다. 시드 둘이 같아지면 시드 간 표준편차가 줄고,
그 위에 선 분모 규칙이 잘못 통과한다 — 수치가 조용히 틀리고 아무 데서도 걸리지 않는다.
C1 과 C2 는 둘 다 강재라 뒤바뀌어도 값이 그럴듯해 참여자 보고가 통째로 뒤집힌다.

그래서 **파일 신원(바이트 해시) · 내용(id 수열) · 조건(등록값과 항목별 대조) · 출처(학습 산출물·원장)** 를 따로 본다.

## 해시가 대신하지 못하는 것

바이트 해시는 "곁 파일이 가리키는 그 파일" 임을 말하고 **그 안에 무엇이 적혀 있는지는 말하지 않는다.**
다른 묶음의 시작 도장을 놓고 그 해시만 곁 파일에 적으면 해시 검사는 통과한다. 그래서 시작 도장과
시도 기록은 **내용을 곁 파일·등록값과 맞댄다**(`_check_start_stamp`·`_check_attempts`).

## 시도 기록은 시도마다 두 줄이다

시작할 때 `start` 한 줄, 스스로 끝날 때 `end` 한 줄이다(§27-1). **죽은 시도는 끝 줄이 없는 것으로
드러나고 아무도 대신 적지 않는다** — 그래서 추정한 시각·추정한 끝 id 가 기록에 들어가지 않는다.
시도가 남긴 줄 수는 **생성 목록의 자리 차**로 센다(§28-1) — 적힌 수를 더하면 죽은 시도가 남긴
줄이 어느 끝 줄에도 세지지 않아 바르게 쓴 묶음이 거부된다.

## 바이트 규약은 다섯 파일 전부에 걸린다

§14-1 가는 generations·tokens·시작 도장·곁 파일·`attempts.jsonl` **다섯 모두**에 UTF-8·LF·BOM 없음을
요구한다. 다만 **JSON 문서와 JSONL 의 개행 규칙은 같지 않다** — 끝 개행과 빈 줄 금지는 줄 단위 파일에만
건다(`_check_file_bytes` 의 `jsonl` 인자).

## 단계와 멈춤

`reject_v14.Stage` 의 A→E 순서로 돌고, **앞 단계에서 걸리면 뒤를 보지 않는다.**
바이트가 깨진 파일의 내용을 해석하면 엉뚱한 사유가 올라와 원인을 가린다.
한 단계 안에서는 걸린 것을 **모두** 모은다 — 한 번에 고칠 수 있게.

## 여기서 하지 않는 것

파일을 **찾아다니지 않는다.** 등록 객체·목록·학습 원장·산출물 meta 는 호출부가 읽어 넘긴다.
이 모듈이 경로를 조립하면 시험이 자산 없이 돌 수 없고, 무엇을 읽는지가 호출부마다 달라진다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from evaluation.eval_list import EvalList, set_digest
from evaluation.reject_v14 import (
    ECHO_EXEMPT,
    BundleRejected,
    RejectCode,
    Rejection,
    Stage,
)
from evaluation.actuals import compare_values, impl_identifiers, impl_problems
from evaluation.prereg_unified import PURPOSES
from evaluation.schema_v14 import UNI_MAIN_TAGS, tag_parts

MODE_MODEL = "model"
MODE_ECHO = "echo"

_NAME = re.compile(r"^(?P<tag>[A-Za-z0-9_]+)_s(?P<seed>\d+)(?P<echo>\.echo)?\.generations\.jsonl$")
"""파일 이름의 꼴. **쪼개지 않고 정규식으로 읽는다** — 밑줄로 나누면 `uni_local_C1` 이 갈린다."""

SIDECAR_REQUIRED: tuple[str, ...] = (
    "cell", "client", "tag", "seed_index", "seed_value", "mode", "n_lines",
    "generations_sha256", "eval_list_file_sha256", "eval_list_set_sha256", "start_stamp_sha256",
    "snapshot_digest", "coord_space", "coord_cfg_hash",
    "generation_sha256", "prompt_sha256", "chat_template_kwargs", "gen_prefix_sha256",
    "max_new_tokens", "decoding", "batch_size", "padding_side",
    "processor_config_sha256", "processor_min_pixels", "processor_max_pixels",
    "patch_size", "merge_size", "image_grid_thw_observed", "transformers_version",
    "base_model_id", "base_model_revision",
    "train_run_id", "train_ledger_path", "train_ledger_sha256", "train_config_sha256",
    "budget_n", "budget_r", "budget_e",
    "export_commit", "started_at", "finished_at", "device", "attempts",
    "purpose", "train_config", "impl_ids", "fault_events",
    "list_split", "start_checkpoint_sha256", "init_adapter_digest",
)
"""곁 파일에 **반드시 있어야 하는** 키. 없으면 `SIDECAR_SCHEMA` 다. `purpose` 부터 넷은 07번 §30-4 가 더했다.
끝의 셋은 검수 16번 M-2 가 더했다 — 빠지면 등록값 대조(D 단계)가 아니라 여기(A 단계)서 드러나게. 출발점 두 칸은
**키가** 있어야 하고 값은 `null` 일 수 있다(정확히 하나가 값 — 등록과 같아야 한다, 리허설 2판 반영판 §2-5).

`scored_adapter_sha256`·`scored_adapter_step` 은 모드에 따라 갈려서 여기 없다(에코에는 없어야 한다).
"""

START_STAMP_REQUIRED: tuple[str, ...] = (
    "tag", "seed_index", "mode", "generation_sha256", "eval_list_file_sha256",
    "train_run_id", "train_ledger_sha256", "started_at", "device", "purpose",
)
"""시작 도장의 필수 열 — **모델 모드**(§26-3, 승인 · `purpose` 는 07번 §30-4 가 더했다).

`tag`·`seed_index` 는 §14-1 다, `mode` 는 §14-1 나, `generation_sha256`·목록 sha 는 §13-3 바-15,
원장 참조·시각·장비는 §13-3 나-4 에서 온다. 도장은 **추론 전에** 쓰므로 아홉이 그 시점에 다 있다.

`scored_adapter_sha256` 은 모드에 따라 갈려서 여기 없다 — 모델에는 있어야 하고 에코에는 없어야 한다.
`finished_at` 은 도장에 **없다** — 추론 전에 쓰는 파일이라 종료 시각을 가질 수 없다.
"""

SIDECAR_ECHO_ABSENT: tuple[str, ...] = (
    "train_run_id", "train_ledger_sha256", "train_ledger_path",
)
"""**에코 곁 파일에 있으면 거부**하는 키(§28-3 다). 도장과 같은 까닭이다 —
에코는 모델을 부르지 않고 그 학습에 기대지 않는다.

**원장을 가리키는 키 셋이다.** `train_ledger_path` 는 이름 그대로 원장 참조라 함께 뺀다 —
빼지 않으면 도장은 "학습과 무관" 이라 하고 곁 파일은 원장을 가리켜 같은 묶음이 두 말을 한다.
`train_config_sha256`·`budget_n`·`budget_r`·`budget_e` 는 **원장 참조가 아니라** 그대로 필수다 —
결정의 문장이 "원장 참조" 라 넓히지 않았고, 에코가 그것들을 실어야 하는지는 미정이다(§28-8).
`scored_adapter_sha256` 은 `SIDECAR_REQUIRED` 에 없고 §16-2 I-3 이 따로 본다."""

START_STAMP_ECHO_ABSENT: tuple[str, ...] = (
    "train_run_id", "train_ledger_sha256", "scored_adapter_sha256",
)
"""**에코 도장에 있으면 거부**하는 키(§28-3 가, 승인). 에코는 모델을 부르지 않고 그 학습에 기대지 않는다.
에코 도장의 필수는 열에서 이 셋과 겹치는 둘을 뺀 **여덟**이다."""

EVENT_START = "start"
EVENT_END = "end"

ATTEMPTS_COMMON_REQUIRED: tuple[str, ...] = (
    "event", "tag", "seed_index", "attempt_no", "start_stamp_sha256",
)
"""시도 기록의 **두 줄 모두**에 있어야 하는 키(§27-1 · §27-5)."""

ATTEMPTS_START_REQUIRED: tuple[str, ...] = ("reason", "started_at", "first_id", "device")
"""**시작 줄**만의 필수(§27-1 · §28-3 나). 장비는 시작 줄마다 적는다."""

ATTEMPTS_END_REQUIRED: tuple[str, ...] = ("last_id", "finished_at", "n_written")
"""**끝 줄**만의 필수(§27-1). 끝 줄이 없는 시작 줄이 곧 죽은 시도다 — 아무도 대신 적지 않는다."""

REASON_FIRST = "first_pass"
REASON_RESUME = "resume_after_interruption"
REASON_TORN_TAIL = "retry_torn_tail"

ATTEMPT_REASONS: tuple[str, ...] = (REASON_FIRST, REASON_RESUME, REASON_TORN_TAIL)
"""시작 까닭의 **닫힌 목록**(§27-3, 뜻은 §28-2 가 넓혔다). 끝 상태는 사유에 싣지 않는다 —
끝 줄의 유무가 말한다. 다른 값은 거부한다."""

_FRACTIONAL = re.compile(r"\d\.\d")
"""초 아래 자릿수(§28-4). 오프셋 있는 ISO 의 범위 안이라 새 필드가 아니다."""

_DEVICE = re.compile(r"^(?P<name>[^:]+(?::[^:]+)*):(?P<index>\d+)$")
"""`"{이름}:{인덱스}"`(§28-3 나). 인덱스만으로는 다른 장비가 같은 값이 된다."""

HASH_FIELDS: tuple[str, ...] = (
    "generations_sha256", "eval_list_file_sha256", "eval_list_set_sha256", "start_stamp_sha256",
    "snapshot_digest", "coord_cfg_hash", "generation_sha256", "prompt_sha256", "gen_prefix_sha256",
    "processor_config_sha256", "train_ledger_sha256", "train_config_sha256",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

ENV_FIELDS: tuple[str, ...] = (
    "processor_config_sha256", "processor_min_pixels", "processor_max_pixels",
    "patch_size", "merge_size", "image_grid_thw_observed", "transformers_version",
    "gen_prefix_sha256", "chat_template_kwargs", "padding_side",
)
"""다섯 모델 × 전 시드에서 **같아야 하는** 것. 다르면 서로 다른 조건에서 만든 출력을 한 비교에 넣는 것이다."""


@dataclass(frozen=True)
class BundleFiles:
    """묶음의 파일 경로. 존재 여부는 검증이 본다."""

    generations: Path

    @property
    def stem(self) -> str:
        return self.generations.name[: -len(".generations.jsonl")]

    @property
    def sidecar(self) -> Path:
        return self.generations.with_name(f"{self.stem}.export_meta.json")

    @property
    def start_stamp(self) -> Path:
        return self.generations.with_name(f"{self.stem}.export_start.json")

    @property
    def tokens(self) -> Path:
        return self.generations.with_name(f"{self.stem}.tokens.jsonl")

    @property
    def attempts(self) -> Path:
        return self.generations.parent / "attempts.jsonl"


@dataclass(frozen=True)
class TrainingArtifact:
    """학습 산출물 meta 에서 온 것. **원장에는 어댑터 sha 열이 없다**(§16-1)."""

    adapter_sha256: str
    step: int
    """그 어댑터가 쓰인 스텝·라운드. 원장의 마지막 행과 맞댄다 — 중간 체크포인트를 채점하는 것을 막는다."""
    train_rows_digest: object = None
    """어댑터 meta 의 `train_rows_digest` — 그 칸이 실제로 학습한 행의 digest(리허설 2판 반영판 §2-7). 연합은 참여자별 사전이다.
    **곁 파일에는 없다** — 등록에 값이 있으면 이것과 맞댄다(§34)."""


@dataclass(frozen=True)
class LedgerView:
    """학습 원장에서 필요한 것만. 파일을 읽는 것은 호출부의 일이다."""

    run_id: str
    file_sha256: str
    final_step: int
    cell: str
    client: str | None


@dataclass(frozen=True)
class Registration:
    """등록 객체 가운데 **생성 쪽** 값. 채점 쪽은 여기 없다(지문이 둘로 나뉜다, §12-5)."""

    generation_sha256: str
    coupling_rule: str
    seeds: Mapping[int, int]
    """시드 번호 → 시드 값. 파일 이름의 번호가 여기 있어야 한다."""
    values: Mapping[str, object]
    """곁 파일의 실제 값과 **항목별로** 맞댈 등록값 — **그 묶음의 모드로** `actuals.registered_values(등록, mode=…)` 가 낸 것.
    모드마다 목록 해시가 다르다. 검증은 값의 목적과 목록 해시가 이 객체의 목적 · 묶음의 모드와 같은지 먼저 본다(검수 16번 I-5)."""
    eval_list: EvalList
    purpose: str
    """등록의 목적 — **완화 스위치**다. `impl_ids` 가 어긋난 묶음을 리허설은 "대역 실행" 으로 적고 본실험 · 진단은 거부한다.
    `fault_events` 는 본실험 · 진단에서 빈 목록이어야 한다(07번 §30-4). 값의 `purpose` · 곁 파일 · 도장의 `purpose` 와 같아야 한다 —
    영수증 종류와의 대조는 진입점이 한다(`check_receipt_kind`)."""
    echo_list: EvalList | None = None
    coord_space: str = "ABS_ORIG"
    train_rows_digest: str | None = None
    """등록의 `generation.train_rows_digest`. 값이 있으면 모델 묶음의 어댑터 meta 와 맞댄다."""


_VERIFY_TOKEN = object()
"""`verify_bundle` 만 가진 표. 검증을 지나지 않은 객체가 채점용으로 넘어가는 길을 막는다."""

_ASSEMBLING = [0]
"""조립 창. `verify_bundle` 이 자기 반환값을 만드는 동안에만 0 보다 크다.

**`dataclasses.replace` 를 막는 자리다** — `replace` 는 초기화 인자를 그대로 옮기므로 표를
필드로 두면 그것까지 복사된다. 창으로 두면 밖에서 부른 `replace` 는 창이 닫혀 있어 걸린다.
한 프로세스에서 동시에 두 검증이 돌면 창이 겹칠 수 있다 — **고의의 우회와 같은 범주**로 본다.
막는 것은 실수로 검증을 건너뛰거나 검증된 객체를 바꿔 끼우는 것이다.
"""


class _Frozen(dict):
    """바꿀 수 없는 사전. **`dict` 를 상속하는 까닭은 읽는 쪽 때문이다**(§26-6).

    검증 뒤 판정에 쓰는 내용은 원본과 떼어져 있고 **안쪽까지** 바뀌지 않아야 한다. 그런데
    `MappingProxyType` 로 싸면 `json.dumps` 가 받지 못해 이미 그 값을 읽는 자리(묶음 사이
    환경 일치 판정·산출물 직렬화)가 깨진다. `dict` 의 하위형이면 둘 다 선다 — 바꾸는 연산은
    막히고 읽는 쪽은 그대로 읽는다.
    """

    __slots__ = ()

    def _frozen(self, *_a, **_k):
        raise TypeError(
            "검증을 지난 곁 파일 내용은 바꿀 수 없다 — 검증한 값과 판정에 쓰는 값이 "
            "달라지는 길을 막는다."
        )

    __setitem__ = _frozen
    __delitem__ = _frozen
    __ior__ = _frozen
    clear = _frozen
    pop = _frozen
    popitem = _frozen
    setdefault = _frozen
    update = _frozen


def _deep_freeze(value):
    """사전은 `_Frozen` 으로, 목록은 튜플로 — **겹쳐 든 값까지** 바꿀 수 없게 한다(§26-6).

    겉만 복사하면 안쪽은 같은 객체를 가리킨다. `ENV_FIELDS` 의 `chat_template_kwargs`·
    `image_grid_thw_observed` 가 값이 사전이라 그 자리가 실제로 있다.
    """
    if isinstance(value, dict):
        return _Frozen((k, _deep_freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class VerifiedBundle:
    """검증을 지난 묶음. **여기 담긴 것만 채점에 쓴다.**

    **`verify_bundle` 밖에서는 만들 수 없고, 만든 뒤에는 바뀌지 않는다**(§26-6).
    진입점이 직접 조립하면 검증을 건너뛴 묶음이 채점에 들어가고, 검증 뒤에 내용이 바뀌면
    검증한 값과 판정에 쓰는 값이 달라진다. 세 길을 함께 막는다.

    | 길 | 막는 것 |
    |---|---|
    | 조립 · `replace` | 조립 창(`_ASSEMBLING`) — `verify_bundle` 안에서만 열린다 |
    | 속성 대입 | `frozen=True` |
    | 안쪽 변경 | 줄·식별자는 **튜플**, 곁 파일 내용과 알림은 **깊이 동결**(`_deep_freeze`) |

    진입점 구현은 아직 없다(§21-1) — 그 자리가 서기 전에 이 관문을 먼저 건다.
    """

    files: BundleFiles
    tag: str
    cell: str
    client: str | None
    seed_index: int
    seed_value: int
    mode: str
    sidecar: Mapping
    lines: tuple[str, ...]
    image_ids: tuple[str, ...]
    notes: Mapping = field(default_factory=dict)
    token: object = field(default=None, init=False, repr=False, compare=False)
    """조립이 성공하면 `verify_bundle` 의 표가 들어간다. 초기화 인자가 아니다."""

    def __post_init__(self) -> None:
        if not _ASSEMBLING[0]:
            raise TypeError(
                "VerifiedBundle 은 `verify_bundle` 만 만든다 — 검증을 지나지 않은 묶음을 "
                "채점에 넣지 않는다. 검증된 객체를 바꿔 끼우는 것(`dataclasses.replace`)도 "
                "이 관문이 막는다."
            )
        # `frozen` 이라 대입은 `object.__setattr__` 로만 된다.
        object.__setattr__(self, "token", _VERIFY_TOKEN)
        object.__setattr__(self, "sidecar", _deep_freeze(dict(self.sidecar)))
        object.__setattr__(self, "notes", _deep_freeze(dict(self.notes)))
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "image_ids", tuple(self.image_ids))

    @property
    def is_echo(self) -> bool:
        return self.mode == MODE_ECHO


def require_verified(obj) -> VerifiedBundle:
    """채점 층의 첫 문장이 부른다(진입점 미니스펙 3판 1-2 나). **`verify_bundle` 이 만든 객체**가 아니면 `TypeError`.

    타입 표기만으로는 런타임에 아무것도 막지 못한다. 표(`_VERIFY_TOKEN`)는 이 모듈 안에만 있어 밖에서 얻을 수 없다.
    """
    if not isinstance(obj, VerifiedBundle) or obj.token is not _VERIFY_TOKEN:
        raise TypeError("채점 층은 verify_bundle 을 지난 묶음(VerifiedBundle)만 받는다 — "
                        f"받은 것: {type(obj).__name__}")
    return obj


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _aware(value) -> datetime | None:
    """오프셋 있는 ISO 시각만 받는다. naive 는 9시간 창에서 거짓 통과를 만든다."""
    if not isinstance(value, str):
        return None
    try:
        t = datetime.fromisoformat(value)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _check_file_bytes(raw: bytes, path: Path, out: list[Rejection], *, jsonl: bool) -> str | None:
    """§14-1 가의 바이트 규약. **다섯 파일 종류 전부**가 이것을 지난다.

    공통은 셋이다 — BOM 없음 · CR 바이트 없음 · UTF-8. 여기까지는 JSON 문서와 JSONL 이 같다.

    **`jsonl` 일 때만** 끝 개행과 빈 줄 금지를 더한다. JSON 문서(곁 파일·시작 도장)에 끝 개행을
    요구하면 규약에 없는 것을 만드는 것이고, 한 줄짜리 문서에 "빈 줄 금지" 는 뜻이 없다.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        out.append(Rejection(RejectCode.BOM_IN_FILE, path.name))
        return None
    if b"\r" in raw:
        out.append(Rejection(RejectCode.CR_IN_FILE, path.name, {"n_cr": raw.count(b"\r")}))
        return None
    if jsonl and raw and not raw.endswith(b"\n"):
        out.append(Rejection(RejectCode.TORN_TAIL, path.name))
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        out.append(Rejection(RejectCode.NOT_UTF8, path.name, {"at": exc.start}))
        return None
    if jsonl:
        for i, line in enumerate(_split_lf(text), 1):
            if not line.strip():
                out.append(Rejection(RejectCode.BLANK_LINE, f"{path.name}:{i}"))
                return None
    return text


def _split_lf(text: str) -> list[str]:
    """LF 로만 나눈다. `str.splitlines()` 는 U+2028·U+0085 에서도 쪼갠다."""
    return text[:-1].split("\n") if text else []


def _read_lines(raw: bytes, path: Path, out: list[Rejection]) -> list[str] | None:
    """JSONL 한 파일을 바이트 규약에 걸고 줄로 나눈다."""
    text = _check_file_bytes(raw, path, out, jsonl=True)
    return None if text is None else _split_lf(text)


def _check_sidecar_schema(meta, path: Path, out: list[Rejection]) -> bool:
    if not isinstance(meta, dict):
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"reason": "객체가 아니다"}))
        return False
    # **에코 곁 파일에도 학습 원장 참조가 없어야 한다**(2026-09-30 결정) — 도장에서 그렇게
    # 정했고 같은 묶음이 두 말을 하지 않게 한다. 에코는 학습 검사를 건너뛰므로 그 값은
    # 아무것과도 맞대지지 않는다.
    echo = meta.get("mode") == MODE_ECHO
    want = tuple(k for k in SIDECAR_REQUIRED
                 if not (echo and k in SIDECAR_ECHO_ABSENT))
    missing = [k for k in want if k not in meta]
    if echo:
        present = [k for k in SIDECAR_ECHO_ABSENT if k in meta]
        if present:
            out.append(Rejection(RejectCode.ECHO_ADAPTER_PRESENT, path.name,
                                 {"present": present}))
    if missing:
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"missing": missing}))
    bad_hash = [k for k in HASH_FIELDS
                if k in meta and not (isinstance(meta[k], str) and _HEX64.match(meta[k]))]
    if bad_hash:
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"not_sha256": bad_hash}))
    if meta.get("mode") not in (MODE_MODEL, MODE_ECHO):
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"mode": meta.get("mode")}))
    for k in ("n_lines", "seed_index", "seed_value", "max_new_tokens", "batch_size", "attempts"):
        if k in meta and (type(meta[k]) is not int or meta[k] < 0):
            out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"not_int": k}))
    # 07번 §30-4 의 넷 — 꼴만 여기서 본다. 값의 대조는 D 단계다.
    if "purpose" in meta and meta["purpose"] not in PURPOSES:
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"not_vocab": "purpose"}))
    if "impl_ids" in meta and impl_identifiers(meta["impl_ids"]) is None:
        # 쓰는 쪽의 꼴 — 식별자 사전(`actuals.IMPL_FIELDS`)의 목록, 또는 이음새 이름 → 식별자 사전(리허설 2판 반영판 §1-4)
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"not_impl_ids": "impl_ids"}))
    if "fault_events" in meta and not isinstance(meta["fault_events"], list):
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"not_list": "fault_events"}))
    if "train_config" in meta and not (isinstance(meta["train_config"], str) and meta["train_config"]):
        out.append(Rejection(RejectCode.SIDECAR_SCHEMA, path.name, {"not_str": "train_config"}))
    return not out


def verify_bundle(
    files: BundleFiles,
    *,
    registration: Registration,
    artifact: TrainingArtifact | None,
    ledger: LedgerView | None,
    scoring_population: Sequence[str] | None = None,
    group_of: Mapping[str, str] | None = None,
    seen_adapters: Mapping[str, tuple[str, int]] | None = None,
    receipt_time: datetime | None = None,
    image_sha256_of: Mapping[str, str] | None = None,
) -> VerifiedBundle:
    """묶음 하나를 검증한다. 통과하면 채점에 쓸 것만 담아 돌려주고, 아니면 **사유를 전부 모아** 거부한다.

    Args:
        artifact: 학습 산출물 meta. 에코에는 `None` 이어야 한다.
        ledger: 학습 원장에서 뽑은 것. 에코에는 `None` 이어야 한다.
        scoring_population: 채점 모집단. 생성 목록의 부분집합이어야 한다(§13-2 가).
        seen_adapters: 이미 본 `어댑터 sha → (태그, 시드 번호)`. 다섯 모델 × 시드 안의 중복을 잡는다.
        image_sha256_of: 동결 매니페스트의 `image_id → 이미지 파일 sha256`. 내보내기가 **그 그림을 읽었는지** 본다.
        receipt_time: 영수증이 적은 등록 시각. `started_at` 이 이보다 이르면 등록 전 생성이다.

    Raises:
        BundleRejected: 단계별 사유를 모두 담아서.
    """
    rej: list[Rejection] = []

    # ── A 존재·스키마 ────────────────────────────────────────────────────────
    m = _NAME.match(files.generations.name)
    if m is None or m["tag"] not in UNI_MAIN_TAGS:
        rej.append(Rejection(RejectCode.TAG_MISMATCH, files.generations.name,
                             {"reason": "파일 이름이 계약의 꼴이 아니다"}))
        raise BundleRejected(rej)
    name_tag, name_seed = m["tag"], int(m["seed"])
    name_mode = MODE_ECHO if m["echo"] else MODE_MODEL

    for p, code in ((files.generations, RejectCode.SIDECAR_MISSING),
                    (files.sidecar, RejectCode.SIDECAR_MISSING),
                    (files.start_stamp, RejectCode.START_STAMP_MISSING),
                    (files.attempts, RejectCode.ATTEMPTS_MISSING)):
        if not p.exists():
            rej.append(Rejection(code, p.name, {"reason": "없다"}))
    if receipt_time is None:
        rej.append(Rejection(RejectCode.RECEIPT_MISSING, files.stem,
                             {"reason": "등록 영수증 없이 채점하지 않는다"}))
    if rej:
        raise BundleRejected(rej)

    # 곁 파일도 **바이트 규약을 먼저** 지난다. `read_text` 는 CRLF 를 조용히 정규화해
    # 위반을 삼킨다 — 그러면 계약이 말하는 거부가 일어나지 않는다(§14-1 가).
    # 사유 코드는 B 단계의 것이지만 곁 파일은 A 단계에서 해석해야 하므로 여기서 건다.
    side_text = _check_file_bytes(files.sidecar.read_bytes(), files.sidecar, rej, jsonl=False)
    if side_text is None:
        raise BundleRejected(rej)
    try:
        meta = json.loads(side_text)
    except (json.JSONDecodeError, ValueError):
        raise BundleRejected([Rejection(RejectCode.SIDECAR_SCHEMA, files.sidecar.name,
                                        {"reason": "읽을 수 없다"})]) from None
    if not _check_sidecar_schema(meta, files.sidecar, rej):
        raise BundleRejected(rej)

    mode = meta["mode"]
    has_tokens_sha = "tokens_sha256" in meta and meta["tokens_sha256"] is not None
    if has_tokens_sha != files.tokens.exists():
        rej.append(Rejection(RejectCode.TOKENS_FILE_MISMATCH, files.tokens.name,
                             {"sha_in_sidecar": has_tokens_sha, "file_exists": files.tokens.exists()}))
    if mode != name_mode:
        rej.append(Rejection(RejectCode.MODE_MISMATCH, files.stem,
                             {"name": name_mode, "sidecar": mode}))
    if rej:
        raise BundleRejected(rej)

    # ── B 바이트 ─────────────────────────────────────────────────────────────
    # 파일마다 **한 번** 읽는다 — 해시 · 바이트 규약 · 내용이 같은 바이트 위에서 돈다(§14-2 의 5, 검수 16번 M-8).
    raw = files.generations.read_bytes()
    stamp_raw = files.start_stamp.read_bytes()
    attempts_raw = files.attempts.read_bytes()
    tokens_raw = files.tokens.read_bytes() if has_tokens_sha else None
    if _sha(raw) != meta["generations_sha256"]:
        rej.append(Rejection(RejectCode.GENERATIONS_HASH_MISMATCH, files.generations.name))
    if _sha(stamp_raw) != meta["start_stamp_sha256"]:
        rej.append(Rejection(RejectCode.START_STAMP_HASH_MISMATCH, files.start_stamp.name))
    if tokens_raw is not None and _sha(tokens_raw) != meta["tokens_sha256"]:
        rej.append(Rejection(RejectCode.TOKENS_HASH_MISMATCH, files.tokens.name))
    if rej:
        raise BundleRejected(rej)

    # 다섯 파일 종류 전부에 바이트 규약을 건다(§14-1 가). JSON 문서와 JSONL 의 개행 규칙은 가른다.
    start_text = _check_file_bytes(stamp_raw, files.start_stamp, rej, jsonl=False)
    attempts_text = _check_file_bytes(attempts_raw, files.attempts, rej, jsonl=True)
    if tokens_raw is not None:
        _check_file_bytes(tokens_raw, files.tokens, rej, jsonl=True)
    if rej:
        raise BundleRejected(rej)

    lines = _read_lines(raw, files.generations, rej)
    if lines is None:
        raise BundleRejected(rej)

    want_list = registration.echo_list if mode == MODE_ECHO else registration.eval_list
    if want_list is None:
        rej.append(Rejection(RejectCode.ECHO_LIST_MISMATCH, files.stem,
                             {"reason": "에코 목록이 등록돼 있지 않다"}))
        raise BundleRejected(rej)
    if len({len(lines), meta["n_lines"], len(want_list)}) != 1:
        rej.append(Rejection(RejectCode.LINE_COUNT_MISMATCH, files.generations.name,
                             {"actual": len(lines), "sidecar": meta["n_lines"], "list": len(want_list)}))
        raise BundleRejected(rej)

    # ── C 내용 ───────────────────────────────────────────────────────────────
    image_ids: list[str] = []
    for i, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, RecursionError, ValueError):
            rej.append(Rejection(RejectCode.LINE_SCHEMA, f"{files.generations.name}:{i}",
                                 {"reason": "JSON 이 아니다"}))
            break
        if not isinstance(row, dict) or not isinstance(row.get("image_id"), str):
            rej.append(Rejection(RejectCode.LINE_SCHEMA, f"{files.generations.name}:{i}",
                                 {"reason": "image_id 가 없다"}))
            break
        image_ids.append(row["image_id"])
    if rej:
        raise BundleRejected(rej)

    if len(set(image_ids)) != len(image_ids):
        rej.append(Rejection(RejectCode.DUPLICATE_IMAGE_ID, files.generations.name,
                             {"n_duplicate": len(image_ids) - len(set(image_ids))}))
    elif image_ids != list(want_list.ids):
        first = next((i for i, (a, b) in enumerate(zip(image_ids, want_list.ids, strict=True), 1)
                      if a != b), None)
        rej.append(Rejection(RejectCode.ID_SEQUENCE_MISMATCH, files.generations.name,
                             {"first_diff_line": first,
                              "same_set": set(image_ids) == want_list.id_set}))

    if want_list.file_sha256 != meta["eval_list_file_sha256"] or \
            want_list.set_sha256 != meta["eval_list_set_sha256"]:
        rej.append(Rejection(RejectCode.ECHO_LIST_MISMATCH if mode == MODE_ECHO
                             else RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem,
                             {"item": "eval_list"}))

    _check_line_env(lines, meta, registration, files, rej)
    _check_line_facts(lines, files, image_sha256_of, rej)
    if tokens_raw is not None:
        _check_tokens(tokens_raw, lines, files, rej)
    if scoring_population is not None and mode != MODE_ECHO:
        _check_population(scoring_population, want_list, group_of, files, rej)
    if rej:
        raise BundleRejected(rej)

    # ── D 등록·원장 ──────────────────────────────────────────────────────────
    # 완화 스위치(목적)를 쓰기 **전에** 등록 투영이 스스로와 · 묶음의 모드와 맞는지 본다(검수 16번 I-5).
    _check_projection(registration, mode, want_list, files, rej)
    if rej:
        raise BundleRejected(rej)
    _check_identity(meta, name_tag, name_seed, registration, files, rej)
    _check_start_stamp(start_text, meta, mode, registration, files, rej)
    stamp = _parse_stamp(start_text)
    attempts_notes = _check_attempts(attempts_text, meta, name_tag, name_seed,
                                     want_list, receipt_time, stamp, files, rej)
    value_notes = _check_registration_values(meta, registration, files, rej)
    _check_train_config(meta, files, rej)
    fault_notes = _check_faults(meta, registration, files, rej)
    _check_times(meta, receipt_time, files, rej)
    _check_training(meta, mode, artifact, ledger, seen_adapters, files, rej, registration)
    if rej:
        raise BundleRejected(rej)

    cell, client = tag_parts(name_tag)
    # 조립 창은 **여기서만** 열린다 — 밖에서 부른 `dataclasses.replace` 는 창이 닫혀 걸린다(§26-6).
    _ASSEMBLING[0] += 1
    try:
        return VerifiedBundle(
            files=files, tag=name_tag, cell=cell, client=client,
            seed_index=name_seed, seed_value=meta["seed_value"], mode=mode,
            sidecar=meta, lines=lines, image_ids=image_ids,
            notes={"n_lines": len(lines), "coupling_rule": registration.coupling_rule,
                   "echo_exempt": sorted(c.value for c in ECHO_EXEMPT) if mode == MODE_ECHO else [],
                   **attempts_notes, **value_notes, **fault_notes},
        )
    finally:
        _ASSEMBLING[0] -= 1


def _grid_problem(grid, wh, patch) -> str | None:
    """줄의 관측 격자 하나 — 정수 셋 · 1 이상 · 시간 축 1(정지 이미지) · 패치 크기를 곱하면 그 줄의 입력 크기."""
    if not (isinstance(grid, list) and len(grid) == 3 and all(type(v) is int and v >= 1 for v in grid)):
        return "image_grid_thw 는 1 이상의 정수 셋이다"
    if grid[0] != 1:
        return "image_grid_thw 의 시간 축은 1 이다(정지 이미지)"
    if type(patch) is int and patch > 0 and [grid[2] * patch, grid[1] * patch] != wh:
        return "image_grid_thw × 패치 크기가 그 줄의 model_input_wh 와 다르다"
    return None


def _check_line_env(lines, meta, registration, files, rej) -> None:
    """줄마다 실린 좌표 규약·입력 크기가 곁 파일·등록과 같은가, 그리고 줄 사이에서 같은가.

    줄의 관측 격자(`image_grid_thw`)는 내용 해시에 든다 — 검사하지 않으면 격자만 바꿔 판정 한 번의 열쇠를 바꿀 수 있다.
    그래서 **줄마다 격자가 있어야 하고**(07번 §37 — 격자는 생성 시점에만, 일부 줄만 빠진 묶음도 거부한다) 꼴 · 시간 축 · 입력 크기를 본다.
    곁 파일의 `image_grid_thw_observed` 는 빈도마다 bool 아닌 양의 정수이고 **합계가 전체 줄 수**여야 하며, 줄들의 격자를 센 것과
    **같아야** 한다 — 줄 하나의 격자를 지우고 곁 파일 집계를 맞춰 줄여도 걸린다.
    """
    seen_wh: set[tuple] = set()
    grids: dict[str, int] = {}
    patch = meta.get("patch_size")
    for i, line in enumerate(lines, 1):
        row = json.loads(line)
        if row.get("coord_space") != meta["coord_space"] or \
                row.get("coord_cfg_hash") != meta["coord_cfg_hash"]:
            rej.append(Rejection(RejectCode.COORD_SPACE_MISMATCH, f"{files.generations.name}:{i}"))
            return
        wh = row.get("model_input_wh")
        if not (isinstance(wh, list) and len(wh) == 2 and all(type(v) is int and v > 0 for v in wh)):
            rej.append(Rejection(RejectCode.LINE_SCHEMA, f"{files.generations.name}:{i}",
                                 {"reason": "model_input_wh"}))
            return
        seen_wh.add(tuple(wh))
        if "image_grid_thw" not in row:
            rej.append(Rejection(RejectCode.LINE_SCHEMA, f"{files.generations.name}:{i}",
                                 {"reason": "image_grid_thw 가 없다 — 격자는 줄마다 생성 시점에 싣는다(07번 §37)"}))
            return
        why = _grid_problem(row["image_grid_thw"], wh, patch)
        if why is not None:
            code = RejectCode.LINE_SCHEMA if "정수" in why or "시간 축" in why else RejectCode.MODEL_INPUT_GRID_MISMATCH
            rej.append(Rejection(code, f"{files.generations.name}:{i}", {"reason": why}))
            return
        key = json.dumps(row["image_grid_thw"])
        grids[key] = grids.get(key, 0) + 1
    if meta["coord_space"] != registration.coord_space:
        rej.append(Rejection(RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem,
                             {"item": "coord_space", "sidecar": meta["coord_space"],
                              "registered": registration.coord_space}))
    if len(seen_wh) > 1:
        # 이 규칙은 **단일 크기 모집단의 전제** 위에 있다(§16-2 m-1). 크기가 섞인 모집단에는 그대로 쓰지 않는다.
        rej.append(Rejection(RejectCode.MODEL_INPUT_WH_MISMATCH, files.generations.name,
                             {"observed": sorted(seen_wh)}))
        return
    observed = meta.get("image_grid_thw_observed")
    if isinstance(observed, Mapping) and not (all(type(v) is int and v > 0 for v in observed.values())
                                              and sum(observed.values()) == len(lines)):
        rej.append(Rejection(RejectCode.MODEL_INPUT_GRID_MISMATCH, files.stem,
                             {"reason": "곁 파일의 image_grid_thw_observed 는 빈도마다 양의 정수이고 합계가 전체 줄 수다",
                              "n_lines": len(lines), "sidecar": dict(observed)}))
        return
    if isinstance(observed, Mapping) and dict(observed) != grids:
        rej.append(Rejection(RejectCode.MODEL_INPUT_GRID_MISMATCH, files.stem,
                             {"reason": "줄의 관측 격자를 센 것이 곁 파일의 image_grid_thw_observed 와 다르다",
                              "in_lines": grids, "sidecar": dict(observed)}))
        return
    if seen_wh:
        _check_grid(next(iter(seen_wh)), meta, files, rej)


def _check_line_facts(lines, files, image_sha256_of, rej) -> None:
    """줄이 말하는 사실 둘 — 생성이 어떻게 끝났는가, 어느 그림을 읽었는가."""
    n_undetermined = 0
    n_bad_image = 0
    first_bad = None
    for i, line in enumerate(lines, 1):
        row = json.loads(line)
        if row.get("gen_stop") == "stop_undetermined":
            n_undetermined += 1
        if image_sha256_of is not None:
            want = image_sha256_of.get(row.get("image_id"))
            got = row.get("image_sha256")
            if want is None or got != want:
                n_bad_image += 1
                first_bad = first_bad or i
    if n_undetermined:
        # 중단이 아니라 거부다 — 원인을 고친 뒤 그 파일만 다시 만들 수 있다(§16-2 m-2).
        rej.append(Rejection(RejectCode.STOP_UNDETERMINED, files.generations.name,
                             {"n_lines": n_undetermined}))
    if n_bad_image:
        rej.append(Rejection(RejectCode.IMAGE_BYTES_MISMATCH, files.generations.name,
                             {"n_lines": n_bad_image, "first_line": first_bad}))


def _check_grid(wh: tuple[int, int], meta, files, rej) -> None:
    """`model_input_wh` 가 관측 격자 × 패치에서 나온 값인가 — 문자 일치의 동어반복을 막는 독립 닻."""
    grid = meta.get("image_grid_thw_observed")
    patch = meta.get("patch_size")
    if not (isinstance(grid, Mapping) and len(grid) == 1 and type(patch) is int and patch > 0):
        rej.append(Rejection(RejectCode.MODEL_INPUT_GRID_MISMATCH, files.stem,
                             {"reason": "관측 격자가 한 가지가 아니거나 패치 크기가 없다",
                              "n_grid": len(grid) if isinstance(grid, Mapping) else None}))
        return
    key = next(iter(grid))
    try:
        _, h, w = (int(v) for v in json.loads(key)) if key.startswith("[") else (0, 0, 0)
    except (ValueError, TypeError):
        rej.append(Rejection(RejectCode.MODEL_INPUT_GRID_MISMATCH, files.stem,
                             {"reason": "격자 표기를 읽을 수 없다"}))
        return
    if (w * patch, h * patch) != wh:
        rej.append(Rejection(RejectCode.MODEL_INPUT_GRID_MISMATCH, files.stem,
                             {"from_grid": [w * patch, h * patch], "in_lines": list(wh)}))


def _check_population(population, want_list: EvalList, group_of, files, rej) -> None:
    """채점 모집단이 생성 목록의 부분집합인가, 그리고 덩어리를 쪼개 담지 않았는가.

    **완전성의 기준은 "덩어리의 전체 구성원" 이 아니라 "생성 목록에 남아 있는 구성원" 이다**(§18-5).
    근사 중복 제외는 **영상 단위**라 덩어리가 정당하게 쪼개진다(평가 묶음 769/1,245 가 걸린다) —
    전체 구성원을 요구하면 정상 모집단이 거부된다. 막으려는 것은 남은 것 가운데 일부만 골라 담는 쪽이다.

    `group_of` 는 묶음이 아니라 **연결 성분**을 가리킨다(§18-4) — 같은 묶음과 근사 중복 쌍을 함께 이은 단위다.
    """
    pop = set(population)
    if not pop <= want_list.id_set:
        rej.append(Rejection(RejectCode.POPULATION_NOT_SUBSET, files.stem,
                             {"n_outside": len(pop - want_list.id_set)}))
        return
    if group_of is None:
        return
    survivors: dict[str, set[str]] = {}
    for image_id in want_list.ids:
        cluster = group_of.get(image_id)
        if cluster is not None:
            survivors.setdefault(cluster, set()).add(image_id)
    incomplete = [c for c in {group_of[i] for i in pop if i in group_of}
                  if not survivors.get(c, set()) <= pop]
    if incomplete:
        rej.append(Rejection(RejectCode.POPULATION_GROUP_INCOMPLETE, files.stem,
                             {"n_incomplete_clusters": len(incomplete)}))


def _check_identity(meta, name_tag, name_seed, registration, files, rej) -> None:
    if meta["tag"] != name_tag:
        rej.append(Rejection(RejectCode.TAG_MISMATCH, files.stem,
                             {"name": name_tag, "sidecar": meta["tag"]}))
    cell, client = tag_parts(name_tag)
    if meta["cell"] != cell or meta["client"] != client:
        rej.append(Rejection(RejectCode.TAG_MISMATCH, files.stem,
                             {"reason": "칸·참여자가 태그와 다르다"}))
    if meta["seed_index"] != name_seed:
        rej.append(Rejection(RejectCode.SEED_MAPPING_MISMATCH, files.stem,
                             {"name": name_seed, "sidecar": meta["seed_index"]}))
    if name_seed not in registration.seeds:
        rej.append(Rejection(RejectCode.SEED_NOT_REGISTERED, files.stem, {"seed_index": name_seed}))
    elif registration.seeds[name_seed] != meta["seed_value"]:
        rej.append(Rejection(RejectCode.SEED_MAPPING_MISMATCH, files.stem,
                             {"registered": registration.seeds[name_seed],
                              "sidecar": meta["seed_value"]}))


def _check_projection(registration: Registration, mode: str, want_list: EvalList, files, rej) -> None:
    """등록 투영의 목적 · 모드 결속. 목적은 대역 면제와 고의 중단 통과를 정하는 스위치라 **값과 같은 등록에서** 나와야 한다."""
    v = registration.values
    why = []
    if registration.purpose not in PURPOSES:
        why.append("목적이 어휘 밖이다")
    if v.get("purpose") != registration.purpose:
        why.append("값의 purpose 와 목적이 다르다")
    if "impl_ids" not in v:
        why.append("값에 impl_ids 가 없다 — 대역 실행을 가를 수 없다")
    if v.get("eval_list_file_sha256") != want_list.file_sha256 or v.get("eval_list_set_sha256") != want_list.set_sha256:
        why.append(f"값의 목록 해시가 {mode} 묶음의 목록과 다르다 — 다른 모드로 만든 투영이다")
    if why:
        rej.append(Rejection(RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem,
                             {"item": "registration_projection", "reason": " · ".join(why)}))


def _check_registration_values(meta, registration: Registration, files, rej) -> dict:
    """곁 파일의 **실제 값**과 등록값을 항목별로 맞댄다. 자료형까지 본다 — `256` 과 `"256"` 은 다르다.

    비교는 쓰는 쪽의 시작 전 점검과 **같은 함수**(`actuals.compare_values`)다. `impl_ids` 가 어긋나면 본실험 ·
    진단은 그 칸을 거부하고, 리허설은 거부하지 않고 **"대역 실행"** 으로 적는다(07번 §30-4) — 통과를 내지 않는 것은
    진입점의 판정이다. 돌려주는 사전은 산출의 알림에 들어간다.
    """
    if meta["generation_sha256"] != registration.generation_sha256:
        rej.append(Rejection(RejectCode.GENERATION_FINGERPRINT_MISMATCH, files.stem))
        return {}
    stand_in = False
    for m in compare_values(registration.values, meta):
        if m.item == "impl_ids" and registration.purpose == "rehearsal":
            stand_in = True
            continue
        rej.append(Rejection(RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem, m.as_detail()))
    # 열쇠가 같아도 **승인된 실제 구현**이어야 한다 — 승인 표시 · 저장소 안 · 커밋된 코드(리허설 2판 반영판 §1-4)
    problems = impl_problems(meta["impl_ids"])
    if problems:
        if registration.purpose == "rehearsal":
            stand_in = True
        else:
            rej.append(Rejection(RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem,
                                 {"item": "impl_ids", "reason": "승인된 실제 구현이 아니다", "problems": problems}))
    return {"stand_in": stand_in}


TRAIN_CONFIG_MAP: tuple[tuple[str, str], ...] = (
    ("prompt_sha256", "prompt_sha256"), ("template_mode", "chat_template_kwargs"),
    ("coord_cfg_hash", "coord_cfg_hash"), ("model_id", "base_model_id"), ("model_revision", "base_model_revision"),
    ("num_rounds", "budget_r"), ("local_epochs", "budget_e"), ("total_epochs", "budget_n"),
    ("init_adapter_digest", "init_adapter_digest"), ("processor_config_sha256", "processor_config_sha256"),
)
"""`train_config` 안의 키(학습) → 곁 파일의 키(생성). **쓰는 쪽의 대응 표 그대로다**(리허설 미니스펙 2판 반영판 2-5).
짝이 없는 키(`pairs_digest` · `lora` · `optimizer` …)는 대조하지 않는다 — `train_config_sha256` 안에서만 고정된다."""


def template_mode_of(chat_template_kwargs: Mapping) -> str:
    """`chat_template_kwargs` → `template_mode` 의 식(대응 표). `{"enable_thinking": False}` → `"enable_thinking=False"`."""
    return ",".join(f"{k}={v!r}" for k, v in sorted(chat_template_kwargs.items()))


def _check_train_config(meta, files, rej) -> None:
    """학습 설정 원문 — 정규형이고, 다시 해시하면 `train_config_sha256` 이고, 원문 안의 값이 대응 표대로 곁 파일과 같다."""
    text = meta["train_config"]
    try:
        cfg = json.loads(text)
    except (json.JSONDecodeError, RecursionError, ValueError):
        cfg = None
    if not isinstance(cfg, dict) or \
            json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")) != text:
        rej.append(Rejection(RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem,
                             {"item": "train_config", "reason": "정규화 JSON 원문이 아니다"}))
        return
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != meta["train_config_sha256"]:
        rej.append(Rejection(RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem,
                             {"item": "train_config", "reason": "다시 해시한 값이 train_config_sha256 과 다르다"}))
    bad: list[str] = []
    for key, side in TRAIN_CONFIG_MAP:
        if key not in cfg or side not in meta:
            bad.append(key)
            continue
        if key == "template_mode":
            ckw = meta[side]
            if not (isinstance(ckw, Mapping) and isinstance(cfg[key], str) and cfg[key] == template_mode_of(ckw)):
                bad.append(key)
        elif type(cfg[key]) is not type(meta[side]) or cfg[key] != meta[side]:
            bad.append(key)
    if bad:
        rej.append(Rejection(RejectCode.REGISTRATION_ITEM_MISMATCH, files.stem,
                             {"item": "train_config", "reason": "대응 표의 값이 곁 파일과 다르다",
                              "keys": sorted(bad)}))


def _check_faults(meta, registration: Registration, files, rej) -> dict:
    """`fault_events` — 본실험 · 진단은 빈 목록이어야 한다. 리허설은 기록을 그대로 알림에 옮긴다(07번 §30-4)."""
    events = meta["fault_events"]
    if registration.purpose == "rehearsal":
        return {"fault_events": list(events)}
    if events:
        rej.append(Rejection(RejectCode.FAULT_EVENTS_PRESENT, files.stem,
                             {"purpose": registration.purpose, "n_events": len(events)}))
    return {}


def _check_tokens(raw: bytes, lines, files, rej) -> None:
    """토큰 파일 — 줄 수 == 생성 줄 수 · id 수열이 같다 · 줄마다 `len(token_ids) == len(token_logprobs) == n_new_tokens`.

    등록 지표는 이 파일을 쓰지 않는다. 그래도 곁 파일이 해시를 싣는 한 그 파일이 생성 파일과 짝인지 본다(3판 1-8).
    """
    where = files.tokens.name
    tok_lines = raw.decode("utf-8").rstrip("\n").split("\n") if raw else []
    if len(tok_lines) != len(lines):
        rej.append(Rejection(RejectCode.TOKENS_CONTENT_MISMATCH, where,
                             {"reason": "줄 수", "tokens": len(tok_lines), "generations": len(lines)}))
        return
    for i, (tl, gl) in enumerate(zip(tok_lines, lines, strict=True), 1):
        try:
            t, g = json.loads(tl), json.loads(gl)
        except (json.JSONDecodeError, RecursionError, ValueError):
            rej.append(Rejection(RejectCode.TOKENS_CONTENT_MISMATCH, f"{where}:{i}", {"reason": "JSON 이 아니다"}))
            return
        if not isinstance(t, dict) or t.get("image_id") != g.get("image_id"):
            rej.append(Rejection(RejectCode.TOKENS_CONTENT_MISMATCH, f"{where}:{i}", {"reason": "id 수열"}))
            return
        ids, lps, n = t.get("token_ids"), t.get("token_logprobs"), g.get("n_new_tokens")
        if not (isinstance(ids, list) and isinstance(lps, list) and type(n) is int
                and len(ids) == len(lps) == n):
            rej.append(Rejection(RejectCode.TOKENS_CONTENT_MISMATCH, f"{where}:{i}", {"reason": "길이"}))
            return


def _hex_ok(value) -> bool:
    return isinstance(value, str) and _HEX64.match(value) is not None


def _sub_second(value) -> bool:
    """초 아래 자릿수가 있는가(§28-4). 문자열의 꼴을 본다 — 파싱한 뒤에는 0 과 구별되지 않는다."""
    return isinstance(value, str) and _FRACTIONAL.search(value) is not None


def _device_ok(value) -> bool:
    """`"{이름}:{인덱스}"`(§28-3 나)."""
    return isinstance(value, str) and _DEVICE.match(value) is not None


def _strict_int(value) -> bool:
    """`True == 1`·`1.0 == 1` 을 막는다(19번 Minor 5-3). 곁 파일 검사와 같은 규칙으로 둔다."""
    return type(value) is int


def _check_start_stamp(text: str, meta, mode: str, registration, files, rej) -> None:
    """시작 도장의 **내용**을 곁 파일·등록값과 맞댄다.

    바이트 해시는 "곁 파일이 가리키는 그 파일" 까지만 말한다. 다른 묶음의 도장을 놓고 그
    해시를 곁 파일에 적으면 해시 검사는 통과하고, 빈 도장(`{"ok": true}`)도 통과한다 —
    검사할 필드가 처음부터 없기 때문이다. §14-1 나·다 · §13-3 나-4·바-15 · §26-3 · §28-3 을 세운다.
    """
    where = files.start_stamp.name
    try:
        stamp = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        rej.append(Rejection(RejectCode.START_STAMP_SCHEMA, where, {"reason": "JSON 이 아니다"}))
        return
    if not isinstance(stamp, dict):
        rej.append(Rejection(RejectCode.START_STAMP_SCHEMA, where, {"reason": "객체가 아니다"}))
        return

    # 필수 항목은 **모드로 갈린다**(§28-3 가) — 에코는 그 학습에 기대지 않는다.
    want = tuple(k for k in START_STAMP_REQUIRED
                 if not (mode == MODE_ECHO and k in START_STAMP_ECHO_ABSENT))
    missing = [k for k in want if k not in stamp]
    if missing:
        rej.append(Rejection(RejectCode.START_STAMP_SCHEMA, where, {"missing": missing}))
        return

    if mode == MODE_ECHO:
        # **키가 있으면 값이 비어도 있는 것이다**(2026-09-30 결정). 쓰는 쪽은 키를 쓰지 않으면 된다.
        present = [k for k in START_STAMP_ECHO_ABSENT if k in stamp]
        if present:
            rej.append(Rejection(RejectCode.ECHO_ADAPTER_PRESENT, where, {"present": present}))
    elif not _hex_ok(stamp.get("scored_adapter_sha256")):
        rej.append(Rejection(RejectCode.START_STAMP_SCHEMA, where,
                             {"missing": ["scored_adapter_sha256"], "reason": "모델 모드"}))
    elif stamp["scored_adapter_sha256"] != meta.get("scored_adapter_sha256"):
        rej.append(Rejection(RejectCode.START_STAMP_MISMATCH, where,
                             {"item": "scored_adapter_sha256"}))

    if not _strict_int(stamp["seed_index"]):
        rej.append(Rejection(RejectCode.START_STAMP_SCHEMA, where, {"not_int": "seed_index"}))
    stamp_at = _aware(stamp["started_at"])
    if stamp_at is None:
        # 나-4·§26-3 — 시도 줄과 곁 파일만 보고 도장을 보지 않으면 오프셋 없는 도장이 통과한다.
        rej.append(Rejection(RejectCode.TIME_NAIVE, where,
                             {"item": "started_at", "reason": "오프셋 있는 ISO 여야 한다"}))
    elif not _sub_second(stamp["started_at"]):
        rej.append(Rejection(RejectCode.START_STAMP_SCHEMA, where,
                             {"item": "started_at", "reason": "초 아래 자릿수가 없다"}))
    if not _device_ok(stamp["device"]):
        rej.append(Rejection(RejectCode.START_STAMP_SCHEMA, where,
                             {"item": "device", "reason": "이름과 번호를 함께 적는다"}))

    for key in ("tag", "seed_index", "mode", "purpose"):
        if stamp[key] != meta[key]:
            rej.append(Rejection(RejectCode.START_STAMP_MISMATCH, where,
                                 {"item": key, "stamp": stamp[key], "sidecar": meta[key]}))
    if mode == MODE_MODEL:
        # 도장과 곁 파일의 원장 참조는 같아야 한다 — 07번 §30-9 가 되살렸다(09-30 에 계약에 없어 뺐던 대조).
        for key in ("train_run_id", "train_ledger_sha256"):
            if stamp.get(key) != meta.get(key):
                rej.append(Rejection(RejectCode.START_STAMP_MISMATCH, where, {"item": key}))
    if stamp["generation_sha256"] != registration.generation_sha256:
        rej.append(Rejection(RejectCode.START_STAMP_MISMATCH, where,
                             {"item": "generation_sha256", "reason": "등록값과 다르다"}))
    if stamp["eval_list_file_sha256"] != meta["eval_list_file_sha256"]:
        rej.append(Rejection(RejectCode.START_STAMP_MISMATCH, where,
                             {"item": "eval_list_file_sha256"}))
    # 집합 해시는 §13-3 바-15 가 이름으로 요구하지 않는다 — **있으면** 맞아야 한다(§26-1 ③).
    if "eval_list_set_sha256" in stamp and \
            stamp["eval_list_set_sha256"] != meta["eval_list_set_sha256"]:
        rej.append(Rejection(RejectCode.START_STAMP_MISMATCH, where,
                             {"item": "eval_list_set_sha256"}))


def _parse_stamp(text: str):
    """도장을 한 번 더 읽는다. 꼴이 어긋나면 `None` — 그때는 `_check_start_stamp` 가 이미 거부한다."""
    try:
        stamp = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def _attempt_rows(text: str, meta, name_tag: str, name_seed: int, files, rej):
    """공유 기록에서 **이 묶음의 줄만** 골라 시도 번호로 묶는다(§27-5).

    `attempts.jsonl` 은 폴더당 하나를 열다섯 묶음이 같이 쓴다(§14-1 다). 개명한 옛 시도의 줄이
    같은 태그·시드로 남으므로 **시작 도장의 해시**까지 맞는 줄만 고른다.
    """
    where = files.attempts.name
    stamp_sha = meta["start_stamp_sha256"]
    mine: dict[int, dict[str, dict]] = {}
    n_total = 0
    for i, line in enumerate(_split_lf(text), 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}:{i}",
                                 {"reason": "JSON 이 아니다"}))
            return None, 0
        if not isinstance(row, dict):
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}:{i}",
                                 {"reason": "객체가 아니다"}))
            return None, 0
        gap = [k for k in ATTEMPTS_COMMON_REQUIRED if k not in row]
        if gap:
            # 주인 없는 줄은 **어느 묶음에도** 귀속할 수 없다 — 고르기 전에 거부한다.
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}:{i}", {"missing": gap}))
            return None, 0
        if row["event"] not in (EVENT_START, EVENT_END):
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}:{i}",
                                 {"item": "event", "value": row["event"]}))
            return None, 0
        n_total += 1
        if row["tag"] != name_tag or not _strict_int(row["seed_index"]) or \
                row["seed_index"] != name_seed:
            # `True == 1`·`1.0 == 1` 인 줄이 시드 1 의 줄로 골라지지 않게 한다(19번 Minor 5-3).
            continue
        if row["start_stamp_sha256"] != stamp_sha:
            continue                      # 다른 도장에 딸린 줄 — 지우지 않고 골라지지도 않는다
        if not _strict_int(row["attempt_no"]) or row["attempt_no"] < 1:
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}:{i}",
                                 {"item": "attempt_no"}))
            return None, 0
        slot = mine.setdefault(int(row["attempt_no"]), {})
        if row["event"] in slot:
            rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, f"{where}:{i}",
                                 {"reason": "같은 시도에 같은 종류의 줄이 둘이다",
                                  "attempt_no": row["attempt_no"], "event": row["event"]}))
            return None, 0
        slot[row["event"]] = row
    return mine, n_total


def _check_attempts(text: str, meta, name_tag: str, name_seed: int,
                    want_list: EvalList, receipt_time: datetime, stamp: Mapping | None,
                    files, rej) -> dict:
    """시도 기록의 **두 줄 꼴**을 본다 — 회계·사유·구간·시각·장비(§27-1·§27-3·§27-4·§27-6 · §28).

    값을 아는 쪽이 아는 때에 적는다. **죽은 시도는 끝 줄이 없는 것으로 드러나고 아무도 대신
    적지 않는다** — 그래서 추정한 시각·추정한 끝 id 가 생기지 않는다.

    보고에 쓰는 시도 횟수의 출처는 **곁 파일의 `attempts`** 다(§13-3 바-17). 이 함수는 그 값이
    실제 **시작 줄**의 수와 같은지 대조하는 쪽이고, 둘이 다르면 어느 쪽도 쓰지 않는다.

    Returns:
        보고용 관측값. 거부가 생기면 빈 사전이다.
    """
    where = files.attempts.name
    mine, n_total = _attempt_rows(text, meta, name_tag, name_seed, files, rej)
    if mine is None:
        return {}

    starts = {k: v[EVENT_START] for k, v in mine.items() if EVENT_START in v}
    orphan = sorted(k for k in mine if EVENT_START not in mine[k])
    if orphan:
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"reason": "시작 줄 없는 끝 줄", "attempt_no": orphan}))
        return {}
    if len(starts) != meta["attempts"]:
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"start_rows_for_bundle": len(starts),
                              "sidecar_attempts": meta["attempts"], "rows_total": n_total}))
        return {}
    if not starts:
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"reason": "이 묶음의 시작 줄이 없다", "rows_total": n_total}))
        return {}
    # `attempt_no` 는 **도장마다 1부터 빈틈 없이** 는다(§28-4).
    if sorted(starts) != list(range(1, len(starts) + 1)):
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"reason": "시도 번호가 1부터 이어지지 않는다",
                              "attempt_no": sorted(starts)}))
        return {}

    order = sorted(starts)
    last_no = order[-1]
    if EVENT_END not in mine[last_no]:
        # 곁 파일은 마지막 시도가 스스로 끝난 뒤에만 생긴다(§27-1).
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"reason": "마지막 시도에 끝 줄이 없다", "attempt_no": last_no}))
        return {}

    ids = list(want_list.ids)
    pos = {image_id: i for i, image_id in enumerate(ids)}
    n_lines = len(ids)
    ok = True

    for k in order:
        row = starts[k]
        gap = [f for f in ATTEMPTS_START_REQUIRED if f not in row]
        if gap:
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}", {"missing": gap}))
            ok = False
            continue
        if row["reason"] not in ATTEMPT_REASONS:
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}",
                                 {"item": "reason", "value": row["reason"]}))
            ok = False
        elif (row["reason"] == REASON_FIRST) != (k == 1):
            # 새 도장의 첫 시도는 번호 1 이고 사유가 `first_pass` 다(§28-4).
            rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, f"{where}#{k}",
                                 {"item": "reason", "value": row["reason"], "attempt_no": k}))
            ok = False
        if not _device_ok(row["device"]):
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}",
                                 {"item": "device", "reason": "이름과 번호를 함께 적는다"}))
            ok = False
        if not _sub_second(row["started_at"]) or _aware(row["started_at"]) is None:
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}",
                                 {"item": "started_at",
                                  "reason": "오프셋 있는 ISO 에 초 아래 자릿수까지"}))
            ok = False
        first_ok = _interval_ok(row, "first_id", pos, f"{where}#{k}", rej)
        if not first_ok:
            ok = False
        # 잘라 낸 id 는 `retry_torn_tail` **일 때만** 있어야 한다(§27-4).
        cut = row.get("truncated_ids")
        if row.get("reason") == REASON_TORN_TAIL:
            if not (isinstance(cut, (list, tuple)) and cut
                    and all(isinstance(c, str) and c in pos for c in cut)):
                rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}",
                                     {"item": "truncated_ids"}))
                ok = False
            elif row["first_id"] != cut[0]:
                rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, f"{where}#{k}",
                                     {"reason": "잘라 낸 자리부터 다시 만들지 않았다"}))
                ok = False
        elif cut is not None:
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}",
                                 {"item": "truncated_ids", "reason": "꼬리 재생성이 아니다"}))
            ok = False

        end = mine[k].get(EVENT_END)
        if end is None:
            continue
        gap = [f for f in ATTEMPTS_END_REQUIRED if f not in end]
        if gap:
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}", {"missing": gap}))
            ok = False
            continue
        if not _sub_second(end["finished_at"]) or _aware(end["finished_at"]) is None:
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}",
                                 {"item": "finished_at",
                                  "reason": "오프셋 있는 ISO 에 초 아래 자릿수까지"}))
            ok = False
        if not (_strict_int(end["n_written"]) and end["n_written"] >= 0):
            rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, f"{where}#{k}",
                                 {"item": "n_written"}))
            ok = False
        last_ok = _interval_ok(end, "last_id", pos, f"{where}#{k}", rej)
        if not last_ok:
            ok = False
        # 구간이 성립하지 않으면 **자리로 하는 교차 검사를 하지 않는다** — 없는 자리를
        # 찾다 예외로 죽으면 거부가 아니라 사고가 된다(19번 Minor 5-4).
        if not (first_ok and last_ok):
            continue
        if row["first_id"] is None and (end["last_id"] is not None
                                        or end.get("n_written") != 0):
            # `null` 은 "그 자리가 비었다" 는 뜻만 갖는다(§27-1).
            rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, f"{where}#{k}",
                                 {"reason": "빈 구간인데 끝 줄이 비어 있지 않다"}))
            ok = False
        elif row["first_id"] is not None and end["last_id"] is not None and \
                pos[row["first_id"]] > pos[end["last_id"]]:
            rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, f"{where}#{k}",
                                 {"reason": "구간의 끝이 시작보다 앞이다"}))
            ok = False
    if not ok:
        return {}

    stamp_at = _aware(stamp["started_at"]) if stamp and "started_at" in stamp else None
    stamp_device = stamp.get("device") if stamp else None
    if not _check_attempt_times(starts, mine, order, meta, receipt_time,
                                stamp_at, stamp_device, where, rej):
        return {}
    spans = _attempt_spans(starts, mine, order, pos, n_lines, where, rej)
    if spans is None:
        return {}

    devices = [starts[k]["device"] for k in order]
    if meta["device"] != devices[-1]:
        # 곁 파일은 마지막 시도가 끝난 뒤에 쓴다(§28-3 나).
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"item": "device", "reason": "곁 파일이 마지막 시작 줄과 다르다"}))
        return {}

    return {
        "attempts": meta["attempts"],
        "attempts_source": "sidecar",
        "attempts_start_rows": len(starts),
        "attempts_end_rows": sum(1 for k in order if EVENT_END in mine[k]),
        "attempts_rows_total": n_total,
        "attempts_spans": spans,
        "attempts_reasons": [starts[k]["reason"] for k in order],
        # 원시 식별자를 알림으로 내보내지 않는다(19번 Minor 5-5) — 자리와 길이로 싣는다.
        "attempts_first_positions": [
            None if starts[k]["first_id"] is None else pos[starts[k]["first_id"]] for k in order
        ],
        "devices": sorted(set(devices)),
        "devices_spanned": len(set(devices)) > 1,
    }


def _interval_ok(row, key: str, pos, where: str, rej) -> bool:
    """구간의 id 는 **문자열이거나 `null`** 이고, `null` 이 아니면 생성 목록 안이다(§27-1)."""
    value = row[key]
    if value is None:
        return True
    if not isinstance(value, str):
        # 해시할 수 없는 값에서 예외로 죽지 않게 먼저 걸른다(19번 Minor 5-4).
        rej.append(Rejection(RejectCode.ATTEMPTS_SCHEMA, where,
                             {"item": key, "reason": "문자열이거나 null 이어야 한다"}))
        return False
    if value not in pos:
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"item": key, "reason": "생성 목록에 없다"}))
        return False
    return True


def _check_attempt_times(starts, mine, order, meta, receipt_time: datetime,
                         stamp_at: datetime | None, stamp_device, where: str, rej) -> bool:
    """시각의 관계 여덟(§27-6 · §28-4). **순간으로 비교하고 전부 비엄격**이다.

    도장 쪽 관계 1·3 은 도장과 시도 기록을 함께 봐야 하므로 여기서 건다 — 앞 판은 시도 줄과
    곁 파일만 보고 **도장을 뺐다.** 그래서 도장이 등록 전 시각이거나 첫 시도보다 늦은 묶음이
    통과했다. 막는 것은 **도장과 시도 기록 사이의 모순**이다(등록 전 생성은 관계 2 가 막는다).
    """
    ok = True
    st = {k: _aware(starts[k]["started_at"]) for k in order}
    fin = {k: _aware(mine[k][EVENT_END]["finished_at"]) for k in order if EVENT_END in mine[k]}

    for k in order:
        if st[k] < receipt_time:                                          # 관계 2
            rej.append(Rejection(RejectCode.TIME_ORDER, f"{where}#{k}",
                                 {"pair": "start<receipt"}))
            ok = False
        if k in fin and fin[k] < st[k]:                                   # 관계 6
            rej.append(Rejection(RejectCode.TIME_ORDER, f"{where}#{k}",
                                 {"pair": "end<start"}))
            ok = False
    for a, b in zip(order, order[1:]):
        if st[b] < st[a]:                                                 # 관계 8 (항상)
            rej.append(Rejection(RejectCode.TIME_ORDER, f"{where}#{b}",
                                 {"pair": "start<prev_start"}))
            ok = False
        if a in fin and st[b] < fin[a]:                                   # 관계 7 (끝 줄이 있을 때만)
            rej.append(Rejection(RejectCode.TIME_ORDER, f"{where}#{b}",
                                 {"pair": "start<prev_end"}))
            ok = False

    if stamp_at is not None:
        if stamp_at < receipt_time:                                   # 관계 1
            rej.append(Rejection(RejectCode.TIME_ORDER, where,
                                 {"pair": "stamp<receipt"}))
            ok = False
        if st[order[0]] < stamp_at:                                   # 관계 3
            rej.append(Rejection(RejectCode.TIME_ORDER, where,
                                 {"pair": "first_start<stamp"}))
            ok = False
    if stamp_device is not None and stamp_device != starts[order[0]]["device"]:
        # §28-3 나 — 도장은 첫 시도와 맞댄다. 곁 파일은 마지막 시작 줄과 맞댄다.
        rej.append(Rejection(RejectCode.START_STAMP_MISMATCH, where,
                             {"item": "device", "reason": "도장이 첫 시작 줄과 다르다"}))
        ok = False

    side_start, side_end = _aware(meta["started_at"]), _aware(meta["finished_at"])
    if side_start is not None and side_start != st[order[0]]:              # 관계 4
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"item": "started_at", "reason": "곁 파일이 첫 시작과 다르다"}))
        ok = False
    if side_end is not None and side_end != fin[order[-1]]:                # 관계 5
        rej.append(Rejection(RejectCode.ATTEMPTS_ROW_MISMATCH, where,
                             {"item": "finished_at", "reason": "곁 파일이 마지막 종료와 다르다"}))
        ok = False
    return ok


def _last_id_ok(start_row, end, pos, n_lines: int, k: int, where: str, rej) -> bool:
    """규칙 5 — **끝 식별자가 그 구간의 끝인가**(§28-1).

    규칙 1~4 는 **시작 자리와 적힌 줄 수**만 본다. 그래서 끝 식별자가 구간의 끝과 달라도
    회계가 선다 — 첫 id 부터 9줄을 쓰고 `n_written = 9` 인데 `last_id` 를 첫 id 나 `null` 로
    적어도 시작보다 앞서지만 않으면 통과했다. **서로 모순되는 기록이 검증 완료로 분류된다.**

    쓴 줄이 없으면 `last_id` 는 `null` 이어야 한다. **`first_id` 까지 `null` 로 강제하지 않는다** —
    쓸 것이 있어 시작했으나 한 줄도 쓰기 전에 정상 종료한 경우가 있다.
    **끝 줄이 없는 시도에는 이 규칙을 걸지 않는다**(부르는 쪽이 거른다).
    """
    n = end["n_written"]
    last = end["last_id"]
    if n == 0:
        if last is not None:
            rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                                 {"reason": "쓴 줄이 없는데 끝 식별자가 있다", "attempt_no": k}))
            return False
        return True
    first = start_row["first_id"]
    if first is None or last is None:
        rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                             {"reason": "쓴 줄이 있는데 구간의 끝이 비었다", "attempt_no": k,
                              "n_written": n}))
        return False
    want = pos[first] + n - 1
    if want >= n_lines:
        rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                             {"reason": "구간의 끝이 목록 범위를 벗어난다", "attempt_no": k,
                              "expected_position": want, "n_lines": n_lines}))
        return False
    if pos[last] != want:
        rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                             {"reason": "끝 식별자가 그 구간의 끝이 아니다", "attempt_no": k,
                              "expected_position": want, "last_position": pos[last]}))
        return False
    return True


def _attempt_spans(starts, mine, order, pos, n_lines: int, where: str, rej):
    """시도가 남긴 줄 수를 **목록 위치로 센다**(§28-1의 규칙 넷).

    끝 줄은 죽은 시도에 없고 그 시도가 쓴 줄은 파일에 남는다(바-14). 그래서 적힌 수를 더하는
    회계는 서지 않는다. 자리는 등록 목록과 **다음 시도가 스스로 적은 시작 id** 에서 오므로
    추정이 끼지 않는다.

    합이 실제로 보는 것은 둘이다 — **첫 시도가 목록 처음에서 시작했는가**와 **시작 자리가
    뒤로 가지 않는가.** 자리 차를 다 더하면 그 둘만 남는다. 그래서 둘을 따로 적어 거부한다.
    """
    def start_pos(k: int) -> int | None:
        value = starts[k]["first_id"]
        return None if value is None else pos[value]

    first = start_pos(order[0])
    if first != 0 and not (first is None and n_lines == 0):
        # `null` 은 "쓸 것이 없었다" 는 뜻이다(§27-1) — 목록이 비지 않았는데 첫 시도가 그렇게
        # 적으면 **목록 처음에서 시작한 것이 아니다.** 판정이 요구한 조건을 문자 그대로 세운다.
        rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                             {"reason": "첫 시도가 목록 처음에서 시작하지 않았다",
                              "first_position": first}))
        return None

    spans: list[int] = []
    for i, k in enumerate(order):
        here = start_pos(k)
        if here is None:
            spans.append(0)                                   # 규칙 2
            continue
        if i + 1 < len(order):
            nxt = start_pos(order[i + 1])
            nxt = n_lines if nxt is None else nxt              # 규칙 3
        else:
            nxt = n_lines                                      # 규칙 1 의 마지막 시도
        if nxt < here:
            rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                                 {"reason": "시작 자리가 뒤로 갔다", "attempt_no": k}))
            return None
        spans.append(nxt - here)

    for i, k in enumerate(order):
        end = mine[k].get(EVENT_END)
        if end is None:
            continue                                           # 죽은 시도에는 규칙 4·5 를 걸지 않는다
        if end["n_written"] != spans[i]:                       # 규칙 4
            rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                                 {"reason": "끝 줄이 적은 수가 자리 차와 다르다",
                                  "attempt_no": k, "n_written": end["n_written"],
                                  "from_positions": spans[i]}))
            return None
        if not _last_id_ok(starts[k], end, pos, n_lines, k, where, rej):   # 규칙 5
            return None
    if sum(spans) != n_lines:
        rej.append(Rejection(RejectCode.ATTEMPTS_ACCOUNTING_MISMATCH, where,
                             {"reason": "자리 차의 합이 줄 수와 다르다",
                              "sum": sum(spans), "n_lines": n_lines}))
        return None
    return spans


def _check_times(meta, receipt_time: datetime, files, rej) -> None:
    started, finished = _aware(meta["started_at"]), _aware(meta["finished_at"])
    if started is None or finished is None:
        rej.append(Rejection(RejectCode.TIME_NAIVE, files.stem,
                             {"reason": "오프셋 있는 ISO 시각이어야 한다"}))
        return
    if started > finished:
        rej.append(Rejection(RejectCode.TIME_ORDER, files.stem, {"pair": "started>finished"}))
    if started < receipt_time:
        rej.append(Rejection(RejectCode.TIME_ORDER, files.stem, {"pair": "started<receipt"}))


def _check_training(meta, mode, artifact, ledger, seen_adapters, files, rej,
                    registration: Registration | None = None) -> None:
    """학습 쪽 출처. **에코는 여섯 검사에서 빠진다**(§16-2 I-3)."""
    if mode == MODE_ECHO:
        if "scored_adapter_sha256" in meta and meta["scored_adapter_sha256"] is not None:
            rej.append(Rejection(RejectCode.ECHO_ADAPTER_PRESENT, files.stem))
        return

    if artifact is None:
        rej.append(Rejection(RejectCode.ARTIFACT_META_MISSING, files.stem,
                             {"reason": "어댑터 sha 와 스텝은 학습 산출물 meta 에서 온다"}))
        return
    if meta.get("scored_adapter_sha256") != artifact.adapter_sha256:
        rej.append(Rejection(RejectCode.LEDGER_MISMATCH, files.stem,
                             {"item": "scored_adapter_sha256"}))
    if meta.get("scored_adapter_step") != artifact.step:
        rej.append(Rejection(RejectCode.LEDGER_MISMATCH, files.stem,
                             {"item": "scored_adapter_step"}))
    want_rows = None if registration is None else registration.train_rows_digest
    if want_rows is not None and artifact.train_rows_digest != want_rows:
        # 곁 파일이 아니라 어댑터 meta 와 맞댄다(리허설 2판 반영판 §2-5 · §2-7, §34). 값은 싣지 않는다.
        rej.append(Rejection(RejectCode.LEDGER_MISMATCH, files.stem,
                             {"item": "train_rows_digest", "reason": "어댑터 meta 의 값이 등록과 다르다"}))
    if ledger is None:
        rej.append(Rejection(RejectCode.LEDGER_MISMATCH, files.stem, {"reason": "원장이 없다"}))
        return
    if meta["train_run_id"] != ledger.run_id or meta["train_ledger_sha256"] != ledger.file_sha256:
        rej.append(Rejection(RejectCode.LEDGER_MISMATCH, files.stem, {"item": "run 신원"}))
    if meta["cell"] != ledger.cell or meta["client"] != ledger.client:
        rej.append(Rejection(RejectCode.LEDGER_MISMATCH, files.stem, {"item": "칸·참여자"}))
    if artifact.step != ledger.final_step:
        # 중간 체크포인트를 채점하는 것을 여기서 막는다 — 불변조건 3-2(best 금지, last 채점)의 자리다.
        rej.append(Rejection(RejectCode.ADAPTER_NOT_FINAL, files.stem,
                             {"artifact_step": artifact.step, "ledger_final_step": ledger.final_step}))
    if seen_adapters is not None and artifact.adapter_sha256 in seen_adapters:
        other = seen_adapters[artifact.adapter_sha256]
        rej.append(Rejection(RejectCode.ADAPTER_DUPLICATE, files.stem,
                             {"also_in": f"{other[0]}_s{other[1]}"}))


def check_env_consistency(bundles: Sequence[VerifiedBundle]) -> list[Rejection]:
    """묶음 사이에서 같아야 하는 것들을 맞댄다. **한 묶음만 보면 알 수 없는 것**이라 따로 있다."""
    out: list[Rejection] = []
    if len(bundles) < 2:
        return out
    first = bundles[0]
    for key in ENV_FIELDS:
        values = {json.dumps(b.sidecar.get(key), sort_keys=True, ensure_ascii=False) for b in bundles}
        if len(values) != 1:
            out.append(Rejection(RejectCode.ENV_MISMATCH, first.files.stem,
                                 {"item": key, "n_distinct": len(values)}))
    return out


def population_digest(ids) -> str:
    """채점 모집단의 집합 신원. 산출물이 이 값을 싣는다."""
    return set_digest(ids)
