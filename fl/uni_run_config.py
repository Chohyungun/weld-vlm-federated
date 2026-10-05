"""통합형 연합의 실행 설정 키 — 선언(pyproject) · 내려보내기(서버) · 읽기(클라이언트)를 한 목록으로 묶는다.

서버와 클라이언트가 같은 모델 · 같은 페어 · 같은 프롬프트로 학습한다는 보장이 전에는 없었다. 서버는
설정의 모델로 초기 어댑터를 만들고, 클라이언트는 모듈 상수로 학습했다(미니스펙 12번 §2-1). 4B 를 설정에
넣으면 서버는 4B 어댑터를, 클라이언트는 0.8B 어댑터를 만들어 라운드 하나를 버린 뒤 모양 불일치로 죽는다.

그래서 키 목록을 **여기 하나**에 둔다. 서버와 인프로세스 서버는 `down_config` 로 같은 키를 내려보내고,
클라이언트는 `client_run_cfg` 로 같은 키를 읽는다. `flwr run` 은 `pyproject.toml` 에 선언하지 않은 키의
덮어쓰기를 거부하므로 선언도 이 목록과 같아야 한다 — 시험이 셋을 맞댄다.

빈 문자열은 "지정하지 않았다" 이고 파일럿 재현의 기본값으로 돈다 — **리허설 · 진단 목적일 때만이다.**
본실험(`purpose="main"`)은 `MAIN_REQUIRED` 가 하나라도 비면 받는 쪽(`client_run_cfg`)이 학습 전에 거부한다 —
서버는 초기 어댑터를 읽기 전에, 클라이언트는 학습 전에. 목적(`purpose`)은 언제나 빈 값을 받지 않는다.
이 모듈은 torch 를 가져오지 않는다 — 서버가 가볍게 부를 수 있어야 한다.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

__all__ = ["UNI_RUN_KEYS", "PURPOSES", "UNI_RUN_DEFAULTS", "UNI_SERVER_DEFAULTS", "MAIN_REQUIRED", "UniMainIncomplete",
           "down_config", "client_run_cfg"]

#: 목적의 값 공간. 판정 07 의 25 가 셋째 값을 받았다.
PURPOSES = ("main", "rehearsal", "frame_diag")

#: 통합형 연합이 run-config 로 받는 키와 선언 기본값.
UNI_RUN_DEFAULTS: dict[str, str] = {
    "uni-model": "",
    "uni-model-revision": "",
    "uni-pairs": "",
    "uni-pairs-digest": "",
    "uni-prompt": "",
    # JSON 문자열이다 — run-config 값은 스칼라만 받는다. 빈 문자열이면 학습 코드의 기본값을 쓴다.
    "uni-chat-template-kwargs": "",
    "uni-coord-space": "",
    "purpose": "main",
    "plan": "",
    "rehearsal-root": "",
    # 시드 번호(1 부터). 연합 칸의 어댑터 meta 신원에 싣는다. 비면 meta 의 값이 null 이다.
    "uni-seed-index": "",
    # 프로세서 설정(JSON 객체 — `min_pixels`·`max_pixels` 등). 학습 쪽 프로세서를 **등록된 설정으로** 연다.
    "uni-processor-kwargs": "",
    # epoch 하나의 감독 토큰 기대값(JSON 객체 — 참여자 이름 → 정수). 저장 전에 실측과 맞댄다(12번 §2-4).
    "uni-expected-supervised-tokens": "",
    # 리허설 · 진단의 학습 행 목록 폴더(루트의 `lists/`). 비면 참여자 행 전체(본실험). 클라이언트는 `train_uni_local_<참여자>.jsonl` 을
    # 목록 단계의 기록 · 페어의 원본 행과 맞댄 뒤에만 학습한다(리허설 2판 §4-1 의 연합 행 — 로컬과 같은 행). 본실험은 받지 않는다.
    "uni-train-lists": "",
}
UNI_RUN_KEYS: tuple[str, ...] = tuple(UNI_RUN_DEFAULTS)

#: 서버만 읽는 통합형 키와 선언 기본값 — 클라이언트에 내려보내지 않는다.
#: `uni-train-root` — 연합 칸의 출력을 로컬 · 중앙과 같은 꼴로 `<학습 루트>/uni_fed_s<n>/` 에 쓴다(리허설 2판 §13-4). 비면 `<project>/fl/uni_fed`.
#: `uni-init-adapter` — 공통 초기 어댑터 파일. 비면 `<project>/fl/uni_fed/initial.npz` 다. 리허설 · 진단은 루트의 `init/initial.npz`
#: (등록 단계가 만든 것 — 2판 §13-4 의 끝)를 준다. 본실험은 받지 않는다(실행기의 캐시 자리 하나다).
UNI_SERVER_DEFAULTS: dict[str, str] = {"uni-train-root": "", "uni-init-adapter": ""}

#: 본실험이 비워 둘 수 없는 키. 비면 파일럿 기본값(모델 · 판 · 페어 · 프롬프트)으로 내려가므로 받지 않는다.
MAIN_REQUIRED: tuple[str, ...] = (
    "uni-model", "uni-model-revision", "uni-pairs", "uni-pairs-digest", "uni-prompt",
    "uni-chat-template-kwargs", "uni-coord-space", "uni-seed-index", "uni-processor-kwargs",
    "uni-expected-supervised-tokens",
)


class UniMainIncomplete(ValueError):
    """본실험 목적인데 필수 설정이 비어 있다. 파일럿 기본값으로 내려가지 않는다."""


def down_config(get: Callable[[str, Any], Any]) -> dict[str, str]:
    """서버가 클라이언트에 내려보낼 통합형 키. `get(key, default)` 는 서버 설정을 읽는 함수다.

    값을 문자열로 맞춘다 — `ConfigRecord` 에 실린 뒤 클라이언트가 같은 꼴로 읽게 한다.
    """
    out: dict[str, str] = {}
    for k, d in UNI_RUN_DEFAULTS.items():
        v = get(k, d)
        out[k] = d if v is None else str(v)
    _check(out)
    return out


def client_run_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """클라이언트가 받은 설정에서 학습 인자를 만든다. 빠진 키는 거부한다 — 서버가 안 보낸 것이다."""
    missing = [k for k in UNI_RUN_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"서버가 통합형 키를 보내지 않았다: {missing}")
    raw = {k: str(cfg[k]) for k in UNI_RUN_KEYS}
    _check(raw)
    if raw["purpose"] == "main":
        empty = [k for k in MAIN_REQUIRED if not raw[k]]
        if empty:
            raise UniMainIncomplete(f"본실험은 이 키를 비워 둘 수 없다 — 파일럿 기본값으로 내려가지 않는다: {empty}")
    kwargs = json.loads(raw["uni-chat-template-kwargs"]) if raw["uni-chat-template-kwargs"] else None
    return {
        "model_id": raw["uni-model"] or None,
        "model_revision": raw["uni-model-revision"] or None,
        "pairs_path": raw["uni-pairs"] or None,
        "pairs_digest": raw["uni-pairs-digest"] or None,
        "prompt_path": raw["uni-prompt"] or None,
        "chat_template_kwargs": kwargs,
        "coord_space": raw["uni-coord-space"] or None,
        "purpose": raw["purpose"],
        "plan": raw["plan"] or None,
        "rehearsal_root": raw["rehearsal-root"] or None,
        "seed_index": int(raw["uni-seed-index"]) if raw["uni-seed-index"] else None,
        "processor_kwargs": json.loads(raw["uni-processor-kwargs"]) if raw["uni-processor-kwargs"] else None,
        "expected_supervised_tokens": (json.loads(raw["uni-expected-supervised-tokens"])
                                       if raw["uni-expected-supervised-tokens"] else None),
        "train_lists": raw["uni-train-lists"] or None,
    }


def _check(raw: Mapping[str, str]) -> None:
    """목적과 템플릿 인자의 꼴. 틀리면 학습 전에 멈춘다."""
    if raw["purpose"] not in PURPOSES:
        raise ValueError(f"purpose 가 {list(PURPOSES)} 가운데 하나가 아니다: {raw['purpose']!r}")
    if raw["purpose"] == "main" and raw.get("uni-train-lists"):
        raise ValueError("본실험은 학습 행 목록(uni-train-lists)을 받지 않는다 — 참여자 행 전체로 학습한다")
    si = raw["uni-seed-index"]
    if si and (not si.isdigit() or int(si) < 1):
        raise ValueError(f"uni-seed-index 가 1 이상의 정수가 아니다: {si!r}")
    for key in ("uni-chat-template-kwargs", "uni-processor-kwargs", "uni-expected-supervised-tokens"):
        s = raw[key]
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} 가 JSON 이 아니다: {exc}") from None
        if not isinstance(obj, dict):
            raise ValueError(f"{key} 가 JSON 객체가 아니다: {s!r}")
        if key == "uni-expected-supervised-tokens" and any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in obj.values()):
            raise ValueError(f"{key} 의 값이 0 이상의 정수가 아니다: {s!r}")
