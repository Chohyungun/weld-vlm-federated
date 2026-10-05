"""export 의 실측값 — 모델을 올리기 **전에** 프로세서만 열어 등록값과 맞댈 값을 잰다(리허설 2판 §2-4 의 1 · §2-5 의 "실측" 행).

잰 값은 평가 쪽 `evaluation.actuals.check_actuals` 가 등록값과 항목별 · 자료형까지 맞댄다. 이 모듈은 그 비교를 하지 않는다.

| 값 | 어디서 |
|---|---|
| `processor_config_sha256` | `vlm.pilot_vlm.processor_config_sha256` — 학습과 같은 함수 |
| `processor_min_pixels` · `processor_max_pixels` | 이미지 프로세서의 `size.shortest_edge` · `size.longest_edge`. transformers 5.15.0 의 Qwen 프로세서는 `min_pixels` · `max_pixels` 인자를 이 두 칸에 넣고 같은 이름의 속성을 두지 않는다(캐시의 `Qwen/Qwen3.5-4B` 프로세서를 CPU 로 열어 확인했다) |
| `patch_size` · `merge_size` | 이미지 프로세서의 같은 이름의 속성 |
| `gen_prefix_sha256` | `vlm.pilot_vlm.gen_prefix_sha256` — 등록 프롬프트를 이미지 없이 렌더한 접두의 토큰 id |
| `transformers_version` | `transformers.__version__` |

값이 없거나 정수가 아니면 지어내지 않고 `MeasureError` 로 멈춘다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MEASURED_KEYS", "MeasureError", "MeasureSpec", "measure_env", "measure_processor"]

MEASURED_KEYS = ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels", "patch_size",
                 "merge_size", "gen_prefix_sha256", "transformers_version")


class MeasureError(ValueError):
    """프로세서에서 잴 값이 없다 — 지어내지 않는다."""


@dataclass(frozen=True)
class MeasureSpec:
    """프로세서를 여는 값 — 학습이 연 것과 같은 출처(설정)에서 온다."""

    model_id: str
    model_revision: str | None
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)
    prompt_text: str = ""
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _int_attr(obj: Any, name: str) -> int:
    v = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    if isinstance(v, bool) or not isinstance(v, int):
        raise MeasureError(f"프로세서의 {name} 가 정수가 아니다: {v!r}")
    return int(v)


def measure_processor(proc: Any, *, prompt_text: str, chat_template_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """연 프로세서에서 일곱 값을 잰다."""
    import transformers

    from vlm.pilot_vlm import gen_prefix_sha256, processor_config_sha256

    ip = getattr(proc, "image_processor", None)
    if ip is None:
        raise MeasureError("이미지 프로세서가 없다")
    size = getattr(ip, "size", None)
    if size is None:
        raise MeasureError("이미지 프로세서에 size 가 없다")
    return {
        "processor_config_sha256": processor_config_sha256(proc),
        "processor_min_pixels": _int_attr(size, "shortest_edge"),
        "processor_max_pixels": _int_attr(size, "longest_edge"),
        "patch_size": _int_attr(ip, "patch_size"),
        "merge_size": _int_attr(ip, "merge_size"),
        "gen_prefix_sha256": gen_prefix_sha256(proc, prompt_text, chat_template_kwargs),
        "transformers_version": str(transformers.__version__),
    }


def measure_env(spec: MeasureSpec) -> dict[str, Any]:
    """**승인된 실제 구현** — 학습과 같은 호출(`AutoProcessor.from_pretrained(모델, revision, **프로세서 인자)`)로 프로세서만 연다.
    모델 가중치는 올리지 않는다."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(spec.model_id, revision=spec.model_revision,
                                         **dict(spec.processor_kwargs or {}))
    return measure_processor(proc, prompt_text=spec.prompt_text, chat_template_kwargs=spec.chat_template_kwargs)
