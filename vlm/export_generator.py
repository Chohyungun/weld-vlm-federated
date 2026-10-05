"""export 의 생성기 적재 — **모드마다 정해진 실제 구현으로 간다**(모드별 구현 결속).

등록의 구현 식별자는 한 칸(`generation.impl_ids`)이고 평가 쪽은 모델 · 에코 두 모드의 곁 파일을 그 한 칸과 맞댄다
(`evaluation.actuals.registered_values` — 모드와 무관하게 같은 칸). 모드마다 다른 함수를 이음새에 넣으면 한쪽은 언제나 등록과 어긋나
대역 실행으로 적힌다. 그래서 두 모드가 **같은 적재 함수**를 지나고, 이 함수가 모드에 따라 정해진 구현으로 간다 —
모드와 구현의 짝은 이 파일의 커밋된 바이트가 정하고, 구현 식별자에 이 파일의 blob 이 실린다.

| 모드 | 구현 |
|---|---|
| `echo` | `vlm.export_echo.load_echo_generator` |
| `model` | `vlm.export_model.load_model_generator` — HF 적재 · 학습 어댑터 주입 · greedy 1회. **실물 검증은 GPU 창의 몫이다** |

**안쪽 구현도 부르기 전에 승인한다**(2판 §1-4). 바깥 함수가 승인돼도 안쪽의 생성기 · 모델 적재기를 모듈 속성에서 찾아 부르면, 그 속성만 바꿔
끼운 대역이 본실험에 들어간다. 그래서 `inner_impl_ids` 가 **지금 부를 객체**(모듈 속성)를 import 시점에 잡아 둔 참조와 `is` 로 맞댄다 —
에코 · 모델 생성기와 모델 적재기(`vlm.pilot_vlm._load_model`) 셋이다. export 는 바깥 함수를 볼 때(순서 3) 이것을 함께 보고 식별자를 곁 파일에 싣고,
이 적재 함수도 부르는 순간 다시 본다. 본실험은 하나라도 대역이면 대역을 한 번도 부르기 전에 거부한다.
"""

from __future__ import annotations

from typing import Any

from vlm.export_echo import load_echo_generator as _ECHO_REAL
from vlm.export_model import load_model_generator as _MODEL_REAL

__all__ = ["INNER_SEAMS", "GeneratorUnavailable", "inner_impl_ids", "load_generator", "model_generator_available"]

#: 안쪽 이음새의 이름 — 곁 파일 · 등록의 구현 식별자에 이 이름으로 실린다.
INNER_SEAMS = ("export_generator_echo", "export_generator_model", "model_loader")


class GeneratorUnavailable(ValueError):
    """이 모드의 실제 생성기가 없다."""


def model_generator_available() -> bool:
    """모델 모드의 실제 생성기가 있는가. 없으면 본실험 모델 export 의 승인 목록이 비고, 명령줄이 시작 전에 거부한다."""
    return True


def inner_impl_ids(*, purpose: str, standin_allowed: bool = False) -> dict[str, dict[str, Any]]:
    """안쪽 구현 셋 — **지금 부를 객체**를 승인 참조와 맞댄다. 본실험은 대역이면 `SeamRejected`. 돌려주는 값은 이음새 → 식별자다."""
    import vlm.export_echo as echo_mod
    import vlm.export_model as model_mod
    import vlm.pilot_vlm as pv
    from vlm.seams import check_seam

    gen_model = check_seam(model_mod.load_model_generator, (_MODEL_REAL,), seam="export_generator_model",
                           purpose=purpose, standin_allowed=standin_allowed)
    if gen_model["approved"]:
        loader = check_seam(pv._load_model, pv._APPROVED_MODEL_LOADERS, seam="model_loader", purpose=purpose,
                            standin_allowed=standin_allowed)
    else:
        # 대역 생성기가 어느 적재기를 부를지는 이 자리에서 알 수 없다 — 기본 적재기를 승인으로 적지 않고,
        # 그 생성기를 적재기의 출처로 적는다(승인 아님 · 외부 검토 재확인 추가 2).
        loader = check_seam(model_mod.load_model_generator, (), seam="model_loader", purpose=purpose,
                            standin_allowed=standin_allowed)
    return {
        "export_generator_echo": check_seam(echo_mod.load_echo_generator, (_ECHO_REAL,), seam="export_generator_echo",
                                            purpose=purpose, standin_allowed=standin_allowed),
        "export_generator_model": gen_model,
        "model_loader": loader,
    }


def load_generator(gen_cfg: Any, **kw: Any):
    """`GenerationConfig` → 생성기. 키워드 인자는 그 구현에 그대로 넘긴다(시험이 연 프로세서 따위 — 파이썬 인자로만).

    부르는 순간 안쪽 구현을 다시 맞댄다 — 바깥 검사 뒤에 속성을 바꿔 끼워도 본실험은 대역을 부르지 않는다."""
    import vlm.export_echo as echo_mod
    import vlm.export_model as model_mod

    inner_impl_ids(purpose=getattr(gen_cfg, "purpose", "main"), standin_allowed=True)
    if gen_cfg.mode == "echo":
        return echo_mod.load_echo_generator(gen_cfg, **kw)
    if gen_cfg.mode == "model":
        return model_mod.load_model_generator(gen_cfg, **kw)
    raise GeneratorUnavailable(f"모르는 모드: {gen_cfg.mode!r}")
