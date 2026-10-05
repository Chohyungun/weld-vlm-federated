"""에코 생성기 — 모델을 부르지 않고 val 행의 **학습 타깃 문자열**을 생성문 자리에 넣는다(리허설 2판 §2-4 의 에코 모드 · 계약 라-12).

같은 파싱 · 역변환 · 기록 경로를 탄다 — 에코 묶음을 채점하면 쓰는 쪽 좌표 경로의 항등이 드러난다(조건 ② 가).

| 줄의 값 | 에코에서의 뜻 |
|---|---|
| `text` | `vlm.pilot_vlm.build_target(행, 이미지 크기, 좌표 규약)` — 학습이 쓰는 함수 그대로 |
| `n_new_tokens` | 타깃을 토크나이저로 센 수 + 종료 토큰 1 |
| `gen_stop` | `eos` |
| `model_input_wh` · 격자 | 프로세서가 그 이미지에 실제로 쓴 격자 `(t, h, w)` 와 `[w × patch_size, h × patch_size]` |
| `latency_ms` | 파싱 경로만의 시간(export 가 잰다) |

**승인된 실제 구현**이다. 프로세서는 학습 · 실측과 같은 호출(`AutoProcessor.from_pretrained(모델, revision, **프로세서 인자)`)로 연다. GPU 를 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any

__all__ = ["EchoUnavailable", "load_echo_generator"]


class EchoUnavailable(ValueError):
    """에코 생성기를 열 수 없다 — 모드가 에코가 아니거나 페어 경로가 없다."""


def load_echo_generator(gen_cfg: Any, *, processor: Any = None):
    """`GenerationConfig` → 생성기 `(이미지, image_id) → GenOut`. `processor` 는 시험이 연 프로세서를 줄 자리다(파이썬 인자로만)."""
    from vlm.coords import CoordCfg, ImageGeom
    from vlm.export_run import GenOut
    from vlm.pilot_vlm import build_target, load_pairs

    if gen_cfg.mode != "echo":
        raise EchoUnavailable(f"에코 생성기는 에코 모드에서만 연다: {gen_cfg.mode!r}")
    if not gen_cfg.pairs_path:
        raise EchoUnavailable("에코 타깃을 읽을 페어 경로가 설정에 없다(uni_pairs.path)")
    if processor is None:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(gen_cfg.model_id, revision=gen_cfg.model_revision,
                                                  **dict(gen_cfg.processor_kwargs or {}))
    rows = {str(r["image_id"]): r for r in load_pairs("val", None, pairs_path=gen_cfg.pairs_path)}
    cfg = CoordCfg(coord_space=gen_cfg.coord_space)
    ip = processor.image_processor
    patch = int(ip.patch_size)

    def generate(img, image_id: str) -> GenOut:
        row = rows.get(str(image_id))
        if row is None:
            raise EchoUnavailable(f"페어의 val 행에 없는 id 다: {image_id}")
        target = build_target(row, ImageGeom(orig_w=img.size[0], orig_h=img.size[1]), cfg)
        n = len(processor.tokenizer(target, add_special_tokens=False)["input_ids"]) + 1
        grid = ip(images=[img.convert("RGB")], return_tensors="np")["image_grid_thw"][0]
        t, h, w = (int(v) for v in grid)
        return GenOut(text=target, gen_stop="eos", n_new_tokens=n, model_input_wh=(w * patch, h * patch),
                      grid_thw=(t, h, w))

    return generate
