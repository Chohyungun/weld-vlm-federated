"""통합형 2칸 예측 raw — 평가 담당 인계용.

원시 출력 계약(평가 스펙 §2-3, 통합형):
    generations.jsonl  {image_id, text, bbox_px_parsed, coord_space, coord_cfg_hash, latency_ms}

규칙:
- **greedy 1회.** 재시도·재프롬프트 없다. 파싱 실패는 실패로 기록한다.
- 배포 `generation_config` 를 신뢰하지 않고 **백지에서 명시 구성**(레지스트리 #5).
- 좌표 역변환은 `vlm/coords.py` `to_px` **1회**. 정수화하지 않는다(명세 판정 5) —
  `bbox_px` 는 float 로 나가고 정수 표기가 필요하면 채점기의 최종 1회다.
- 평가셋 이미지에 대한 추론이며 학습·선택에 쓰지 않는다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.manifest_io import load_snapshot, split_view
from detection import serialize
from vlm.coords import CoordCfg, ImageGeom, coord_cfg_hash, to_px
from vlm.pilot_vlm import COORD_CFG, MODEL_ID, PROMPT_PATH, _load_model

SNAPSHOT_DIR = "data/processed/aihub71761_rt_v1_pilot3000"
OUT_ROOT = Path("outputs/pilot_c").resolve()
MAX_NEW_TOKENS = 256

CELLS = {
    "uni_central": OUT_ROOT / "uni_central" / "adapter_last.npz",
    "uni_fed": OUT_ROOT / "uni_fed" / "adapter_last.npz",
}


def parse_and_backproject(text: str, geom: ImageGeom) -> tuple[list | None, str | None]:
    """생성문 → 원본 픽셀 bbox. 추출은 관대하게, 검증은 엄격하게."""
    s = text.strip()
    i = s.find("{")
    if i < 0:
        return None, "no_json"
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                block = s[i : j + 1]
                break
    else:
        return None, "truncated"
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return None, "json_decode"
    if not isinstance(obj.get("defects"), list):
        return None, "schema_violation"
    out = []
    for d in obj["defects"]:
        b = d.get("bbox_2d")
        if not isinstance(b, list) or len(b) != 4:
            return None, "bbox_invalid"
        # 역변환 1회. 정수화하지 않는다 (판정 5)
        out.append({"iso_code": str(d.get("iso_code", "")),
                    "bbox_px": [round(float(v), 3) for v in to_px(b, geom, COORD_CFG)]})
    return {"defects": out, "verdict": obj.get("verdict"),
            "cited_clauses": obj.get("cited_clauses", [])}, None


def main() -> None:
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    from PIL import Image
    from transformers import GenerationConfig

    sn = load_snapshot(SNAPSHOT_DIR)
    eval_m = split_view(sn.manifest, "eval")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    cfg_hash = coord_cfg_hash(COORD_CFG)
    repo = Path.cwd().resolve()
    print(f"평가셋 {len(eval_m)}장 · greedy 1회 · max_new_tokens {MAX_NEW_TOKENS}")

    # 배포 generation_config 를 쓰지 않고 백지에서 구성한다 (레지스트리 #5)
    gen = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                           repetition_penalty=1.0)
    assert gen.do_sample is False

    for cell, npz in CELLS.items():
        if not npz.exists():
            print(f"  {cell}: 어댑터 없음 — 건너뜀"); continue
        model, proc = _load_model()
        loaded = np.load(npz)
        keys = list(loaded.files)
        ref = get_peft_model_state_dict(model)
        sd = serialize.ndarrays_to_state_dict([loaded[k] for k in keys], keys, ref)
        set_peft_model_state_dict(model, sd)
        model.eval()

        out_path = OUT_ROOT / "predictions" / f"{cell}.generations.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fails: dict[str, int] = {}
        t_cell = time.perf_counter()
        with out_path.open("w", encoding="utf-8") as fh:
            for n, (image_id, rel) in enumerate(zip(eval_m["image_id"], eval_m["rel_path"])):
                img = Image.open(repo / rel).convert("RGB")
                geom = ImageGeom(orig_w=img.size[0], orig_h=img.size[1])
                msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                     {"type": "text", "text": prompt}]}]
                enc = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                               return_tensors="pt", add_generation_prompt=True)
                enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
                t0 = time.perf_counter()
                with torch.no_grad():
                    ids = model.generate(**enc, generation_config=gen)
                dt = (time.perf_counter() - t0) * 1000
                text = proc.batch_decode(ids[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0]
                parsed, err = parse_and_backproject(text, geom)
                if err:
                    fails[err] = fails.get(err, 0) + 1
                fh.write(json.dumps({
                    "image_id": image_id, "text": text,
                    "bbox_px_parsed": parsed, "parse_error": err,
                    "coord_space": COORD_CFG.coord_space, "coord_cfg_hash": cfg_hash,
                    "latency_ms": round(dt, 1)}, ensure_ascii=False) + "\n")
                if (n + 1) % 50 == 0:
                    print(f"    {n+1}/{len(eval_m)} ({time.perf_counter()-t_cell:.0f}s)", flush=True)
        n_fail = sum(fails.values())
        print(f"  {cell}: {len(eval_m)}장 ({time.perf_counter()-t_cell:.0f}s) "
              f"파싱 실패 {n_fail} ({n_fail/len(eval_m):.1%}) {fails or ''} → {out_path}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
