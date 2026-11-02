"""묶음 생성의 실물 동치 탐침 — 1 장 생성 대 N 장 묶음 생성(묶음 생성 1 판 §6).

같은 적재(4bit nf4 · 학습 어댑터 주입 · greedy 1 회) 하나로, 같은 행들을 (1) 한 장씩, (2) N 장 묶음으로 생성해 **토큰 열이 정확히
같은지**를 행마다 본다. 묶음은 구성의 차이이지 디코딩의 차이가 아니지만, bf16 · 역양자화의 행렬곱 꼴이 바뀌면 argmax 가 뒤집힐 수
있다 — 동치를 가정하지 않고 잰다. 결정(등록할 배치)은 사람이 산출 JSON 을 보고 한다. 이 스크립트는 판정하지 않는다.

입력은 **학습 행**(A3 의 80 행 — 평가셋이 아니다)이고 어댑터는 A3 재시도의 것이다. 참고 대조로 A100 이 배치 1 로 낸 생성문과
로컬 배치 1 의 생성문을 맞대 **장비 간 차이의 크기**도 함께 적는다(묶음 차이와 견주는 잣대).

    .venv/Scripts/python.exe -X utf8 -B scripts/probe/batch_equivalence.py --adapter <adapter.npz> --rows <rows.txt>
        [--reference <generations.jsonl>] [--batches 4,8] [--n 80] [--out <json>]

산출 JSON 만 쓴다(outputs/ 아래). 이미지는 열어 생성에만 쓰고 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAIRS = ROOT / "data/processed/pairs_main_v2/pairs.jsonl"
CONFIG = ROOT / "configs/base.yaml"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _uni_cfg():
    import yaml

    from vlm.uni_config import uni_config_from

    return uni_config_from(yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {})


def _rows(rows_file: Path, n: int | None) -> list[tuple[str, Path]]:
    ids = rows_file.read_text(encoding="utf-8").split()
    ids = ids if n is None else ids[:n]
    want = set(ids)
    path_of: dict[str, Path] = {}
    with PAIRS.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["image_id"] in want:
                path_of[d["image_id"]] = ROOT / d["image_path"]
    missing = [i for i in ids if i not in path_of or not path_of[i].is_file()]
    if missing:
        raise SystemExit(f"페어 v2 에서 경로를 못 찾은 행 {len(missing)}: {missing[:3]}")
    return [(i, path_of[i]) for i in ids]


def _parse(text: str, img):
    from vlm.coords import CoordCfg, ImageGeom
    from vlm.gen_parse import line_fields, parse_generation

    f = line_fields(parse_generation(text), ImageGeom(orig_w=img.size[0], orig_h=img.size[1]), CoordCfg(coord_space="ABS_ORIG"))
    return {"parse_error": f["parse_error"], "bbox_px_parsed": f["bbox_px_parsed"]}


def _record(out, img) -> dict:
    return {"token_ids": list(out.token_ids or []), "text": out.text, "gen_stop": out.gen_stop,
            "n_new_tokens": int(out.n_new_tokens), "latency_ms": out.latency_ms, "batch_n": out.batch_n,
            "parse": _parse(out.text, img)}


def _compare(a: dict, b: dict) -> dict:
    ta, tb = a["token_ids"], b["token_ids"]
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    return {"tokens_equal": ta == tb, "first_divergence": first, "text_equal": a["text"] == b["text"],
            "stop_equal": a["gen_stop"] == b["gen_stop"], "parse_equal": a["parse"] == b["parse"],
            "n_a": len(ta), "n_b": len(tb)}


def main(argv=None) -> int:
    import torch
    from PIL import Image

    from vlm.export_model import GREEDY, load_model_generator
    from vlm.pilot_vlm import load_prompt

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None, help="A100 배치 1 생성문(generations.jsonl) — 장비 간 대조")
    ap.add_argument("--batches", default="4,8")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    cfg = _uni_cfg()
    batches = [int(x) for x in a.batches.split(",") if x.strip()]
    prompt_text, prompt_sha = load_prompt(ROOT / cfg.prompt_path)
    rows = _rows(a.rows, a.n)
    out_path = a.out or ROOT / "outputs" / f"probe_batchgen_{datetime.now().astimezone():%Y%m%d}" / "batch_equivalence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()

    gen_cfg = SimpleNamespace(mode="model", decoding=dict(GREEDY), batch_size=max(batches + [1]), adapter_path=str(a.adapter),
                              purpose="rehearsal", model_id=cfg.model_id, model_revision=cfg.model_revision,
                              processor_kwargs=dict(cfg.processor_kwargs or {}), padding_side="left",
                              chat_template_kwargs=dict(cfg.chat_template_kwargs or {}), prompt_text=prompt_text,
                              max_new_tokens=int(cfg.max_new_tokens))
    t_load = time.perf_counter()
    generate = load_model_generator(gen_cfg)
    load_s = time.perf_counter() - t_load
    imgs = {}
    for iid, p in rows:
        im = Image.open(p)
        im.load()
        imgs[iid] = im

    result: dict = {
        "kind": "batch_equivalence_probe", "version": 1, "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_commit": head, "model_id": cfg.model_id, "model_revision": cfg.model_revision, "max_new_tokens": int(cfg.max_new_tokens),
        "prompt_sha256": prompt_sha, "adapter_sha256": _sha(a.adapter.read_bytes()), "rows_sha256": _sha(a.rows.read_bytes()),
        "n_rows": len(rows), "row_ids": [i for i, _ in rows], "batches": batches, "load_s": round(load_s, 1),
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "passes": {},
    }

    def run_pass(n: int) -> dict:
        torch.cuda.reset_peak_memory_stats()
        recs: dict[str, dict] = {}
        call_s: list[float] = []
        try:
            for k in range(0, len(rows), n):
                chunk = rows[k:k + n]
                t0 = time.perf_counter()
                if n == 1:
                    outs = [generate(imgs[chunk[0][0]], chunk[0][0])]
                else:
                    outs = generate.generate_batch([(imgs[iid], iid) for iid, _ in chunk])
                call_s.append(time.perf_counter() - t0)
                for (iid, _), o in zip(chunk, outs):
                    recs[iid] = _record(o, imgs[iid])
                print(f"[batch {n}] {len(recs)}/{len(rows)} 행 · 호출 {call_s[-1]:.1f} s", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            return {"status": "oom", "rows_done": len(recs), "error": str(exc)[:200],
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        total = sum(call_s)
        return {"status": "ok", "records": recs, "calls": len(call_s), "total_s": round(total, 1),
                "s_per_row": round(total / len(rows), 2), "s_per_call": round(statistics.mean(call_s), 2),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}

    result["passes"]["1"] = run_pass(1)
    base = result["passes"]["1"]["records"]
    for n in batches:
        p = run_pass(n)
        if p["status"] == "ok":
            cmp = {iid: _compare(base[iid], p["records"][iid]) for iid in base}
            p["compare_vs_batch1"] = {
                "tokens_equal_rate": sum(c["tokens_equal"] for c in cmp.values()) / len(cmp),
                "text_equal_rate": sum(c["text_equal"] for c in cmp.values()) / len(cmp),
                "parse_equal_rate": sum(c["parse_equal"] for c in cmp.values()) / len(cmp),
                "stop_equal_rate": sum(c["stop_equal"] for c in cmp.values()) / len(cmp),
                "first_divergence": {iid: c["first_divergence"] for iid, c in cmp.items() if not c["tokens_equal"]},
                "speedup_vs_batch1": round(result["passes"]["1"]["s_per_row"] / p["s_per_row"], 2) if p["s_per_row"] else None,
            }
        result["passes"][str(n)] = p
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    if a.reference is not None and a.reference.is_file():
        ref = {}
        for line in a.reference.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            ref[d["image_id"]] = d
        both = [iid for iid in base if iid in ref]
        result["reference_vs_local_batch1"] = {
            "reference": a.reference.as_posix(), "reference_sha256": _sha(a.reference.read_bytes()), "n_both": len(both),
            "text_equal_rate": sum(base[i]["text"] == ref[i]["text"] for i in both) / len(both) if both else None,
            "stop_equal_rate": sum(base[i]["gen_stop"] == ref[i]["gen_stop"] for i in both) / len(both) if both else None,
            "n_new_tokens_equal_rate": sum(base[i]["n_new_tokens"] == ref[i]["n_new_tokens"] for i in both) / len(both) if both else None,
            "note": "A100 배치 1 과 로컬 배치 1 — 장비 · 커널 차이의 크기. 묶음 차이와 견주는 잣대이고 어느 쪽이 맞다는 뜻이 아니다",
        }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
