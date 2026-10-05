"""채팅 템플릿 고정안 실측 — 학습 렌더와 생성 접두가 토큰 단위로 일치하는가 (02번 I-4).

## 왜 이것을 재는가

`vlm/pilot_vlm.py` 의 감독 마스킹은 **생성 접두 길이**로 자른다(`_encode`).

    prompt_only = apply_chat_template([user], add_generation_prompt=True)
    labels[:, :len(prompt_only)] = -100

이 산술은 "생성 접두가 학습 렌더의 토큰 접두와 같다"는 전제 위에 있다. 전제가 깨지면
두 가지가 동시에 틀어진다.

1. 감독 구간이 어긋나 엉뚱한 위치에 손실이 걸린다(마스킹이 길이로만 자르므로).
2. 추론이 학습에서 본 적 없는 문맥에서 생성을 시작한다.

Qwen3.5 계열의 템플릿은 `enable_thinking` **미지정 시의 기본값이 모델마다 반대**다.
0.8B·2B 는 미지정이면 비생각(`<think>\\n\\n</think>\\n\\n`), 4B 는 미지정이면
생각 모드(`<think>\\n`)를 붙인다. 학습 렌더(assistant 턴)는 두 모델 모두 비생각
블록을 넣는다. 그래서 **4B 만 전제가 깨진다** — 같은 코드로 0.8B 파일럿은 통과하고
4B 본실험은 조용히 어긋난다. 70번·75번이 이 차이를 "토크나이저 구성이 조금 다르다"
로 적었는데, 두 모델의 `tokenizer.json`·`merges.txt` 는 바이트가 같다(이 스크립트가
해시로 확인한다).

## 무엇을 고정안으로 내는가

`enable_thinking=False` 를 **두 호출에 모두 명시**한다. 미지정에 기대지 않는다 —
기본값이 모델·판마다 갈리는 것이 이 사고의 원인이었다.

    uv run python scripts/probe/template_prefix.py

산출: `outputs/probe_c/template_prefix.json`. GPU 를 쓰지 않는다(토크나이저·프로세서만).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8")

OUT = Path("outputs/probe_c/template_prefix.json")

#: 파일럿 모델 · 크기 곡선 프로브 모델 · 본실험 모델. 셋에서 같은 시험이 통과해야 한다.
MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"]

#: 시험할 설정. `None` 은 인자를 주지 않는 것(현행 코드) — 이것이 깨지는 것을 보인다.
MODES: list[tuple[str, bool | None]] = [
    ("미지정", None),
    ("enable_thinking=False", False),
    ("enable_thinking=True", True),
]

#: 실제 학습 타깃 모양. 좌표는 `ABS_ORIG` 정수다(`vlm/pilot_vlm.build_target`).
TARGET = ('{"defects":[{"iso_code":"2011","bbox_2d":[40,344,120,408]}],'
          '"verdict":"판정불가","cited_clauses":[]}')


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _snapshot_dir(model_id: str) -> Path | None:
    """로컬 HF 캐시의 스냅샷 디렉터리. 없으면 None(해시 대조를 건너뛴다)."""
    name = "models--" + model_id.replace("/", "--")
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / name
    snaps = sorted((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else []
    return snaps[0] if snaps else None


def check_model(model_id: str, prompt: str) -> dict:
    """한 모델에서 세 설정을 잰다. 토크나이저만 쓴다 — 이미지 토큰은 접두 등식에 무관하다."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    user = {"role": "user", "content": prompt}
    asst = {"role": "assistant", "content": TARGET}

    modes = {}
    for label, flag in MODES:
        kw = {} if flag is None else {"enable_thinking": flag}
        full_txt = tok.apply_chat_template([user, asst], tokenize=False, **kw)
        gen_txt = tok.apply_chat_template([user], tokenize=False,
                                          add_generation_prompt=True, **kw)
        full = tok(full_txt, add_special_tokens=False)["input_ids"]
        gen = tok(gen_txt, add_special_tokens=False)["input_ids"]
        is_prefix = full[:len(gen)] == gen
        sup = full[len(gen):]
        modes[label] = {
            "생성접두_끝": gen_txt[-32:],
            "생성접두가_학습렌더의_토큰접두인가": is_prefix,
            "n_full": len(full), "n_gen_prefix": len(gen), "n_supervised": len(sup),
            "감독_첫_토큰": [tok.decode([t]) for t in sup[:4]],
            # 접두가 아니면 마스킹이 자를 자리가 학습 렌더의 어느 토큰인지도 남긴다.
            "길이로_자른_자리의_토큰": None if is_prefix else tok.decode(full[len(gen) - 1:len(gen) + 1]),
        }

    snap = _snapshot_dir(model_id)
    return {
        "model_id": model_id,
        "스냅샷": None if snap is None else snap.name[:12],
        "tokenizer_json_sha256_12": _file_digest(snap / "tokenizer.json") if snap else None,
        "merges_txt_sha256_12": _file_digest(snap / "merges.txt") if snap else None,
        "chat_template_sha256_12": _file_digest(snap / "chat_template.jinja") if snap else None,
        "설정별": modes,
    }


def check_processor(model_id: str, prompt: str) -> dict:
    """프로세서 경로에서도 인자가 템플릿까지 전달되는가 — 학습·추론이 쓰는 호출이다.

    `vlm/pilot_vlm._encode` 와 `scripts/pilot_export_vlm.main` 은 토크나이저가 아니라
    **프로세서**의 `apply_chat_template` 을 부른다. 이미지 자리표시가 들어가므로 여기서도
    같은 등식이 성립하는지 따로 본다. 합성 단색 이미지를 쓴다(평가 자산 무접촉).
    """
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    img = Image.new("RGB", (1280, 720), "white")
    user = {"role": "user", "content": [{"type": "image", "image": img},
                                        {"type": "text", "text": prompt}]}
    asst = {"role": "assistant", "content": [{"type": "text", "text": TARGET}]}

    out = {}
    for label, flag in MODES:
        kw = {} if flag is None else {"enable_thinking": flag}
        full = proc.apply_chat_template([user, asst], tokenize=True, return_dict=True,
                                        return_tensors="pt", **kw)
        gen = proc.apply_chat_template([user], tokenize=True, return_dict=True,
                                       return_tensors="pt", add_generation_prompt=True, **kw)
        f = full["input_ids"][0].tolist()
        g = gen["input_ids"][0].tolist()
        out[label] = {
            "생성접두가_학습렌더의_토큰접두인가": f[:len(g)] == g,
            "n_full": len(f), "n_gen_prefix": len(g), "n_supervised": len(f) - len(g),
        }
    return out


def main() -> int:
    prompt = Path("vlm/prompts/unified_v2_absorig.txt").read_text(encoding="utf-8")
    print(f"프롬프트 {len(prompt)}자 · 타깃 {len(TARGET)}자", flush=True)

    rows = []
    for mid in MODELS:
        print(f"\n=== {mid}", flush=True)
        try:
            row = check_model(mid, prompt)
        except Exception as exc:                      # noqa: BLE001
            print(f"  토크나이저 실패: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"model_id": mid, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for label, r in row["설정별"].items():
            print(f"  {label:24s} 접두일치={r['생성접두가_학습렌더의_토큰접두인가']!s:5s} "
                  f"감독토큰={r['n_supervised']:4d}  끝={r['생성접두_끝']!r}", flush=True)
        try:
            row["프로세서_경로"] = check_processor(mid, prompt)
            for label, r in row["프로세서_경로"].items():
                print(f"    [프로세서] {label:22s} 접두일치="
                      f"{r['생성접두가_학습렌더의_토큰접두인가']!s:5s} 감독토큰={r['n_supervised']:4d}",
                  flush=True)
        except Exception as exc:                      # noqa: BLE001
            row["프로세서_경로"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    [프로세서] 실패: {type(exc).__name__}: {exc}", flush=True)
        rows.append(row)

    ok = [r for r in rows if "설정별" in r]
    fixed = "enable_thinking=False"
    all_pass = bool(ok) and all(
        r["설정별"][fixed]["생성접두가_학습렌더의_토큰접두인가"] for r in ok)
    현행_실패 = [r["model_id"] for r in ok
               if not r["설정별"]["미지정"]["생성접두가_학습렌더의_토큰접두인가"]]
    같은_토크나이저 = len({r["tokenizer_json_sha256_12"] for r in ok
                     if r["tokenizer_json_sha256_12"]}) == 1

    rep = {
        "무엇을_쟀나": "학습 렌더와 생성 접두의 토큰 단위 일치 — 감독 마스킹의 전제",
        "고정안": f"{fixed} 를 학습·추론 두 호출에 모두 명시한다",
        "고정안_전_모델_통과": all_pass,
        "현행(미지정)_실패_모델": 현행_실패,
        "토크나이저_동일": 같은_토크나이저,
        "타깃": TARGET,
        "결과": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n고정안({fixed}) 전 모델 통과: {all_pass}")
    print(f"현행(미지정) 실패: {현행_실패 or '없음'}")
    print(f"tokenizer.json 해시 동일: {같은_토크나이저} → {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
