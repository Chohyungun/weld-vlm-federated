"""통합형 학습 경로의 합성 대역 — 모델 없이 `train_rounds` 를 실제로 돌린다.

- `TinyLM` + LoRA(`peft`): 실제 적재기가 내는 것과 같은 이름 규칙(`...lora_A.default.weight` ↔ `...lora_A.weight`)을
  갖는 작은 모델이다. CPU 에만 두어 CUDA 를 건드리지 않는다.
- `FakeProcessor`: 채팅 템플릿을 흉내 낸다. 생성 접두가 학습 렌더의 토큰 접두가 되게 만들고(감독 마스킹의
  전제), 이미지가 있으면 `image_grid_thw = [1, h//16, w//16]` 를 낸다. 받은 템플릿 인자를 기록한다.
- `CountingLoader`: 불린 횟수를 센다 — "대역을 부르기 전에 거부한다" 를 호출 수 0 으로 확인한다.

HF 캐시에 기대지 않는다. 실제 데이터를 읽지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

VOCAB = 64


class TinyLM(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.proj = nn.Linear(d, d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, input_ids, attention_mask=None, image_grid_thw=None, logits_to_keep=0, **kw):
        logits = self.head(torch.tanh(self.proj(self.emb(input_ids))))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


class _ImageProcessor:
    def __init__(self, **over):
        self.cfg = {"min_pixels": 1024, "max_pixels": 1 << 20, "patch_size": 16, "merge_size": 2, **over}

    def to_dict(self):
        return dict(self.cfg)


def _text_ids(text: str, limit: int = 12) -> list[int]:
    return [10 + (ord(c) % 50) for c in text[:limit]]


class FakeProcessor:
    """`apply_chat_template` 만 흉내 낸다. user=1 · image=3 · assistant/생성 시작=4 · 끝=5."""

    def __init__(self, **image_over):
        self.image_processor = _ImageProcessor(**image_over)
        self.calls: list[dict] = []

    def apply_chat_template(self, msgs, tokenize=True, return_dict=True, return_tensors=None,
                            add_generation_prompt=False, **kwargs):
        self.calls.append(dict(kwargs))
        ids: list[int] = []
        grid = None
        for m in msgs:
            if m["role"] == "user":
                ids.append(1)
                for c in m["content"]:
                    if c["type"] == "image":
                        w, h = c["image"].size
                        grid = [1, max(h // 16, 1), max(w // 16, 1)]
                        ids.append(3)
                    else:
                        ids.extend(_text_ids(c["text"]))
            else:
                ids.append(4)
                ids.extend(_text_ids(m["content"][0]["text"], limit=48))
                ids.append(5)
        if add_generation_prompt:
            ids.append(4)
        if return_tensors == "pt":
            out = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            if grid is not None:
                out["image_grid_thw"] = torch.tensor([grid])
            return out
        return {"input_ids": [ids]}


def fake_loader(model_id=None, *, init_seed, revision=None, processor_kwargs=None):
    """실제 적재기와 같은 시그니처. 기저 가중치는 고정 시드, LoRA A 는 `init_seed` 아래에서 만든다."""
    from peft import LoraConfig, get_peft_model

    from fl.seeding import seeded

    with seeded(12345):
        base = TinyLM()
    with seeded(int(init_seed)):
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
                                                target_modules=["proj"]))
    return model, FakeProcessor(**dict(processor_kwargs or {}))


class CountingLoader:
    def __init__(self, fn=fake_loader):
        self.fn = fn
        self.n = 0

    def __call__(self, *a, **kw):
        self.n += 1
        return self.fn(*a, **kw)


def make_rows(tmp: Path, client: str, n: int, *, w: int = 64, h: int = 32, start: int = 0) -> list[dict]:
    """합성 이미지(단색 PNG)와 페어 행. 결함 박스는 원본 픽셀이다."""
    from PIL import Image

    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(start, start + n):
        p = tmp / f"{client}_{i}.png"
        if not p.exists():
            Image.new("RGB", (w, h), (i * 7 % 255, 30, 60)).save(p)
        rows.append({"image_id": f"{client}_{i:04d}", "image_path": str(p), "client": client, "split": "train",
                     "skeleton": {"defects": [{"type": "2011", "bbox_px": [1, 2, 10 + i % 5, 12]}],
                                  "verdict": "불합격", "clauses": []}})
    return rows


def write_pairs(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path
