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
                            add_generation_prompt=False, padding=False, **kwargs):
        """대화 하나(메시지 목록) 또는 대화 목록(묶음 — 메시지 목록의 목록). 묶음은 `padding` 으로 채운다(pad 0 · 마스크 0,
        방향은 토크나이저의 `padding_side`, 없으면 왼쪽) — HF 프로세서의 묶음 렌더 꼴. 한 대화의 출력은 종전과 같다."""
        self.calls.append(dict(kwargs))
        batched = bool(msgs) and isinstance(msgs[0], list)
        rows, grids = [], []
        for conv in (msgs if batched else [msgs]):
            ids: list[int] = []
            grid = None
            for m in conv:
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
            rows.append(ids)
            grids.append(grid)
        if return_tensors == "pt":
            length = max(len(r) for r in rows)
            if len(rows) > 1 and not padding and any(len(r) != length for r in rows):
                raise ValueError("묶음의 길이가 다른데 padding 이 없다")
            side = getattr(getattr(self, "tokenizer", None), "padding_side", "left")
            ids_t = torch.zeros(len(rows), length, dtype=torch.long)
            mask = torch.zeros(len(rows), length, dtype=torch.long)
            for i, r in enumerate(rows):
                sl = slice(length - len(r), length) if side == "left" else slice(0, len(r))
                ids_t[i, sl] = torch.tensor(r)
                mask[i, sl] = 1
            out = {"input_ids": ids_t, "attention_mask": mask}
            if all(g is not None for g in grids):
                out["image_grid_thw"] = torch.tensor(grids)
            return out
        return {"input_ids": rows}


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


def use_registration(monkeypatch, tmp: Path, seeds=(1,)) -> Path:
    """임시 등록 파일(seed_indices 만 시험이 고른다)을 학습 쪽 판독기의 기본 자리로 — 실행기 명령줄의 시드 경계
    (`scripts/main_uni.registered_seeds`)가 저장소의 등록 파일(데이터 쪽 소유)을 읽지 않게 한다(§137 반영의 시험 격리)."""
    import yaml

    from vlm import pilot_vlm
    from vlm import registration as R

    root = tmp / "_registration_root"
    (root / "configs").mkdir(parents=True, exist_ok=True)
    body = {"kind": R.KIND, "version": R.VERSION, "seed_indices": list(seeds), "num_rounds": 1, "local_epochs": 1,
            "total_epochs": 1, "recipe": R.TRAIN_RECIPES[0], "eval_gen_limit": 512,
            "quantization": dict(pilot_vlm.QUANT_4BIT), "micro_batch": pilot_vlm.MICRO_BATCH,
            "grad_accum": pilot_vlm.GRAD_ACCUM, "lr": pilot_vlm.LR,
            "early_coord_check_rules": {"version": None, "sha256": None}, "init_adapter_digest": None}
    p = root / R.REG_PATH
    p.write_text(yaml.safe_dump(body, allow_unicode=True, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(R, "REPO_ROOT", root)
    return p


#: 합성 토크나이저 환경의 파일 — 이름은 실제 스냅샷의 것(`vlm.token_table.IDENTITY_FILES` 가운데 넷), 바이트는 합성이다.
FAKE_TOKENIZER_FILES = {"tokenizer.json": b'{"model": "bytes"}\n', "tokenizer_config.json": b'{"tokenizer_class": "Bytes"}\n',
                        "chat_template.jinja": b"{{ messages }}\n", "preprocessor_config.json": b'{"patch_size": 16}\n'}


def fake_tokenizer_env(monkeypatch, tmp: Path, *, transformers: str = "9.9.9", tokenizers: str = "0.0.1") -> Path:
    """토큰 표 출처의 토크나이저 신원 이음새(외부 검토 200 M-01) — 합성 토크나이저 · 프로세서 파일을 `tmp` 아래 한 폴더에 쓰고,
    `vlm.token_table` 의 캐시 해석(어느 모델 · 판이든 그 폴더) · 꾸러미 판을 그것으로 바꾼다. HF 캐시를 읽지 않는다.

    돌려주는 값은 그 폴더다 — 시험이 파일 바이트를 바꿔 '같은 판 이름 · 다른 토크나이저' 를 만든다."""
    from vlm import token_table as T

    d = tmp / "_hf_snapshot"
    d.mkdir(parents=True, exist_ok=True)
    for name, raw in FAKE_TOKENIZER_FILES.items():
        if not (d / name).exists():
            (d / name).write_bytes(raw)
    versions = {"transformers": transformers, "tokenizers": tokenizers}
    monkeypatch.setattr(T, "_cached_file", lambda model_id, revision, name: (d / name) if (d / name).is_file() else None)
    monkeypatch.setattr(T, "_package_version", lambda name: versions.get(name))
    return d


def use_v2_token_table(monkeypatch, tmp: Path, *, pairs_digest: str, model_id: str = "tiny", revision: str | None = "r1",
                       rows: list[dict] | None = None, prov_over: dict | None = None) -> Path:
    """새 레시피(`json_line_then_basis`)의 행별 토큰 표 자리를 합성 표 · 출처로 바꿔 끼운다 — 출처는 합성 토크나이저 환경
    (`fake_tokenizer_env`)에서 계측 함수 그대로(`vlm.token_table` — 토큰은 타깃의 UTF-8 바이트) 낸 꼴이라, 표 검사
    (`vlm.budget.token_table_problems`)가 실물 표 · 실물 캐시에 기대지 않는다. `rows` 를 주지 않으면 머리 줄만 있는 표다.
    `prov_over` 는 출처 칸을 덮는다(어긋남 시험). 돌려주는 값은 표 파일."""
    import hashlib

    from vlm import budget as B
    from vlm import token_table as T

    fake_tokenizer_env(monkeypatch, tmp)
    recipe = "json_line_then_basis"
    counts = T.count_rows(rows, lambda ts: [list(t.encode("utf-8")) for t in ts], recipe=recipe) if rows else []
    raw = T.table_bytes(counts)
    table = tmp / "_token_table_v2" / "per_row.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_bytes(raw)
    prov = T.provenance(recipe=recipe, counts=counts, table=raw, pairs_path="pairs.jsonl", pairs_sha256="0" * 64,
                        pairs_digest=pairs_digest, tokenizer_identity=T.tokenizer_identity(model_id, revision))
    prov.update(prov_over or {})
    (table.parent / (table.name + B.PROVENANCE_SUFFIX)).write_text(json.dumps(prov), encoding="utf-8")
    monkeypatch.setitem(B.TOKEN_TABLES, recipe, (table, hashlib.sha256(raw).hexdigest()))
    return table
