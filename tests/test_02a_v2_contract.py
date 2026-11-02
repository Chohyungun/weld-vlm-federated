"""02a 의 2 판 입력 계약(C 08 2 판 §6-1 · 221 I-01 — 시험 18 · 19).

- 계측 전에 멈춘다: 같은 폴더의 스냅샷 계약이 페어 · 메타의 바이트와 `--pairs-digest` 에 맞지 않음(다른 폴더의 메타로 바꾼 사본 포함) ·
  계약서가 없음 · 2 판 메타의 신원이 실제 파일과 다름 · 프롬프트 · 회귀 입력의 sha 불일치 · 인자 일부만. 그때는 출력 파일이 없다.
- 1 판 입력을 2 판 인자로 주면 1 판 판(상수 · 기준 설정)과 같은 렌더 입력으로 풀린다 — 계측 규칙은 두 판이 같다.
- 실제 프로세서(캐시 · CPU · 망 없음)로 — `--prompt` 로 준 판이 채팅 렌더(B)와 학습 렌더 `_encode`(C)에 실제로 쓰이고, 표본 행마다 감독
  구간 = 타깃 토큰 + 2 다. 기준 설정을 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from corpus.generate import make_pairs_main as MM
from corpus.generate import make_pairs_pilot as P
from corpus.generate import pairs_text_v2 as T2
from corpus.generate.run_cycle_corpus import defect_names
from corpus.rules.limits_loader import load_limits
from vlm.pilot_vlm import load_prompt

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "docs/dev_log/2026-10-20-설계복구/02a_타깃길이_C.py"
PROMPT = "vlm/prompts/unified_v3_1_basis.txt"
#: 구판 회귀는 S13과 같은 보존 입력을 쓴다. 현재 base는 새 레시피의 기대값을 담으므로 대신 쓰지 않는다.
BASE_REL = "outputs/diag_20261027/base_b16e2306.yaml"
BASE_SHA256 = "052e73116d0ad6d51a54e3350af856b899cc20c7a79fb6727d631dd7d4c2f511"


@pytest.fixture(scope="module")
def m02a():
    spec = importlib.util.spec_from_file_location("t02a_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rows(tmp: Path, n: int = 30) -> list[dict]:
    """합성 레코드 — 빌더의 조립 함수(2 판 서술)로 짓는다. 이미지는 작은 합성 그림(경계 안 상자)."""
    from PIL import Image

    table = load_limits(MM.EDITIONS["v2"].limits_csv, pilot=True)
    names = defect_names()
    codes = [["2011"], ["301", "2011"], ["100"], [], ["401"], ["2011", "100"]]
    out = []
    for i in range(n):
        c = codes[i % len(codes)]
        mat = "AL" if i % 7 == 3 and c else "ST"
        img = tmp / "img" / f"i{i:03d}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (i % 255, 0, 0)).save(img)
        anns = [{"ann_id": f"i{i:03d}#{k}", "geom_valid": True, "iso_code": code, "bbox_x1_px": 2 + k, "bbox_y1_px": 3,
                 "bbox_x2_px": 20 + k, "bbox_y2_px": 30, "major_axis_px": 10.0, "equiv_diameter_px": 5.0}
                for k, code in enumerate(c)]
        row = {"image_id": f"i{i:03d}", "rel_path": str(img), "client": ("C1", "C2", "C3")[i % 3],
               "split": "val" if i % 5 == 4 else "train", "material": mat, "width_px": 64, "height_px": 48,
               "has_defect": bool(c)}
        base = T2.augment_basis(P.clause_basis(table, mat), table, mat)
        rec, _ = P.assemble_record(row, anns, names, base, edition="v2")
        out.append(rec)
    return out


def _folder(tmp: Path, rows: list[dict], *, meta: dict | None = None) -> tuple[Path, str]:
    """페어 폴더 — 레코드 · 회계 · 폐기 · 2 판 메타(실제 파일에서 낸 신원)와 계약서. (pairs.jsonl, digest)."""
    d = tmp / "pairs_v2"
    d.mkdir(parents=True)
    P.write_jsonl(d / "pairs.jsonl", rows)
    P.write_jsonl(d / "discarded.jsonl", [])
    P.write_json(d / "counts.json", {"synthetic": True})
    ed = MM.EDITIONS["v2"]
    lt = {"path": ed.limits_rel, "sha256": _sha(ed.limits_csv), "status": ed.limits_status}
    by_split: dict = {}
    for r in rows:
        cell = by_split.setdefault(r["split"], {}).setdefault(r["client"], {"n_total": 0})
        cell["n_total"] += 1
    P.write_json(d / "PAIRS_META.json", meta or {"limits_table": lt, "identity": MM.identity_block(lt), "text_contract": "v2",
                                                 "counts_by_split": by_split})
    digest = P.snapshot_digest(d, ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json"))
    return d / "pairs.jsonl", digest


def _args(m02a, out: Path, pairs: Path, digest: str, **over):
    from vlm.uni_config import load_uni_config

    cfg = load_uni_config(REPO / "configs/base.yaml")
    assert _sha(REPO / BASE_REL) == BASE_SHA256
    vals = {"--pairs": str(pairs), "--pairs-digest": digest, "--prompt": str(REPO / PROMPT),
            "--prompt-sha256": load_prompt(REPO / PROMPT)[1], "--model-id": cfg.model_id, "--revision": cfg.model_revision,
            "--processor-kwargs": json.dumps(dict(cfg.processor_kwargs)),
            "--chat-template-kwargs": json.dumps(dict(cfg.chat_template_kwargs)),
            "--regress-base": str(REPO / BASE_REL), "--regress-base-sha256": _sha(REPO / BASE_REL),
            "--regress-07": str(REPO / m02a.REPORT07), "--regress-07-sha256": _sha(REPO / m02a.REPORT07)}
    vals.update(over)
    argv = [str(out)]
    for k, v in vals.items():
        if v is not None:
            argv += [k, v]
    return argv


# ---------------------------------------------------------------- 시험 18 — 계측 전에 멈춘다

def test_맞는_2판_입력은_계측_전_검사를_지난다(m02a, tmp_path):
    pairs, digest = _folder(tmp_path, _rows(tmp_path, 6))
    inp, bad = m02a.resolve_inputs(m02a.parse_args(_args(m02a, tmp_path / "o.json", pairs, digest)))
    assert bad == [] and inp["mode"] == "v2" and inp["pairs_meta_edition"] == "v2" and inp["identity_failures"] == []


@pytest.mark.parametrize("label", ["다른 폴더의 메타", "digest 어긋남", "계약서 없음", "인자 일부만", "프롬프트 sha",
                                   "회귀 입력 sha", "2 판 메타 신원"])
def test_계약에_맞지_않는_입력은_출력_없이_멈춘다(m02a, tmp_path, label):
    rows = _rows(tmp_path, 6)
    pairs, digest = _folder(tmp_path, rows)
    over: dict = {}
    if label == "다른 폴더의 메타":
        other = tmp_path / "other.json"
        P.write_json(other, {"limits_table": {"path": MM.EDITIONS["v1"].limits_rel}})
        shutil.copyfile(other, pairs.parent / "PAIRS_META.json")       # 계약서는 그대로 — 메타만 바꾼 스크래치 사본
    elif label == "digest 어긋남":
        over["--pairs-digest"] = "0" * 64
    elif label == "계약서 없음":
        (pairs.parent / "SNAPSHOT.sha256").unlink()
    elif label == "인자 일부만":
        over["--regress-07"] = None
    elif label == "프롬프트 sha":
        over["--prompt-sha256"] = "0" * 64
    elif label == "회귀 입력 sha":
        over["--regress-base-sha256"] = "0" * 64
    elif label == "2 판 메타 신원":
        meta = json.loads((pairs.parent / "PAIRS_META.json").read_text(encoding="utf-8"))
        meta["identity"]["generator"]["files"][0]["sha256"] = "0" * 64
        P.write_json(pairs.parent / "PAIRS_META.json", meta)
        P.snapshot_digest(pairs.parent, ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json"))
        digest = next(ln.split()[-1] for ln in (pairs.parent / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines()
                      if ln.startswith("# snapshot_digest"))
    out = tmp_path / "o.json"
    argv = _args(m02a, out, pairs, digest, **over)
    _inp, bad = m02a.resolve_inputs(m02a.parse_args(argv))
    assert bad, label
    assert m02a.main(argv) == 2 and not out.exists()


def test_1판_입력을_2판_인자로_주면_1판과_같은_렌더_입력으로_풀린다(m02a, tmp_path, monkeypatch):
    assert _sha(REPO / BASE_REL) == BASE_SHA256
    monkeypatch.setattr(m02a, "BASE", REPO / BASE_REL)
    pairs = REPO / m02a.PAIRS
    if not pairs.is_file():
        pytest.skip("1 판 봉인 페어가 이 트리에 없다")
    from vlm.uni_config import load_uni_config

    cfg = load_uni_config(REPO / m02a.BASE)
    v1, bad1 = m02a.resolve_inputs(m02a.parse_args([str(tmp_path / "o.json")]))
    v2, bad2 = m02a.resolve_inputs(m02a.parse_args(_args(m02a, tmp_path / "o.json", pairs, cfg.pairs_digest,
                                                         **{"--prompt": str(REPO / cfg.prompt_path),
                                                            "--prompt-sha256": load_prompt(REPO / cfg.prompt_path)[1]})))
    assert bad1 == [] and bad2 == [] and v2["pairs_meta_edition"] == "v1"
    for k in ("model_id", "revision", "processor_kwargs", "chat_template_kwargs"):
        assert v1[k] == v2[k], k
    assert load_prompt(v1["prompt"])[1] == load_prompt(v2["prompt"])[1]
    assert Path(v1["pairs"]).resolve() == Path(v2["pairs"]).resolve() and Path(v1["meta"]).resolve() == Path(v2["meta"]).resolve()


# ---------------------------------------------------------------- 시험 19 — 신판 프롬프트의 렌더 대조(실제 프로세서)

def _cached_processor_or_skip(model_id: str, revision: str | None):
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(model_id, "preprocessor_config.json", revision=revision), str):
        pytest.skip("프로세서 캐시가 없다")


def test_준_프롬프트가_두_렌더에_쓰이고_감독_구간은_타깃_더하기_둘이다(m02a, tmp_path, monkeypatch):
    from vlm.uni_config import load_uni_config

    cfg = load_uni_config(REPO / "configs/base.yaml")
    _cached_processor_or_skip(cfg.model_id, cfg.model_revision)
    rows = _rows(tmp_path, 30)
    pairs, digest = _folder(tmp_path, rows)
    monkeypatch.setattr(m02a, "BASE", str(tmp_path / "없는_기준_설정.yaml"))     # 2 판 입력 계약은 기준 설정을 읽지 않는다
    monkeypatch.chdir(REPO)
    out = tmp_path / "o.json"
    argv = _args(m02a, out, pairs, digest)
    assert m02a.main(argv) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    want = load_prompt(REPO / PROMPT)
    assert doc["tokenizer"]["prompt_sha256_text"] == want[1] == doc["input_contract"]["render_check"]["prompt_sha256_text"]
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.model_revision, local_files_only=True,
                                         **dict(cfg.processor_kwargs))
    kw = dict(cfg.chat_template_kwargs)

    def prefix_len(text: str) -> int:
        user = {"role": "user", "content": [{"type": "text", "text": text}]}
        return len(proc.tokenizer(proc.apply_chat_template([user], tokenize=False, add_generation_prompt=True, **kw),
                                  add_special_tokens=False)["input_ids"])

    assert doc["input_contract"]["render_check"]["gen_prefix_len"] == prefix_len(want[0])
    assert prefix_len(want[0]) != prefix_len(load_prompt(REPO / "vlm/prompts/unified_v2_absorig.txt")[0])
    encs = doc["cross_check"]["encode_C"]
    assert all(encs["token_identity_ok"][f] == encs["rows"] for f in ("v1", "ga", "na"))
    assert all(encs["supervised_count_equal"][f] == encs["rows"] for f in ("v1", "ga", "na"))
    assert doc["checks"]["template_B_all_match"] and doc["checks"]["encode_C_all_match"]
    assert doc["inputs"]["prompt"]["sha256"] == _sha(REPO / PROMPT) and doc["inputs"]["pairs"]["sha256"] == _sha(pairs)
