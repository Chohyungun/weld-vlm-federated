"""단계 시한의 측정 · 제안 도구(`scripts/rehearsal_timeouts.py`, 결정 19 의 R-1). 모델 없이 CPU 로 돈다.

기대값은 손셈이다 — 크기를 고정한 합성 계획(저장소 계획의 지금 값에 기대지 않는다)에 머리말의 식을 그대로 적용했다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.rehearsal_timeouts as RT


def _plan(tmp_path, name: str, **over):
    """저장소 계획에서 크기 칸만 고정한 합성 계획."""
    from vlm.rehearsal_plan import load_plan

    doc = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    for dotted, v in over.items():
        head, key = dotted.split(".")
        doc[head][key] = v
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_plan(p, purpose=doc["kind"]), p


def test_리허설_시한의_제안은_머리말의_식이다(tmp_path):
    plan, _ = _plan(tmp_path, "rehearsal_uni.yaml", **{"train.rows": {"C1": 100, "C2": 60, "C3": 30},
                                                      "gen.max_n": 64, "gen_limit.max_new_tokens": 512})
    out = RT.propose(plan, load_s=600)
    # 중앙 190 × 2 × 3.621 + 600 = 1,975.98 · 연합 2 × (3 × 600 + 190 × 1 × 3.621) = 4,975.98 → 1.5 × 4,975.98 + 60 = 7,523.97 → 7,600
    # export 1.5 × (64 × (0.816 + 512 / 5.87) + 600) = 9,351.8 → 9,400 · register 1.5 × 1,200 + 600 = 2,400
    assert out["timeouts_s"] == {"lists": 1800, "register": 2400, "train": 7600, "export": 9400, "score": 1800}
    assert out["inputs"]["train_per_cell_s"]["uni_fed"] == 4976.0 and out["inputs"]["rows"]["uni_central"] == 190
    assert out["plan_sha256"] == plan.sha256 and out["purpose"] == "rehearsal"


def test_진단_시한의_제안은_중앙_한_칸이다(tmp_path):
    plan, _ = _plan(tmp_path, "frame_diag_uni.yaml", **{"train.rows": {"central": 4000}, "gen.max_n": 480,
                                                       "gen_limit.max_new_tokens": 512})
    out = RT.propose(plan, load_s=600)
    # 1.5 × (4,000 × 1 × 3.621 + 600) + 60 = 22,686 → 22,700 · 1.5 × (480 × 88.0392 + 600) = 64,288.2 → 64,300
    assert out["timeouts_s"]["train"] == 22700 and out["timeouts_s"]["export"] == 64300
    assert set(out["inputs"]["train_per_cell_s"]) == {"uni_central"}


def test_적재_시간을_재고_배타로_쓴다(tmp_path):
    ticks = iter([0.0, 5.0, 10.0, 13.0])
    calls = []

    def loader(model_id, **kw):
        calls.append((model_id, kw))
        return object(), object()

    rep = RT.measure_load(repeats=2, loader=loader, clock=lambda: next(ticks))
    assert rep["load_s"] == [5.0, 3.0] and rep["load_s_max"] == 5.0 and len(calls) == 2
    assert set(calls[0][1]) == {"revision", "processor_kwargs"}               # 본실험 설정의 판 · 프로세서 설정으로 연다
    out = tmp_path / "load.json"
    assert RT.main(["measure-load", "--out", str(out), "--repeats", "1"], loader=loader) == RT.EXIT_OK
    assert RT.main(["measure-load", "--out", str(out)], loader=loader) == RT.EXIT_REFUSED      # 덮지 않는다


def test_잰_값으로_제안하고_계획_파일은_쓰지_않는다(tmp_path, capsys):
    _, path = _plan(tmp_path, "rehearsal_uni.yaml", **{"train.rows": {"C1": 100, "C2": 60, "C3": 30},
                                                      "gen.max_n": 64, "gen_limit.max_new_tokens": 512})
    before = path.read_bytes()
    lj = tmp_path / "load.json"
    lj.write_text(json.dumps({"load_s_max": 600}), encoding="utf-8")
    assert RT.main(["propose", "--plan", str(path), "--load-json", str(lj)]) == RT.EXIT_OK
    assert json.loads(capsys.readouterr().out)["timeouts_s"]["train"] == 7600
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["rehearsal_uni.yaml", "frame_diag_uni.yaml"])
def test_저장소의_계획으로_제안이_선다(name):
    """저장소 계획의 지금 크기로 다섯 단계 모두 값이 난다 — 값 자체는 보지 않는다(크기는 계획이 바꿀 수 있다)."""
    from vlm.rehearsal_plan import load_plan

    kind = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))["kind"]
    out = RT.propose(load_plan(ROOT / "configs" / name, purpose=kind), load_s=600)
    assert set(out["timeouts_s"]) == set(RT.STAGES) and all(type(v) is int and v > 0 for v in out["timeouts_s"].values())


# ================================================================ 실제 목록의 누락은 제안을 거부한다(외부 검토 §51)
def _lists(root: Path, rows: dict[str, int]) -> Path:
    lists = root / "lists"
    lists.mkdir(parents=True)
    for tag, n in rows.items():
        (lists / f"train_{tag}.jsonl").write_text("".join(json.dumps({"image_id": f"x{k}"}) + "\n" for k in range(n)),
                                                 encoding="utf-8")
    return root


FULL = {"uni_local_C1": 101, "uni_local_C2": 61, "uni_local_C3": 33, "uni_central": 195}


@pytest.mark.parametrize(("case", "code"), [
    ("no_root", "root_missing"), ("no_lists", "lists_missing"), ("missing_C2", "train_list_missing"),
    ("empty_C3", "train_list_empty"), ("missing_central", "train_list_missing"),
])
def test_루트의_목록이_모자라면_제안하지_않는다(tmp_path, case, code):
    plan, _ = _plan(tmp_path, "rehearsal_uni.yaml")
    root = tmp_path / "reh"
    if case == "no_lists":
        root.mkdir()
    elif case != "no_root":
        rows = dict(FULL)
        if case == "missing_C2":
            rows.pop("uni_local_C2")
        elif case == "missing_central":
            rows.pop("uni_central")
        _lists(root, rows)
        if case == "empty_C3":
            (root / "lists" / "train_uni_local_C3.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(RT.ProposalRefused) as exc:
        RT.propose(plan, load_s=600, root=root)
    assert exc.value.code == code


def test_루트의_목록이_다_있으면_실제_행으로_제안한다(tmp_path, capsys):
    plan, path = _plan(tmp_path, "rehearsal_uni.yaml")
    root = _lists(tmp_path / "reh", FULL)
    out = RT.propose(plan, load_s=600, root=root)
    assert out["inputs"]["rows"] == FULL and out["inputs"]["rows_from"] == "root"
    # 연합 2 × (3 × 600 + (101 + 61 + 33) × 1 × 3.621) = 2 × 2,506.095 = 5,012.19 → 1.5 × 5,012.19 + 60 = 7,578.3 → 7,600
    assert out["inputs"]["train_per_cell_s"]["uni_fed"] == 5012.2 and out["timeouts_s"]["train"] == 7600
    assert RT.main(["propose", "--plan", str(path), "--load-s", "600", "--root", str(tmp_path / "nowhere")]) == RT.EXIT_REFUSED
    assert "[root_missing]" in capsys.readouterr().err

