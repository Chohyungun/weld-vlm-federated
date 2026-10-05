"""통합형 연합 칸의 출력 자리 — 학습 루트 아래 `<학습 루트>/uni_fed_s<n>/`, 로컬 · 중앙과 같은 꼴(리허설 2판 §13-4 · 총괄 결정 04 의 3절).

평가 쪽 본채점은 `--train-root` 하나에서 `<tag>_s<n>/adapter_last.meta.json` 을 읽는다. 서버가 그 자리에 쓰면 평가 쪽을 바꾸지 않아도 연합 칸이 읽힌다.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl import server_app
from fl.uni_fed import UniFedRejected
from fl.uni_run_config import UNI_RUN_DEFAULTS, UNI_SERVER_DEFAULTS, down_config


def test_학습_루트가_있으면_연합_칸은_그_아래_uni_fed_s_n_에_쓴다(tmp_path):
    cfg = {"project": str(tmp_path / "p"), "uni-train-root": str(tmp_path / "train"), "uni-seed-index": "2",
           "purpose": "rehearsal"}
    assert server_app.cell_out_dir("uni_fed", cfg) == tmp_path / "train" / "uni_fed_s2"
    assert server_app.cell_out_dir("sep_fed", cfg) == tmp_path / "p" / "fl" / "sep_fed"      # 검출 칸은 그대로
    assert server_app.cell_out_dir("uni_fed", {**cfg, "uni-train-root": ""}) == tmp_path / "p" / "fl" / "uni_fed"


@pytest.mark.parametrize(("cfg", "code"), [
    ({"purpose": "main", "uni-train-root": "", "uni-seed-index": "1"}, "main_train_root_missing"),
    ({"purpose": "main", "uni-seed-index": "1"}, "main_train_root_missing"),
    ({"purpose": "rehearsal", "uni-train-root": "x", "uni-seed-index": ""}, "train_root_needs_seed_index"),
    ({"purpose": "rehearsal", "uni-train-root": "x", "uni-seed-index": "0"}, "train_root_needs_seed_index"),
])
def test_본실험은_학습_루트_없이_돌지_않고_시드_번째_없이는_자리를_내지_않는다(tmp_path, cfg, code):
    with pytest.raises(UniFedRejected) as exc:
        server_app.cell_out_dir("uni_fed", {"project": str(tmp_path), **cfg})
    assert exc.value.code == code and isinstance(exc.value, ValueError)


def test_키는_선언되고_서버만_읽는다():
    decl = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    for k, d in UNI_SERVER_DEFAULTS.items():
        assert decl.get(k) == d, f"pyproject 가 {k} 를 선언하지 않았다 — flwr run 이 덮어쓰기를 거부한다"
        assert k not in UNI_RUN_DEFAULTS
    assert "uni-train-root" not in down_config(lambda k, d: d)                  # 클라이언트에 내려보내지 않는다


def test_서버는_학습_루트_아래의_원장으로_신선도를_본다(tmp_path, monkeypatch):
    """새 자리에 다른 실행의 원장 행이 있으면 초기 어댑터를 읽기 전에 멈춘다 — 서버가 그 자리를 본다는 증거다."""
    from fl.atomic_log import AtomicLog, new_run_id
    from vlm import run_root

    root = tmp_path / "reh-20261001T000000"
    root.mkdir()
    (root / "plan.yaml").write_text("kind: rehearsal\n", encoding="utf-8")
    monkeypatch.setattr(run_root, "NON_MAIN_PARENT", tmp_path)
    out = root / "train" / "uni_fed_s1"
    AtomicLog(out / "atomic_log.csv", run_id=new_run_id("uni_fed", 7, "t1"), seed=7, cell="uni_fed",
              split_hash="s" * 64).log_round(round_idx=0, client_id="server", n_train_samples=0,
                                             metrics={"global_l2": 1.0})
    calls = []
    monkeypatch.setattr(server_app, "_load_initial", lambda *a, **k: calls.append(1))
    cfg = {"cell": "uni_fed", "num-server-rounds": 2, "local-epochs": 1, "total-epochs": 2, "project": str(root),
           "base-seed": 7, "run-stamp": "t1", "split-hash": "s" * 64, "purpose": "rehearsal",
           "plan": str(root / "plan.yaml"), "rehearsal-root": str(root), "uni-train-root": str(root / "train"),
           "uni-model": "tiny", "uni-model-revision": "r1", "uni-pairs": str(tmp_path / "pairs.jsonl"),
           "uni-pairs-digest": "p" * 64, "uni-chat-template-kwargs": '{"enable_thinking": false}',
           "uni-coord-space": "ABS_ORIG", "uni-seed-index": "1"}
    with pytest.raises(UniFedRejected) as exc:
        server_app.main(None, SimpleNamespace(run_config=cfg))
    assert exc.value.code == "fed_ledger_not_fresh" and calls == []
    assert not (root / "fl" / "uni_fed" / "atomic_log.csv").exists()


def test_본실험_실행기는_학습_루트를_넘기고_같은_자리를_본다(tmp_path):
    from scripts import main_uni as M
    from tests.test_main_uni import _cfg_yaml

    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    out = tmp_path / "out"
    rc = M.fed_run_config(cfg, 1, out=out)
    want = Path(str(M.seed_dir(out, 1).resolve())).as_posix()
    assert f'uni-train-root="{want}"' in rc
    assert M.fed_dir(out, 1) == M.seed_dir(out, 1) / "uni_fed_s1"
    fed = M.fed_dir(out, 1)
    fed.mkdir(parents=True)
    (fed / "adapter_last.npz").write_bytes(b"x")
    assert M.status_grid(out, 1)[1]["uni_fed"] == "open"                         # 상태 격자도 새 자리를 본다
