"""착수 허가 면제(`--ecc-permit-waiver`, 지시 1017a) — 사용자 결정으로 ECC 모델 점검 허가를 기다리지 않고 본 칸을 연다.

시작 문(`vlm.train_cell.main_start_gate`)이 통과 파일 대신 면제 파일을 맞대는지(`vlm.ecc.check_waiver`), 허가와 면제를 함께 주면
거부하는지, 처음 받은 면제의 바이트가 바뀌면 거부하는지, 기록(칸 meta 의 `start_gate` · `gate-check` 진행 기록)에
`ecc_permit_sha256` 대신 `ecc_permit_waiver_sha256` 이 실리는지 본다. 세계는 `tests/test_ecc_train.py` 의 시작 문 세계를 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import main_uni as M
from tests import test_ecc_train as ET
from vlm import ecc as E
from vlm.train_cell import CellRejected

ADMISSION = hashlib.sha256(b"ECC_MODEL_FREE_ADMISSION").hexdigest()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _waiver(tmp: Path, name: str = "waiver.json", **over) -> Path:
    body = {"decision": "모델 점검 허가를 기다리지 않고 다섯 칸을 연다", "decision_log_section": "2026-10-17 착수 허가 면제",
            "model_free_admission_sha256": ADMISSION, "step100_geometry_sha256": None,
            "decided_at": "2026-10-17T06:05:00+09:00", **over}
    p = tmp / name
    p.write_text(json.dumps({k: v for k, v in body.items() if v != "__drop__"}, ensure_ascii=False), encoding="utf-8")
    return p


def _code(fn) -> str:
    with pytest.raises(CellRejected) as exc:
        fn()
    return exc.value.code


def test_면제_파일이면_통과_파일_없이_시작_문을_지나고_기록에_면제_sha_를_싣는다(tmp_path):
    g = ET._gate_world(tmp_path, accept=False)
    w = _waiver(tmp_path)
    spec = replace(g.main, ecc_permit_waiver_path=w)
    gate = ET._gate(g, spec)
    assert gate["ecc_permit_waiver_sha256"] == _sha(w) and "ecc_permit_sha256" not in gate
    assert gate["pin_sha256"] == g.pin_sha and gate["ecc_permit_waiver"]["model_free_admission_sha256"] == ADMISSION
    key = json.loads((g.seed_dir / E.WAIVER_KEY_FILE).read_text(encoding="utf-8"))
    assert key == {"waiver_sha256": _sha(w), "seed": 1, "pin_sha256": g.pin_sha, "start_adapter_digest": g.digest}
    assert ET._gate(g, spec) == gate                                    # 재개 — 같은 기록에 같은 값
    assert _code(lambda: ET._gate(g)) == "ecc_permit_missing"            # 면제를 주지 않으면 지금처럼 거부


def test_허가와_면제를_함께_주면_거부한다(tmp_path):
    g = ET._gate_world(tmp_path, accept=True)                           # 통과 파일 사본 · 허가 키가 시드 폴더에 있다
    spec = replace(g.main, ecc_permit_waiver_path=_waiver(tmp_path))
    assert _code(lambda: ET._gate(g, spec)) == "ecc_permit_and_waiver"
    assert not (g.seed_dir / E.WAIVER_KEY_FILE).exists()


def test_처음_받은_면제의_바이트가_바뀌면_거부한다(tmp_path):
    g = ET._gate_world(tmp_path, accept=False)
    w = _waiver(tmp_path)
    ET._gate(g, replace(g.main, ecc_permit_waiver_path=w))
    other = _waiver(tmp_path, name="waiver2.json", step100_geometry_sha256="c" * 64)   # 꼴은 맞지만 다른 바이트
    assert _code(lambda: ET._gate(g, replace(g.main, ecc_permit_waiver_path=other))) == "ecc_waiver_changed"
    w.write_text(w.read_text(encoding="utf-8") + " ", encoding="utf-8")                 # 같은 이름의 파일을 고쳤다
    assert _code(lambda: ET._gate(g, replace(g.main, ecc_permit_waiver_path=w))) == "ecc_waiver_changed"


@pytest.mark.parametrize(("over", "code"), [
    ({"model_free_admission_sha256": "x"}, "ecc_waiver_form"),
    ({"step100_geometry_sha256": "y" * 64}, "ecc_waiver_form"),
    ({"decided_at": "어제"}, "ecc_waiver_form"),
    ({"decision": " "}, "ecc_waiver_form"),
    ({"decision_log_section": "__drop__"}, "ecc_waiver_form"),
    ({"extra": 1}, "ecc_waiver_form"),
])
def test_면제_파일의_꼴(tmp_path, over, code):
    g = ET._gate_world(tmp_path, accept=False)
    spec = replace(g.main, ecc_permit_waiver_path=_waiver(tmp_path, **over))
    assert _code(lambda: ET._gate(g, spec)) == code
    missing = replace(g.main, ecc_permit_waiver_path=tmp_path / "none.json")
    assert _code(lambda: ET._gate(g, missing)) == "ecc_waiver_missing"


def test_동시에_켜진_칸들이_겨뤄도_키_기록은_하나이고_모두_지난다(tmp_path):
    w = _waiver(tmp_path)
    seed_dir = tmp_path / "seed1"
    kw = {"seed_dir": seed_dir, "permit_path": seed_dir / "ecc_permit.json", "pin_sha256": "a" * 64, "seed": 1,
          "start_adapter_digest": "d" * 64, "recipe": "json_only"}
    with ThreadPoolExecutor(max_workers=4) as ex:
        got = list(ex.map(lambda _: E.check_waiver(w, **kw)["waiver_sha256"], range(8)))
    assert set(got) == {_sha(w)} and sorted(p.name for p in seed_dir.iterdir()) == [E.WAIVER_KEY_FILE]


def test_gate_check_는_면제면_진행_기록에_면제_sha_를_적는다(tmp_path, monkeypatch, capsys):
    init = tmp_path / "initial.npz"
    init.write_bytes(b"x")
    spec = SimpleNamespace(init_adapter_path=init)
    monkeypatch.setattr(M, "spec_for", lambda *a, **k: spec)
    monkeypatch.setattr(M, "_start_gate", lambda s: {"pin_sha256": "p" * 64, "ecc_permit_waiver_sha256": "w" * 64,
                                                     "ecc_permit_waiver": {}})
    M.stage_gate_check(None, 1, out=tmp_path, ecc_permit_waiver=tmp_path / "w.json")
    row = json.loads((tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["ecc_permit_waiver_sha256"] == "w" * 64 and "ecc_permit_sha256" not in row
    assert json.loads(capsys.readouterr().out.strip())["ecc_permit_waiver_sha256"] == "w" * 64


def test_명세에_면제_자리가_실리고_신원에는_들지_않는다(tmp_path):
    from vlm.train_cell import _run_identity

    g = ET._gate_world(tmp_path, accept=False)
    a = _run_identity(g.main, "uni_central", client_idx=0, rows_digest="r", n_rows=1, prompt_sha256="p", init_digest="i")
    b = _run_identity(replace(g.main, ecc_permit_waiver_path=tmp_path / "w.json"), "uni_central", client_idx=0,
                      rows_digest="r", n_rows=1, prompt_sha256="p", init_digest="i")
    assert a == b


def test_명령줄의_면제_인자는_칸_명령에서만_받는다(capsys):
    assert M.main(["pin", "--seed", "1", "--ecc-permit-waiver", "w.json"]) == 2
    assert "[waiver_args]" in capsys.readouterr().err
