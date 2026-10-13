"""리허설의 연합 칸 — 실행 설정 키 · 클라이언트의 학습 행 · 서버의 초기 어댑터 자리 · 프로세스 찾기 · run-config 문자열(리허설 2판 §1-3 의 연합 행 · §13-4).

모델 없이 CPU 로 돈다. 연합 한 바퀴(학습 → export → 평가 쪽 채점)의 끝에서 끝은 `tests/test_rehearsal_orch.py` 의 `lap` 이 본다 —
전송 계층(`flwr run`)만 서버 루프의 부품을 직접 잇는 대역으로 바꾸고, 실제 `flwr run` 경로는 GPU 창이 본다(2판 §6-2 의 5).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl.uni_fed import UniFedRejected, UniFedRun
from fl.uni_run_config import (
    UNI_RUN_DEFAULTS,
    UNI_SERVER_DEFAULTS,
    client_run_cfg,
    down_config,
)
from tests.test_uni_train_cell import reh
from tests.uni_fakes import make_rows, write_pairs

TAGS = ["C1", "C2", "C3"]


# ================================================================ 실행 설정 키
def _cfg(**over) -> dict:
    return {**UNI_RUN_DEFAULTS, **over}


def test_본실험은_학습_행_목록을_받지_않는다():
    main = _cfg(**{k: "x" for k in ("uni-model", "uni-model-revision", "uni-pairs", "uni-pairs-digest", "uni-prompt",
                                   "uni-coord-space")},
                **{"uni-chat-template-kwargs": "{}", "uni-seed-index": "1", "uni-processor-kwargs": "{}",
                   "uni-expected-supervised-tokens": '{"C1": 1, "C2": 1, "C3": 1}', "purpose": "main"})
    assert client_run_cfg(main)["train_lists"] is None
    with pytest.raises(ValueError, match="uni-train-lists"):
        client_run_cfg({**main, "uni-train-lists": "/x/lists"})
    reh_cfg = _cfg(purpose="rehearsal", plan="/x/plan.yaml", **{"uni-train-lists": "/x/lists"})
    assert client_run_cfg(reh_cfg)["train_lists"] == "/x/lists"


def test_새_키는_선언되고_서버_전용_키는_내려보내지_않는다():
    import tomllib

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    assert declared["uni-train-lists"] == UNI_RUN_DEFAULTS["uni-train-lists"] == ""
    assert declared["uni-init-adapter"] == UNI_SERVER_DEFAULTS["uni-init-adapter"] == ""
    assert "uni-init-adapter" not in down_config(lambda k, d: d)            # 클라이언트는 초기 어댑터 자리를 모른다


# ================================================================ 클라이언트의 학습 행
def _lists(tmp: Path):
    rows = {c: make_rows(tmp / "img", c, n, start=10 * i) for i, (c, n) in enumerate(zip(TAGS, (3, 2, 2)))}
    pairs = write_pairs(tmp / "pairs.jsonl", [r for c in TAGS for r in rows[c]])
    lists = tmp / "lists"
    lists.mkdir()
    files = {}
    for c in TAGS:
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[c][:-1]).encode("utf-8")   # 목록은 참여자 행의 일부
        (lists / f"train_uni_local_{c}.jsonl").write_bytes(raw)
        files[f"train_uni_local_{c}.jsonl"] = hashlib.sha256(raw).hexdigest()
    (lists / "lists_record.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return SimpleNamespace(rows=rows, pairs=pairs, lists=lists, files=files)


def test_연합_클라이언트는_목록_단계의_그_참여자_행만_받는다(tmp_path):
    from vlm.rehearsal_run import fed_client_rows

    w = _lists(tmp_path)
    got = fed_client_rows(w.lists, client="C1", pairs_path=w.pairs, run_root=tmp_path)
    assert got == w.rows["C1"][:-1]                                          # 참여자 행 전체가 아니라 목록의 행


def test_연합_클라이언트는_실행_루트_밖의_목록을_읽지_않는다(tmp_path):
    """서버가 내려보낸 실행 루트 아래가 아닌 목록 폴더 — 바이트 · 행이 다 맞아도 읽지 않는다(외부 검토 회신 `rehearsal_blockers` 의 3)."""
    from vlm.rehearsal_run import fed_client_rows

    w = _lists(tmp_path)
    other = tmp_path / "other_root"
    other.mkdir()
    for root in (other, None):
        with pytest.raises(ValueError, match="실행 루트"):
            fed_client_rows(w.lists, client="C1", pairs_path=w.pairs, run_root=root)


@pytest.mark.parametrize("case", ["bytes", "row", "other_client", "empty"])
def test_연합_클라이언트는_어긋난_목록으로_학습하지_않는다(tmp_path, case):
    from vlm.rehearsal_run import fed_client_rows

    w = _lists(tmp_path)
    p = w.lists / "train_uni_local_C1.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    if case == "bytes":
        p.write_bytes(p.read_bytes() + b"\n")
    else:
        if case == "row":
            rows[0] = {**rows[0], "image_path": "elsewhere.png"}
        elif case == "other_client":
            rows[0] = w.rows["C2"][0]
        elif case == "empty":
            rows = []
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
        p.write_bytes(raw)
        rec = json.loads((w.lists / "lists_record.json").read_text(encoding="utf-8"))
        rec["files"][p.name] = hashlib.sha256(raw).hexdigest()               # 기록도 맞춰 바이트 검사 뒤의 검사를 본다
        (w.lists / "lists_record.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError):
        fed_client_rows(w.lists, client="C1", pairs_path=w.pairs, run_root=tmp_path)


def _link(case: str, src: Path, dst: Path) -> None:
    if case == "symlink":
        try:
            os.symlink(src, dst)
        except OSError as exc:                                   # 파일 기호 링크는 권한이 있어야 만든다(개발자 모드 · 관리자)
            pytest.skip(f"이 환경은 파일 기호 링크를 만들 수 없다: {exc}")
    else:
        os.link(src, dst)


@pytest.mark.parametrize("which", ["list", "record"])
@pytest.mark.parametrize("case", ["symlink", "hardlink"])
def test_연합_클라이언트는_다른_루트의_파일로_가는_링크를_읽지_않는다(tmp_path, case, which):
    """루트 안의 목록 폴더에 다른 실행 루트의 정상 파일로 가는 링크를 둔다 — 폴더 검사 · 바이트 · 행 대조를 다 지나는 꼴이다(외부 검토 회신
    `rehearsal_blockers2` 의 3). 링크는 이 시험의 임시 폴더 안에서만 만든다."""
    from vlm.rehearsal_run import fed_client_rows

    other = tmp_path / "other_root"
    other.mkdir()
    w = _lists(other)                                            # 다른 루트의 정상 목록
    mine = tmp_path / "my_root"
    (mine / "lists").mkdir(parents=True)
    names = ["train_uni_local_C1.jsonl", "lists_record.json"]
    for n in names:
        if (n == names[0]) == (which == "list"):
            _link(case, w.lists / n, mine / "lists" / n)
        else:
            (mine / "lists" / n).write_bytes((w.lists / n).read_bytes())
    with pytest.raises(ValueError, match="읽지 않는다"):
        fed_client_rows(mine / "lists", client="C1", pairs_path=w.pairs, run_root=mine)


def test_읽는_파일의_최종_경로가_루트_밖이면_읽지_않는다(tmp_path):
    """파일 하나의 최종 경로 검사 — 루트 안의 정션 폴더를 지나 밖의 파일에 닿는 경로(파일 기호 링크를 만들 수 없는 환경에서도 같은 갈래를 본다)."""
    import _winapi

    from vlm.rehearsal_run import _read_under

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "in.json").write_text("{}", encoding="utf-8")
    assert _read_under(root / "in.json", root) == b"{}"
    junction = root / "j"
    _winapi.CreateJunction(str(outside.resolve()), str(junction))
    try:
        with pytest.raises(ValueError, match="실행 루트"):
            _read_under(junction / "f.json", root)
    finally:
        os.rmdir(junction)
    assert (outside / "f.json").is_file()


def test_연합_클라이언트는_다른_루트로_가는_정션_폴더를_읽지_않는다(tmp_path):
    import _winapi

    from vlm.rehearsal_run import fed_client_rows

    other = tmp_path / "other_root"
    other.mkdir()
    w = _lists(other)
    mine = tmp_path / "my_root"
    mine.mkdir()
    junction = mine / "lists"
    _winapi.CreateJunction(str(w.lists.resolve()), str(junction))
    try:
        with pytest.raises(ValueError, match="실행 루트"):
            fed_client_rows(junction, client="C1", pairs_path=w.pairs, run_root=mine)
    finally:
        os.rmdir(junction)                                       # 링크만 뗀다 — 가리키던 폴더는 남는다
    assert (w.lists / "train_uni_local_C1.jsonl").is_file()


# ================================================================ 서버의 초기 어댑터 자리
def _server_get(tmp: Path, **over):
    vals = {"uni-model": "tiny", "uni-model-revision": "r1", "uni-pairs": str(tmp / "pairs.jsonl"),
            "uni-pairs-digest": "p" * 64, "uni-chat-template-kwargs": '{"enable_thinking": false}',
            "uni-coord-space": "ABS_ORIG", "purpose": "rehearsal", "plan": str(reh(tmp) / "plan.yaml"),
            "uni-seed-index": "1", "rehearsal-root": str(reh(tmp))}
    vals.update(over)
    return lambda k, d: vals.get(k, d)


def _uni_fed(tmp: Path, get):
    return UniFedRun(get=get, out_dir=reh(tmp) / "train" / "uni_fed_s1", run_id="x", base_seed=7, split_hash="s" * 64,
                     num_rounds=2, local_epochs=1, total_epochs=2, client_tags=TAGS, non_main_parent=tmp)


@pytest.fixture
def root(tmp_path):
    reh(tmp_path).mkdir(parents=True)
    (reh(tmp_path) / "plan.yaml").write_text("kind: rehearsal\n", encoding="utf-8")
    write_pairs(tmp_path / "pairs.jsonl", [])
    return tmp_path


def test_리허설_서버는_루트_아래의_초기_어댑터만_받는다(root):
    inside = reh(root) / "init" / "initial.npz"
    _uni_fed(root, _server_get(root, **{"uni-init-adapter": str(inside)}))            # 루트 아래 — 받는다
    with pytest.raises(UniFedRejected) as exc:
        _uni_fed(root, _server_get(root, **{"uni-init-adapter": str(root / "elsewhere" / "initial.npz")}))
    assert exc.value.code == "run_root"


def test_리허설_서버는_루트_아래의_학습_행_목록만_받는다(root):
    """`uni-train-lists` 도 실행 루트 검사에 든다(2판 §1-4 입력 경로 격리, 외부 검토 회신 `rehearsal_blockers` 의 3)."""
    _uni_fed(root, _server_get(root, **{"uni-train-lists": str(reh(root) / "lists")}))   # 루트 아래 — 받는다
    with pytest.raises(UniFedRejected) as exc:
        _uni_fed(root, _server_get(root, **{"uni-train-lists": str(root / "elsewhere" / "lists")}))
    assert exc.value.code == "run_root"


def test_본실험_서버는_초기_어댑터_파일을_받지_않는다(root):
    main = {"purpose": "main", "plan": "", "rehearsal-root": "", "uni-prompt": str(ROOT / "vlm/prompts/unified_v2_absorig.txt"),
            "uni-processor-kwargs": "{}", "uni-expected-supervised-tokens": '{"C1": 1, "C2": 1, "C3": 1}',
            "uni-init-adapter": str(root / "initial.npz")}
    with pytest.raises(UniFedRejected) as exc:
        _uni_fed(root, _server_get(root, **main))
    assert exc.value.code == "main_with_init_adapter"


def test_서버는_초기_어댑터_자리를_그_키에서_읽는다(tmp_path, monkeypatch):
    import vlm.init_adapter as IA
    from fl import server_app

    seen = {}

    def fake_build(**kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(IA, "build_initial_adapter", fake_build)
    with pytest.raises(RuntimeError, match="stop"):
        server_app._load_initial("uni_fed", {"project": str(tmp_path / "proj"), "base-seed": 7, "purpose": "rehearsal",
                                             "uni-init-adapter": str(tmp_path / "root" / "init" / "initial.npz")})
    assert Path(seen["cache_path"]) == tmp_path / "root" / "init" / "initial.npz"
    with pytest.raises(RuntimeError, match="stop"):
        server_app._load_initial("uni_fed", {"project": str(tmp_path / "proj"), "base-seed": 7, "purpose": "main"})
    assert Path(seen["cache_path"]) == (tmp_path / "proj").resolve() / "fl" / "uni_fed" / "initial.npz"   # 비면 종전 자리


# ================================================================ 계획 — 연합 칸에 중단을 넣지 않는다
def test_계획은_연합_칸의_중단을_받지_않는다(tmp_path):
    """연합 칸은 중단을 넣지 않는다(2판 §1-1 — 서버 쪽 재개가 없다). 저장소의 리허설 계획에 연합 칸의 중단 한 줄을 더하면 거부한다."""
    from vlm.rehearsal_plan import PlanRejected, load_plan

    text = (ROOT / "configs" / "rehearsal_uni.yaml").read_text(encoding="utf-8")
    anchor = "faults:\n  train:\n"
    assert text.count(anchor) == 1
    p = tmp_path / "plan.yaml"
    p.write_text(text.replace(anchor, anchor + "    uni_fed: 'train:after_ckpt:uni_fed:ep=0'\n"), encoding="utf-8")
    with pytest.raises(PlanRejected) as exc:
        load_plan(p, purpose="rehearsal")
    assert "faults.train.uni_fed" in str(exc.value)
    load_plan(ROOT / "configs" / "rehearsal_uni.yaml", purpose="rehearsal")        # 저장소의 계획은 받는다


# ================================================================ 프로세스 찾기 · run-config 문자열
def test_상주_SuperLink_와_시뮬레이션만_찾고_자기는_뺀다():
    """오케스트레이터가 부르는 이름이 정확한 신원의 선택기다(`fl/flwr_procs.py`) — 이 venv 의 진입점만 대상, 다른 자리면 불명."""
    from vlm.rehearsal_orch import flwr_processes

    fake = [{"pid": 11, "name": "flower-superlink.exe", "exe": "C:/v/Scripts/flower-superlink.exe",
             "cmdline": ["flower-superlink", "--insecure"]},
            {"pid": 12, "name": "python.exe", "exe": "C:/py/python.exe",
             "cmdline": ["C:/v/python.exe", "C:/v/Scripts/flwr-simulation.exe", "--app"]},
            {"pid": 13, "name": "python.exe", "exe": "C:/py/python.exe", "cmdline": ["python", "-m", "pytest"]},
            {"pid": 14, "name": "notepad.exe", "exe": "C:/Windows/notepad.exe", "cmdline": ["notepad", "superlink.txt"]},
            {"pid": 15, "name": "flower-superlink.exe", "exe": "D:/other/Scripts/flower-superlink.exe", "cmdline": []},
            {"pid": os.getpid(), "name": "flower-superlink.exe", "exe": "C:/v/Scripts/flower-superlink.exe", "cmdline": []}]
    got = flwr_processes(scripts_dir=Path("C:/v/Scripts"), iter_procs=lambda: fake)
    assert [(p["pid"], p["kind"]) for p in got] == [(11, "target"), (12, "target"), (15, "unclear")]


def test_연합_실행_설정은_flwr_의_파서를_지나고_선언된_키만_쓴다():
    import tomllib

    from flwr.cli.config_utils import parse_config_args

    from vlm.rehearsal_run import fed_run_config_text

    items = {"cell": "uni_fed", "num-server-rounds": 2, "local-epochs": 1, "total-epochs": 2, "base-seed": 20260828,
             "uni-chat-template-kwargs": '{"enable_thinking":false}', "purpose": "rehearsal",
             "uni-train-lists": "C:/r/reh-20261001T120000/lists", "uni-init-adapter": "C:/r/reh-20261001T120000/init/initial.npz",
             "uni-model-revision": ""}
    parsed = parse_config_args([fed_run_config_text(items)])
    assert parsed == items
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    assert set(items) <= set(declared)
