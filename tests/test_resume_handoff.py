"""34번 §3-1 8-2 — 최종 산출물 인계 뒤에만 재개 파일을 지운다.

종전: `train_round` 가 반환 **전에** `clear_resume` 했고 최종 NPZ 는 호출자가 그 **뒤** 저장했다.
그 사이(수 초)에 죽으면 재개 파일도 최종 산출물도 없다 — N=100 epoch 을 처음부터다.
지금: `clear_resume_on_success` 기본 False. 호출자가 "내구 저장(tmp→replace) → 재로드 검증 →
정리" 순서를 지킨다. ②③ 은 `train_cell`, ④ 는 클라이언트 응답 직전.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from detection import train_cell
from detection.round_runner import RoundResult, derive_seed


def _result(**kw) -> RoundResult:
    base = dict(ndarrays=[np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(3, dtype=np.float32)],
                num_examples=1, round_idx=0, client_idx=0, epochs_ran=2, seed=derive_seed(7, 0, 0),
                optimizer_steps=1, param_l2_norm=0.0, payload_bytes=36)
    base.update(kw)
    return RoundResult(**base)


def _seed_resume(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "resume_ep0001.pt"
    p.write_bytes(b"not-a-real-checkpoint")
    return p


def _resume_files(d: Path) -> list[str]:
    return sorted(p.name for p in d.glob("resume_ep*")) if d.is_dir() else []


# ==========================================================================
# save_cell_weights — 내구 저장·재로드 검증
# ==========================================================================

def test_저장은_tmp를_거쳐_최종이_되고_meta에_sha256과_배열수를_남긴다(tmp_path):
    r = _result()
    path = train_cell.save_cell_weights(tmp_path, "sep_central", r, identity={"run_id": "x"})
    assert path == tmp_path / "sep_central.npz"
    assert not list(tmp_path.glob("*.tmp"))
    meta_path = tmp_path / "sep_central.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wf = meta["weights_file"]
    assert wf["name"] == "sep_central.npz" and wf["n_arrays"] == 2
    # 해시·크기는 tmp 에서 재지만 `os.replace` 는 바이트를 바꾸지 않는다 — 최종 파일과 같아야 한다.
    assert wf["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert wf["file_bytes"] == path.stat().st_size
    assert meta["identity"] == {"run_id": "x"}
    # meta 를 바이너리로 쓰므로 개행은 LF 다(종전 write_text 는 Windows 에서 CRLF). D 통지 항목.
    assert b"\r\n" not in meta_path.read_bytes()
    with np.load(path) as z:
        assert [z[k].tolist() for k in z.files] == [a.tolist() for a in r.ndarrays]


def test_완료_마커는_meta_뒤에_확정된다(monkeypatch, tmp_path):
    """검토 I3. `<tag>.npz` 는 체인의 칸 건너뜀 마커다. meta 보다 먼저 확정되면 "마커만 있고
    meta 결손" 창이 열리고, 다음 실행은 마커만 보고 그 칸을 영구히 건너뛴다(감사 3 의 고장)."""
    seen = []
    real_replace = train_cell.os.replace

    def spy(src, dst):
        if str(dst).endswith("sep_central.npz"):
            seen.append((tmp_path / "sep_central.meta.json").exists())
        return real_replace(src, dst)

    monkeypatch.setattr(train_cell.os, "replace", spy)
    train_cell.save_cell_weights(tmp_path, "sep_central", _result())
    assert seen == [True], "npz 교체 시점에 meta 가 이미 있어야 한다"


def test_배열이_여러_개면_순서_그대로_왕복한다(tmp_path):
    """실물 검출 모델은 배열이 수백 개다. `z.files` 를 사전식으로 정렬하면 arr_10 과 arr_2 가
    어긋나 엉뚱한 텐서끼리 비교된다(serialize.py 머리말이 규정한 최대 위험)."""
    arrays = [np.full((i + 1,), float(i), dtype=np.float32) for i in range(13)]
    train_cell.save_cell_weights(tmp_path, "many", _result(ndarrays=arrays))
    with np.load(tmp_path / "many.npz") as z:
        back = [z[name] for name in z.files]
    assert [a.tolist() for a in back] == [a.tolist() for a in arrays]
    meta = json.loads((tmp_path / "many.meta.json").read_text(encoding="utf-8"))
    assert meta["weights_file"]["n_arrays"] == 13


def test_재로드_검증에_실패하면_최종_npz_와_meta_가_생기지_않는다(monkeypatch, tmp_path):
    """최종 `.npz` 는 체인의 완료 마커다 — 깨진 파일이 마커가 되면 다음 실행이 칸을 건너뛴다."""
    monkeypatch.setattr(train_cell, "_reload_arrays", lambda p: [])
    with pytest.raises(RuntimeError, match="재로드"):
        train_cell.save_cell_weights(tmp_path, "sep_central", _result())
    assert not (tmp_path / "sep_central.npz").exists()
    assert not (tmp_path / "sep_central.meta.json").exists()
    assert (tmp_path / "sep_central.npz.tmp").exists()


def test_값이_다르게_읽히면_거부한다(monkeypatch, tmp_path):
    real = train_cell._reload_arrays

    def corrupt(p):
        arrs = real(p)
        arrs[0] = arrs[0] + 1
        return arrs

    monkeypatch.setattr(train_cell, "_reload_arrays", corrupt)
    with pytest.raises(RuntimeError, match="불일치"):
        train_cell.save_cell_weights(tmp_path, "sep_central", _result())
    assert not (tmp_path / "sep_central.npz").exists()


# ==========================================================================
# ②③ — 저장 예외면 재개 파일이 남고, 정상이면 저장·검증 뒤 정리된다
# ==========================================================================

def _drive(monkeypatch):
    calls = []

    def fake_train(**kw):
        calls.append(kw)
        return _result(client_idx=kw["client_idx"], seed=derive_seed(kw["base_seed"], 0, kw["client_idx"]),
                       loader_reseed_per_epoch=kw["loader_reseed_per_epoch"])

    monkeypatch.setattr(train_cell, "train_round", fake_train)
    return calls


def _run(kind: str, out: Path, resume_root: Path):
    common = dict(model="m", total_epochs=2, base_seed=7, out_dir=out, split_hash="h",
                  run_stamp="s", resume_root=resume_root)
    if kind == "local":
        return train_cell.run_local_cell(client_data_yamls={0: "a.yaml", 1: "b.yaml"},
                                         client_num_examples={0: 1, 1: 1}, **common)
    return train_cell.run_central_cell(data_yaml="a.yaml", num_examples=1, **common)


def _rdirs(kind: str, root: Path) -> list[Path]:
    return [root / "sep_local_c0", root / "sep_local_c1"] if kind == "local" else [root / "sep_central"]


def _tags(kind: str) -> list[str]:
    return ["sep_local_c0", "sep_local_c1"] if kind == "local" else ["sep_central"]


def test_정리가_실패하면_조용히_넘어가지_않는다(monkeypatch, tmp_path, capsys):
    """검토 I2. `clear_resume` 는 OSError 를 삼키고 개수만 돌려준다 — Windows 에서 다른 핸들이
    `.pt` 를 잡으면 조용히 no-op 이 된다. 잔존을 로그로 알려야 사람이 치운다."""
    _drive(monkeypatch)
    root = tmp_path / "_resume"
    d = _rdirs("central", root)[0]
    _seed_resume(d)
    monkeypatch.setattr(train_cell, "clear_resume", lambda p: 0)      # 삼켜진 실패를 흉내
    _run("central", tmp_path / "out", root)
    out = capsys.readouterr().out
    assert "[resume] 정리 실패" in out and "resume_ep0001.pt" in out
    assert _resume_files(d) == ["resume_ep0001.pt"]


@pytest.mark.parametrize("kind", ["local", "central"])
def test_저장이_던지면_재개_파일이_남는다(fake_trainer, monkeypatch, tmp_path, kind):
    """(a) **실제 `train_round`** 를 탄다 — 학습 함수를 가짜로 바꿔 버리면 종전 코드의 내부
    `clear_resume` 가 애초에 실행되지 않아 기본값 전환을 잡지 못한다(초안의 무이빨 지점)."""
    root = tmp_path / "_resume"
    for d in _rdirs(kind, root):
        _seed_resume(d)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(train_cell, "save_cell_weights", boom)
    with pytest.raises(OSError):
        _run(kind, tmp_path / "out", root)
    for d in _rdirs(kind, root):
        assert _resume_files(d) == ["resume_ep0001.pt"]


@pytest.mark.parametrize("kind", ["local", "central"])
def test_실제_경로도_저장_뒤_정리한다(fake_trainer, tmp_path, kind):
    """(b) 실 트레이너 스텁 + 실제 `train_round`. 산출물과 정리가 함께 성립한다."""
    root = tmp_path / "_resume"
    for d in _rdirs(kind, root):
        _seed_resume(d)
    out = tmp_path / "out"
    _run(kind, out, root)
    for d in _rdirs(kind, root):
        assert _resume_files(d) == []
    for tag in _tags(kind):
        assert (out / f"{tag}.npz").exists() and (out / f"{tag}.meta.json").exists()


@pytest.mark.parametrize("kind", ["local", "central"])
def test_최종_산출물이_있으면_잔해를_학습_전에_치운다(fake_trainer, tmp_path, kind):
    """검토 I1. 저장과 정리 사이에서 죽으면 **완주** 체크포인트가 남고, 그대로 재기동하면
    `validate_detection_resume` 가 "예산 완료" 로 거부해 사람이 손으로 지워야 했다.
    최종 npz 가 있으면 그 재개 파일은 잔해이므로 학습 전에 치워 하드스톱을 없앤다."""
    from detection.resume import ResumeCheckpointer, ResumeIdentity, latest_resume
    from detection.round_runner import _LRTrace, validate_detection_resume
    from tests.test_detection_resume import _Counter, _FakeTrainer as _CkTrainer, _Net as _CkNet

    root = tmp_path / "_resume"
    out = tmp_path / "out"
    _run(kind, out, root)                       # 1회차: 산출물 + 정리 완료
    for d in _rdirs(kind, root):
        assert _resume_files(d) == []

    # "저장은 됐고 정리 전에 죽은" 상태를 만든다 — 완주(2/2) 체크포인트를 되돌려 놓는다.
    idxs = (0, 1) if kind == "local" else (0,)
    yamls = {0: "a.yaml", 1: "b.yaml"}
    for d, cidx in zip(_rdirs(kind, root), idxs):
        data = yamls[cidx] if kind == "local" else "a.yaml"
        ident = ResumeIdentity(run_id="s", round_idx=0, client_idx=cidx,
                               seed=derive_seed(7, 0, cidx), total_epochs=2, local_epochs=2,
                               model="m", data=str(Path(data).resolve()), profile="main")
        tr = _CkTrainer(_CkNet(), epoch=1, start_epoch=0)
        tr.n_optimizer_updates = 2
        tr._resume_lr_trace = _LRTrace([(0, 0.01), (1, 0.009)])
        ResumeCheckpointer(d, identity=ident, step_counter=_Counter(4)).save(tr)
        state = latest_resume(d, identity=ident)
        assert state is not None
        with pytest.raises(ValueError, match="예산이 이미 완료"):   # 치우지 않으면 하드스톱
            validate_detection_resume(state)

    _run(kind, out, root)                       # 2회차: 거부 없이 돈다
    for d in _rdirs(kind, root):
        assert _resume_files(d) == []


@pytest.mark.parametrize("kind", ["local", "central"])
def test_정상_경로는_저장_검증_뒤_재개_파일을_지운다(monkeypatch, tmp_path, kind):
    """(b) 배선 관점 — `train_round` 에 무엇이 넘어가고 무엇이 남는가."""
    calls = _drive(monkeypatch)
    root = tmp_path / "_resume"
    for d in _rdirs(kind, root):
        _seed_resume(d)
    _run(kind, tmp_path / "out", root)
    # 양변을 정규화한다 — `train_cell` 은 `resolve()` 한 경로를 넘기고, 정션(outputs) 아래
    # basetemp 에서는 미해결 경로와의 동등 비교가 깨진다(검토 I4).
    assert all(kw["resume_dir"] in [d.resolve() for d in _rdirs(kind, root)] for kw in calls)
    assert "clear_resume_on_success" not in calls[0]          # train_round 기본값(False)에 맡긴다
    for d in _rdirs(kind, root):
        assert _resume_files(d) == []
    for tag in _tags(kind):
        meta = json.loads((tmp_path / "out" / f"{tag}.meta.json").read_text(encoding="utf-8"))
        assert meta["weights_file"]["n_arrays"] == 2


@pytest.mark.parametrize("kind", ["local", "central"])
def test_정리_순서는_저장_다음이다(monkeypatch, tmp_path, kind):
    """저장 시점에 재개 파일이 **아직 있어야** 한다 — 순서가 뒤집히면 (a) 가 무의미해진다."""
    _drive(monkeypatch)
    root = tmp_path / "_resume"
    for d in _rdirs(kind, root):
        _seed_resume(d)
    seen = []
    real = train_cell.save_cell_weights

    def spy(out_dir, tag, result, identity=None):
        seen.append([_resume_files(d) for d in _rdirs(kind, root)])
        return real(out_dir, tag, result, identity=identity)

    monkeypatch.setattr(train_cell, "save_cell_weights", spy)
    _run(kind, tmp_path / "out", root)
    assert seen and all(any(files) for files in seen)


def test_resume_root_없이도_돈다(monkeypatch, tmp_path):
    _drive(monkeypatch)
    train_cell.run_central_cell(data_yaml="a.yaml", num_examples=1, model="m", total_epochs=2,
                                base_seed=7, out_dir=tmp_path, split_hash="h", run_stamp="s")
    assert (tmp_path / "sep_central.npz").exists()


# ==========================================================================
# (c) train_round 단독 호출은 재개 파일을 건드리지 않는다 — 가짜 트레이너
# ==========================================================================

@pytest.fixture
def fake_trainer(monkeypatch):
    """`train_round` 가 실제 트레이너 대신 중립 가짜(tests/_round_fakes.py)를 타게 한다."""
    pytest.importorskip("ultralytics")
    import detection.fed_trainer as ft

    from tests._round_fakes import RoundTrainer

    monkeypatch.setattr(ft, "FedDetectionTrainer", RoundTrainer)


def _round(tmp_path, **kw):
    from detection.round_runner import train_round
    return train_round(data_yaml=str(tmp_path / "d.yaml"), model="m.pt", total_epochs=2, local_epochs=2,
                       project=tmp_path / "runs", profile="pilot", run_id="t", **kw)


def test_train_round_단독_호출은_재개_파일을_건드리지_않는다(fake_trainer, tmp_path):
    """(c) 기본값. 읽을 수 없는 재개 파일(→ latest_resume None)은 그 자리에 그대로 남는다."""
    rdir = tmp_path / "_resume"
    _seed_resume(rdir)
    res = _round(tmp_path, resume_dir=rdir)
    assert res.epochs_ran == 2
    assert _resume_files(rdir) == ["resume_ep0001.pt"]


def test_train_round_는_명시_opt_in_이면_종전처럼_지운다(fake_trainer, tmp_path):
    rdir = tmp_path / "_resume"
    _seed_resume(rdir)
    _round(tmp_path, resume_dir=rdir, clear_resume_on_success=True)
    assert _resume_files(rdir) == []


def test_기본값이_False_다():
    import inspect
    from detection.round_runner import train_round
    assert inspect.signature(train_round).parameters["clear_resume_on_success"].default is False


def test_resume_dir_을_주는_호출자는_모두_정리_주체를_갖는다():
    """검토 I6. 기본값이 False 이므로 `resume_dir` 을 주는 호출자는 스스로 치워야 한다.
    빠뜨리면 두 번째 실행이 "예산 완료 체크포인트" 로 죽는다(`gate_reduced_pilot` 이 그랬다)."""
    import detection

    root = Path(detection.__file__).resolve().parent.parent
    for rel in ("detection/train_cell.py", "fl/client_det.py", "scripts/gate_reduced_pilot.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "resume_dir" in src, rel
        assert "clear_resume" in src, f"{rel}: resume_dir 을 주면서 정리 주체가 없다"


def test_clear_resume_on_success_True_를_명시하는_호출자가_없다():
    """(d) 학습·연합 경로 어디서도 옛 동작을 되살리지 않는다."""
    import detection

    root = Path(detection.__file__).resolve().parent.parent
    hits = []
    for base in ("detection", "fl", "scripts"):
        for p in (root / base).rglob("*.py"):
            if "clear_resume_on_success=True" in p.read_text(encoding="utf-8"):
                hits.append(p.name)
    assert hits == []


# ==========================================================================
# ④ fl/client_det — 응답 직전 정리, 응답 조립이 던지면 남는다
# ==========================================================================

def _client_cfg(tmp_path):
    return {"data_yaml": "d.yaml", "model": "m", "total_epochs": 2, "local_epochs": 2, "base_seed": 7,
            "num_examples": 1, "project": str(tmp_path / "runs"), "resume_root": str(tmp_path / "_resume"),
            "run_id": "s"}


def test_클라이언트는_응답_직전에_재개_파일을_지운다(monkeypatch, tmp_path):
    from fl import client_det

    def fake_train(**kw):
        assert kw["resume_dir"] == (tmp_path / "_resume" / "r003_c1").resolve()
        return _result(client_idx=1, round_idx=3)

    monkeypatch.setattr(client_det, "train_round", fake_train)
    rdir = _seed_resume(tmp_path / "_resume" / "r003_c1").parent
    arrays, metrics, strings = client_det.run_client_round(
        weights_in=[], canonical_keys=["w"], round_idx=3, client_idx=1, cfg=_client_cfg(tmp_path))
    assert len(arrays) == 2 and metrics["epochs-ran"] == 2.0
    assert _resume_files(rdir) == []


def test_클라이언트_응답_조립이_던지면_재개_파일이_남는다(monkeypatch, tmp_path):
    from fl import client_det

    monkeypatch.setattr(client_det, "train_round",
                        lambda **kw: _result(client_idx=1, round_idx=3, stopper_calls=None))
    rdir = _seed_resume(tmp_path / "_resume" / "r003_c1").parent
    with pytest.raises(TypeError):
        client_det.run_client_round(weights_in=[], canonical_keys=["w"], round_idx=3, client_idx=1,
                                    cfg=_client_cfg(tmp_path))
    assert _resume_files(rdir) == ["resume_ep0001.pt"]


def test_클라이언트는_resume_root_없이도_돈다(monkeypatch, tmp_path):
    from fl import client_det

    monkeypatch.setattr(client_det, "train_round", lambda **kw: _result(client_idx=1, round_idx=3))
    cfg = _client_cfg(tmp_path)
    cfg["resume_root"] = None
    arrays, metrics, strings = client_det.run_client_round(
        weights_in=[], canonical_keys=["w"], round_idx=3, client_idx=1, cfg=cfg)
    assert strings["weight-unit"] == "num_examples"
