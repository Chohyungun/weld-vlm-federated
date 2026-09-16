"""본실험 산출물 보호 — 채점기·추론 부명령이 옛 판과 본실험 레코드를 덮지 않는다.

지키는 것(총괄 09-16 23:25 추기, C 42번 §9-6, D 49번 §7-4):

1. **n-1** 본실험 채점 루트(`…/outputs/main_d/…`)에서는 `score` 가 **v1 도** 대상이 있으면 멈춘다.
   경계는 경로 성분(`outputs`·`main_d` 연속, 대소문자 무시)이다. 파일럿 루트의 v1 다시 쓰기는 그대로다.
2. **n-2** `cmd_score` 는 모집단을 적재하기 **전에** 대상 파일을 확인한다.
3. **§7-4** 체크포인트 표(파일럿 배치)는 파일럿 프로파일에서만 쓴다. 본실험 프로파일의 `predict` 는
   모집단 적재 전에 멈춘다 — 파일럿 경로를 조용히 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.detect_infer import CHECKPOINT_LAYOUT_PROFILES, checkpoint_paths
from scripts.probe import score_cells as sc

# ---- n-1 — 경계 판정 --------------------------------------------------------------------


@pytest.mark.parametrize("rel,main", [
    ("outputs/main_d", True),
    ("outputs/main_d/seed1", True),
    ("OUTPUTS/Main_D/seed2", True),                  # 윈도우 — 대소문자 무시
    ("other_tree/outputs/main_d/seed3", True),       # 다른 워크트리·임시 폴더 아래도 같은 규칙
    ("outputs/pilot_d", False),
    ("outputs/main_dx/seed1", False),
    ("outputs/main_c/seed1", False),
    ("main_d/seed1", False),                         # `outputs` 가 앞에 없다
    ("outputs/x/main_d", False),                     # 연달아 있지 않다
])
def test_n1_본실험_루트는_경로_성분으로_가른다(tmp_path, rel, main):
    assert sc.is_main_output(tmp_path / rel) is main
    assert sc.is_main_output(Path(rel)) is main      # 상대경로도 같은 답(작업 디렉터리 아래로 절대화)


def test_m2_resolve_로_정규화된_형태도_본다(tmp_path, monkeypatch):
    """abspath 로는 본실험이 아닌데(짧은 이름·정션 별칭) resolve 가 본실험 경로로 확정하면 본실험이다(A 54번 m-2)."""
    alias = tmp_path / "ALIAS~1" / "seed1"
    real = tmp_path / "outputs" / "main_d" / "seed1"
    orig = Path.resolve

    def fake(self, strict=False):
        return real if "ALIAS~1" in str(self) else orig(self, strict=strict)

    assert sc.is_main_output(alias) is False
    monkeypatch.setattr(Path, "resolve", fake)
    assert sc.is_main_output(alias) is True
    assert sc.is_main_output(tmp_path / "outputs" / "pilot_d") is False

    def broken(self, strict=False):
        raise OSError("resolve 실패")

    monkeypatch.setattr(Path, "resolve", broken)          # 실패하면 abspath 형태만 본다
    assert sc.is_main_output(real) is True
    assert sc.is_main_output(alias) is False


def test_n1_본실험_루트에서는_v1_도_덮지_않는다(tmp_path):
    out = tmp_path / "outputs" / "main_d" / "seed1"
    out.mkdir(parents=True)
    (out / "score_cells_v1.json").write_bytes(b"v1 historical\n")
    with pytest.raises(SystemExit, match="본실험 루트"):
        sc.artifact_dest(out, "v1")
    assert (out / "score_cells_v1.json").read_bytes() == b"v1 historical\n"
    assert sc.artifact_is_protected(out, "v1") is True
    assert sc.artifact_dest(out, "v3") == out / "score_cells_v3.json"          # 새 판은 된다


def test_n1_파일럿_루트의_v1_은_그대로_다시_쓴다(tmp_path):
    out = tmp_path / "outputs" / "pilot_d"
    out.mkdir(parents=True)
    (out / "score_cells_v1.json").write_bytes(b"pilot\n")
    assert sc.artifact_dest(out, "v1") == out / "score_cells_v1.json"
    assert sc.artifact_is_protected(out, "v1") is False
    (out / "score_cells_v2.json").write_bytes(b"pilot v2\n")
    with pytest.raises(SystemExit, match="새 판 경로"):                          # v2 부터는 어디서든
        sc.artifact_dest(out, "v2")


# ---- n-2 · §7-4 — 적재 전에 멈추는가 ----------------------------------------------------

def _stub(monkeypatch, out: Path, profile: str = "main") -> list[str]:
    """지문·파라미터·모집단 적재를 가짜로 — 적재가 불리면 기록하고 실패한다."""
    calls: list[str] = []
    monkeypatch.setattr(sc, "scorer_code_digest", lambda *a, **k: {"combined": "stub"})
    monkeypatch.setattr(sc, "params_from_args",
                        lambda args: SimpleNamespace(out=Path(out), pilot=Path(out), profile=profile))

    def _no_load(*a, **k):
        calls.append("load_population")
        raise AssertionError("모집단 적재까지 갔다")

    monkeypatch.setattr(sc, "load_population", _no_load)
    return calls


@pytest.mark.parametrize("sub,version", [
    ("outputs/main_d/seed1", "v1"),          # n-1 경로
    ("outputs/pilot_d", "v2"),
    ("outputs/main_d/seed2", "v3"),
])
def test_n2_cmd_score_는_적재_전에_대상_파일을_확인한다(tmp_path, monkeypatch, sub, version):
    out = tmp_path / sub
    out.mkdir(parents=True)
    target = out / sc.ARTIFACT_VERSIONS[version]
    target.write_bytes(b"old\n")
    calls = _stub(monkeypatch, out)
    with pytest.raises(SystemExit, match="덮지 않는다"):
        sc.cmd_score(SimpleNamespace(artifact_version=version))
    assert calls == []
    assert target.read_bytes() == b"old\n"


def test_n2_파일럿_루트_v1_은_적재까지_간다(tmp_path, monkeypatch):
    """막지 않는 경로가 실제로 막히지 않는지 — 적재 단계에 닿으면 가짜 적재가 실패한다."""
    out = tmp_path / "outputs" / "pilot_d"
    out.mkdir(parents=True)
    (out / "score_cells_v1.json").write_bytes(b"old\n")
    calls = _stub(monkeypatch, out, profile="pilot")
    with pytest.raises(AssertionError, match="적재까지"):
        sc.cmd_score(SimpleNamespace(artifact_version="v1"))
    assert calls == ["load_population"]


def test_74_체크포인트_표는_파일럿_프로파일에서만_쓰고_본실험_predict_는_적재_전에_멈춘다(
        tmp_path, monkeypatch):
    assert CHECKPOINT_LAYOUT_PROFILES == frozenset({"pilot"})
    pilot = checkpoint_paths(tmp_path, profile="pilot")
    assert pilot[("sep_fed", None)] == tmp_path / "sep_fed" / "global_r003.npz"
    for profile in ("main", "probe_nondet"):
        with pytest.raises(ValueError, match="체크포인트 표"):
            checkpoint_paths(tmp_path, profile=profile)

    calls = _stub(monkeypatch, tmp_path / "outputs" / "main_d" / "seed1", profile="main")
    with pytest.raises(SystemExit, match="adapt_main_detections"):
        sc.cmd_predict(SimpleNamespace(at_conf=False, root="."))
    assert calls == []
