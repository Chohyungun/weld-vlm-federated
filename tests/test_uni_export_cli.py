"""export 진입점(`scripts/export_uni.py`)의 명령줄 — 인자 · 종료 코드 · 이음새가 명령줄로 열리지 않는 것 · 이미지 경로 규칙.

합성 세계(`tests/uni_export_world.py`) 위에서 돈다. 모델을 올리지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import export_uni as CLI
from tests import uni_export_world as W


@pytest.fixture
def w(tmp_path):
    return W.build(tmp_path)


def _argv(w, *, mode="model", purpose="rehearsal", folder=None, extra=()):
    a = ["--purpose", purpose, "--mode", mode, "--tag", "uni_local_C1", "--seed-index", "1",
         "--receipt", str(w.receipt), "--list", str(w.lists[mode]), "--snapshot", str(w.snap),
         "--folder", str(folder or w.root / "export"), "--device", W.DEVICE]
    if purpose != "main":
        a += ["--run-root", str(w.root), "--plan", str(w.plan)]
    if mode == "model":
        a += ["--adapter-dir", str(w.adapter)]
    return a + list(extra)


def _seams(w, **kw):
    return dict(measure_env=W.measure, image_path_of=w.images, repo=w.repo, code_repo=w.code,
                exposure_ledger=w.exposure, non_main_parent=w.parent, clock=W.Clock(), **kw)


def test_명령줄로_모델과_에코_묶음을_같은_폴더에_쓰고_평가_쪽_검증기가_받는다(w, capsys):
    folder = w.root / "export"
    assert CLI.main(_argv(w), load_generator=W.Loader(), **_seams(w)) == CLI.EXIT_OK
    assert CLI.main(_argv(w, mode="echo"), load_generator=W.Loader(), **_seams(w)) == CLI.EXIT_OK
    names = sorted(p.name for p in folder.iterdir())
    assert "uni_local_C1_s1.export_meta.json" in names and "uni_local_C1_s1.echo.export_meta.json" in names
    assert names.count("attempts.jsonl") == 1                                  # 폴더 공용 시도 기록
    assert W.verify(w).mode == "model" and W.verify(w, mode="echo", folder=folder).mode == "echo"
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("sealed attempt=1 reason=first_pass") and out[1].startswith("sealed attempt=1")


def test_계획된_멈춤은_명령줄_인자로_열고_다음_호출이_이어_간다(w, capsys):
    clock = W.Clock()
    s = _seams(w)
    s["clock"] = clock
    assert CLI.main(_argv(w, extra=["--stop-after-lines", "2"]), load_generator=W.Loader(), **s) == CLI.EXIT_OK
    assert CLI.main(_argv(w), load_generator=W.Loader(), **s) == CLI.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "stopped attempt=1 reason=first_pass n_written=2"
    assert out[1].startswith("sealed attempt=2 reason=resume_after_interruption n_written=2")


def test_모델_생성기가_없으면_모델_모드는_시작하지_않는다(w, capsys, monkeypatch):
    """실제 구현이 없는 판을 흉내 낸다 — 명령줄이 시작 전에 거부한다(이 판에는 실제 구현이 있다)."""
    import vlm.export_generator as G

    monkeypatch.setattr(G, "model_generator_available", lambda: False)
    assert CLI.main(_argv(w), **_seams(w)) == CLI.EXIT_REFUSED
    assert capsys.readouterr().err.startswith("[model_generator_missing]")
    assert not (w.root / "export").exists()


def test_거부는_사유_코드를_첫_줄에_쓰고_종료_2_다(w, capsys):
    w.exposure.write_bytes(b"")                                            # planned 줄이 없다
    assert CLI.main(_argv(w), load_generator=W.Loader(), **_seams(w)) == CLI.EXIT_REFUSED
    assert capsys.readouterr().err.startswith("[exposure_not_planned]")
    assert CLI.main(_argv(w, extra=["--stop-after-lines", "0"]), load_generator=W.Loader(), **_seams(w)) == \
        CLI.EXIT_REFUSED
    assert capsys.readouterr().err.startswith("[stop_after_lines]")


def test_본실험은_대역_실측을_명령줄_밖에서도_받지_않는다(w, capsys):
    loader = W.Loader()
    assert CLI.main(_argv(w, purpose="main", mode="echo", folder=w.tmp / "main_out"), load_generator=loader,
                    **_seams(w)) == CLI.EXIT_REFUSED
    assert "[" in capsys.readouterr().err and loader.calls == 0 and not (w.tmp / "main_out").exists()


def test_명령줄에는_구현을_고르는_인자가_없다():
    opts = {s for a in CLI._parser()._actions for s in a.option_strings}
    assert not {o for o in opts if any(k in o for k in ("measure", "generator", "loader", "seam", "impl", "fake"))}


def test_이미지_경로는_페어의_val_행에서_내고_본실험_모델_모드는_거부한다(tmp_path):
    from vlm.export_writer import ExportRefused

    pairs = tmp_path / "pairs.jsonl"
    rows = [{"image_id": "a1", "image_path": "data/x/a1.png", "client": "C1", "split": "val"},
            {"image_id": "a2", "image_path": str(tmp_path / "a2.png"), "client": "C2", "split": "val"},
            {"image_id": "t1", "image_path": "data/x/t1.png", "client": "C1", "split": "train"}]
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    class G:
        mode, pairs_path = "model", str(pairs)

    got = CLI.pair_image_paths(G, purpose="rehearsal")
    assert got == {"a1": CLI.REPO_ROOT / "data/x/a1.png", "a2": tmp_path / "a2.png"}     # train 행은 없다
    with pytest.raises(ExportRefused) as exc:
        CLI.pair_image_paths(G, purpose="main")
    assert exc.value.code == "main_image_paths_undefined"
    G.mode = "echo"
    assert set(CLI.pair_image_paths(G, purpose="main")) == {"a1", "a2"}                 # 본실험 에코는 val 이다
    G.pairs_path = None
    with pytest.raises(ExportRefused) as exc:
        CLI.pair_image_paths(G, purpose="rehearsal")
    assert exc.value.code == "pairs_path_missing"
