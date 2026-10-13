"""합성 입력의 장비 단가 계측(`scripts/probe/synth_rate.py`)의 입력 쪽. 모델 없이 CPU 로 돈다.

계측 자체(학습 · 생성)는 GPU 가 있어야 하므로 여기서 돌리지 않는다. 여기서 보는 것은 입력이 **합성뿐인지**(작업 폴더 밖을 가리키지 않는다),
형상이 실제 이미지와 같은지(1280×720), 타깃이 학습 경로의 함수로 만들어지는지, 혼합 비중, 산출을 덮지 않는지다.
외부 검토 §73-3 뒤로는 합성 폴더가 이 스크립트 전용인지(표지 · 이름 꼴 · 같은 이름의 화소), 입력 해시가 폴더 위치와 무관한지,
실제로 적재된 판이 설정과 다를 때 멈추는지도 본다.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from scripts.probe import synth_rate as SR


@pytest.fixture(autouse=True)
def _fresh_used():
    SR.USED_IMAGES.clear()
    yield
    SR.USED_IMAGES.clear()


def test_합성_행은_작업_폴더_안의_1280x720_이미지만_가리킨다(tmp_path):
    from PIL import Image

    work = tmp_path / "w"
    rows = SR.make_rows(work, [0, 2, 8, 2], tag="t", n_images_per_k=2)
    assert [len(r["skeleton"]["defects"]) for r in rows] == [0, 2, 8, 2]
    for r in rows:
        p = Path(r["image_path"])
        assert p.parent == work
        with Image.open(p) as im:
            assert im.size == (SR.IMG_W, SR.IMG_H) == (1280, 720)
        for d in r["skeleton"]["defects"]:
            x1, y1, x2, y2 = d["bbox_px"]
            assert 0 <= x1 < x2 <= SR.IMG_W and 0 <= y1 < y2 <= SR.IMG_H
    assert rows[1]["image_path"] == rows[3]["image_path"]          # 장은 행 번호를 장수로 나눈 나머지 자리로 돌려 쓴다 — 1 · 3 은 같은 자리
    assert sorted(p.name for p in work.iterdir()) == [SR.WORK_MARK, "syn_k00_00.png", "syn_k02_01.png", "syn_k08_00.png"]


def test_유형_코드는_사상표에서_온다():
    lm = yaml.safe_load((ROOT / "configs/label_map.yaml").read_text(encoding="utf-8"))
    used = set(lm["sources"]["aihub71761"]["mapping"].values())
    assert sorted(SR._codes()) == sorted(str(lm["defect_types"][k]["iso_code"]) for k in used)


def test_타깃은_학습_경로의_build_target_으로_만든다(tmp_path):
    from vlm.coords import ImageGeom
    from vlm.pilot_vlm import build_target

    r = SR.make_rows(tmp_path, [3], tag="t")[0]
    t = json.loads(build_target(r, ImageGeom(orig_w=1280, orig_h=720)))
    assert [d["iso_code"] for d in t["defects"]] == [d["type"] for d in r["skeleton"]["defects"]]
    assert t["verdict"] == "판정불가"                                   # 실제 페어도 verdict 가 null 이다
    assert t["cited_clauses"] == sorted({f"SYN-{d['type']}" for d in r["skeleton"]["defects"]})


def test_혼합은_비중의_최대_잔여_몫대로_나뉘고_평균_결함은_1_73():
    from vlm.rehearsal_lists import largest_remainder

    ks = SR._mix_ks(384)
    want = largest_remainder(384, {str(k): w for k, w in SR.MIX.items()})
    assert Counter(ks) == Counter({int(k): n for k, n in want.items() if n})
    assert sum(k * w for k, w in SR.MIX.items()) / sum(SR.MIX.values()) == 1.73
    assert SR._mix_ks(384) == ks                                       # 같은 씨앗이면 같은 순서다


def test_산출이_있으면_덮지_않는다(tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    out.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["synth_rate", "--part", "tokens", "--work", str(tmp_path / "w"), "--out", str(out)])
    assert SR.main() == 2
    assert out.read_text(encoding="utf-8") == "{}"
    assert not (tmp_path / "w").exists()


def test_빈_폴더에서_시작하면_합성_규칙의_표지를_쓴다(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    SR.make_rows(work, [1], tag="t")
    assert json.loads((work / SR.WORK_MARK).read_text(encoding="utf-8")) == SR.WORK_RULE


def test_표지_없는_폴더는_거부하고_그_안의_파일을_건드리지_않는다(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "syn_k01_00.png").write_bytes(b"not ours")          # 이름 꼴이 같아도 표지가 없으면 우리 폴더가 아니다
    with pytest.raises(SR.SynthWorkRefused) as e:
        SR.make_rows(work, [1], tag="t")
    assert e.value.code == "work_not_dedicated"
    assert (work / "syn_k01_00.png").read_bytes() == b"not ours"
    assert not (work / SR.WORK_MARK).exists()


@pytest.mark.parametrize("name,kind", [("real.png", "file"), ("syn_k01_00.jpg", "file"), ("syn_k01_00.png", "dir")])
def test_표지가_있어도_이_스크립트가_쓰지_않는_항목이_있으면_거부한다(tmp_path, name, kind):
    work = tmp_path / "w"
    SR.prepare_work(work)
    (work / name).mkdir() if kind == "dir" else (work / name).write_bytes(b"x")
    with pytest.raises(SR.SynthWorkRefused) as e:
        SR.make_rows(work, [0], tag="t")
    assert e.value.code == "work_foreign_entry"


def test_표지의_합성_규칙이_다르면_거부한다(tmp_path):
    work = tmp_path / "w"
    SR.prepare_work(work)
    (work / SR.WORK_MARK).write_text(json.dumps({**SR.WORK_RULE, "gen_seed": SR.GEN_SEED + 1}), encoding="utf-8")
    with pytest.raises(SR.SynthWorkRefused) as e:
        SR.make_rows(work, [0], tag="t")
    assert e.value.code == "work_rule_mismatch"


def test_같은_이름의_파일이_지금_규칙의_합성_그림과_다르면_거부한다(tmp_path):
    from PIL import Image

    work = tmp_path / "w"
    SR.make_rows(work, [2], tag="t")
    q = work / "syn_k02_00.png"
    Image.new("RGB", (1280, 720), (128, 128, 128)).save(q)        # 크기 · 이름은 같고 화소만 다르다
    with pytest.raises(SR.SynthWorkRefused) as e:
        SR.make_rows(work, [2], tag="t")
    assert e.value.code == "synth_content_mismatch"


def test_같은_규칙이면_다시_써도_폴더_위치가_달라도_입력_해시가_같다(tmp_path):
    a = SR.make_rows(tmp_path / "a", [0, 2, 8], tag="t")
    ha, ia = SR.rows_sha256(a), dict(SR.USED_IMAGES)
    again = SR.make_rows(tmp_path / "a", [0, 2, 8], tag="t")          # 같은 폴더를 다시 쓴다 — 화소를 맞대고 통과
    assert SR.rows_sha256(again) == ha
    SR.USED_IMAGES.clear()
    b = SR.make_rows(tmp_path / "b", [0, 2, 8], tag="t")
    assert SR.rows_sha256(b) == ha and SR.USED_IMAGES == ia
    assert str(tmp_path) not in json.dumps({"r": ha, "i": ia})


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_실제로_적재된_판이_설정과_다르면_계측하지_않고_멈춘다():
    cfg = _Obj(model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    same = _Obj(config=_Obj(_commit_hash=cfg.model_revision))
    proc = _Obj(tokenizer=_Obj(init_kwargs={"_commit_hash": cfg.model_revision}))
    assert SR.loaded_identity(same, proc, cfg) == {"model_commit_hash": cfg.model_revision, "processor_commit_hash": cfg.model_revision,
                                                   "model_matches_config": True, "processor_matches_config": True}
    with pytest.raises(SR.SynthWorkRefused) as e:
        SR.loaded_identity(_Obj(config=_Obj(_commit_hash="0" * 40)), proc, cfg)
    assert e.value.code == "loaded_revision_mismatch"
    with pytest.raises(SR.SynthWorkRefused):
        SR.loaded_identity(same, _Obj(tokenizer=_Obj(init_kwargs={"_commit_hash": "0" * 40})), cfg)
    unknown = SR.loaded_identity(_Obj(config=_Obj()), _Obj(), cfg)          # 판을 알 수 없으면 '모름' 으로 싣는다
    assert unknown["model_commit_hash"] is None and unknown["model_matches_config"] is None


def test_명령은_모델을_올리기_전에_전용이_아닌_폴더를_거부한다(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    (work / "real.png").write_bytes(b"x")
    monkeypatch.setattr(SR, "_env", lambda: (_ for _ in ()).throw(AssertionError("장비를 열었다")))
    monkeypatch.setattr(SR, "_cfg", lambda: (_ for _ in ()).throw(AssertionError("설정을 읽었다")))
    monkeypatch.setattr(sys, "argv", ["synth_rate", "--part", "train", "--work", str(work), "--out", str(tmp_path / "o.json")])
    assert SR.main() == 2
    assert not (tmp_path / "o.json").exists()
