"""본실험 페어 빌더(`make_pairs_main`) — 얇은 모듈이 스스로 정하는 것만 본다: 쓰기 전 가드와 메타.

조립·검증·회계는 `make_pairs_pilot` 의 것을 그대로 부르므로 `test_pairs_gate.py` 가 맡는다.
실물 매니페스트·봉인 자산은 읽지 않는다.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from corpus.generate import make_pairs_main as MM
from corpus.generate import make_pairs_pilot as M
from corpus.generate.frozen_out import CONTRACT_NAME, FrozenDirectoryError
from corpus.generate.run_cycle_corpus import defect_names
from corpus.rules import limits_loader
from corpus.rules.skeleton_gen import load_defect_lexicon

NAMES = defect_names()
TABLE = limits_loader.load_limits(str(M.LIMITS_CSV), pilot=True)
DIM = next(c for c, b in sorted(M.clause_basis(TABLE, "ST").items()) if "none_permitted" not in b["rule_kinds"])


def _no_loading(monkeypatch):
    """가드가 스냅샷 적재보다 **먼저** 도는지 본다 — 적재가 불리면 실패다."""
    def boom(*_a, **_k):
        raise AssertionError("가드보다 먼저 스냅샷을 읽었다")
    monkeypatch.setattr("data.manifest_io.verify_snapshot", boom)
    monkeypatch.setattr("data.manifest_io.load_snapshot", boom)


@pytest.mark.parametrize("argv", [[], ["--out", "x"], ["--date", "2026-09-21"], ["--out", "x", "--date", "어제"]])
def test_out_과_date_는_필수이고_날짜는_ISO_꼴이다(argv):
    with pytest.raises(SystemExit) as e:
        MM.main(argv)
    assert e.value.code == 2


def test_봉인된_경로에는_쓰지_않는다(tmp_path, monkeypatch):
    _no_loading(monkeypatch)
    sealed = tmp_path / "pairs_main_v1"
    sealed.mkdir()
    (sealed / CONTRACT_NAME).write_text("0" * 64 + "  pairs.jsonl\n", encoding="utf-8")
    before = (sealed / CONTRACT_NAME).read_bytes()
    with pytest.raises(FrozenDirectoryError):
        MM.main(["--out", str(sealed), "--date", "2026-09-21"])
    assert (sealed / CONTRACT_NAME).read_bytes() == before and len(list(sealed.iterdir())) == 1


def test_비어_있지_않은_경로에는_쓰지_않는다(tmp_path, monkeypatch):
    _no_loading(monkeypatch)
    (tmp_path / "남의_파일.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="비어 있지 않다"):
        MM.main(["--out", str(tmp_path), "--date", "2026-09-21"])
    assert [p.name for p in tmp_path.iterdir()] == ["남의_파일.txt"]


def test_main_이_경로_실재_검사를_켜서_부른다(monkeypatch, tmp_path):
    """메타의 '경로를 확인했다' 가 상수였다 — 인자를 끄면 메타는 계속 참이라 적고 레코드는 그대로라
    독립 대조도 못 잡는다. 지금은 빌드가 실제로 본 수를 싣지만, **그 인자를 주는 것 자체**는 여기서 고정한다."""
    seen = {}

    def fake_build(*a, **kw):
        seen.update(kw)
        raise RuntimeError("여기까지만 본다")

    monkeypatch.setattr("data.manifest_io.verify_snapshot", lambda *a, **k: "d" * 64)
    monkeypatch.setattr("data.manifest_io.load_snapshot", lambda *a, **k: SimpleNamespace(manifest=None, annotations=None))
    monkeypatch.setattr(MM.P, "build_pairs", fake_build)
    with pytest.raises(RuntimeError):
        MM.main(["--out", str(tmp_path / "o"), "--date", "2026-09-21"])
    assert seen.get("check_paths") is True and seen.get("root") == M.REPO


def test_메타의_경로_확인_주장은_빌드가_실제로_본_수다():
    """상수로 적으면 검사를 꺼도 '확인했다' 가 남는다(검토 M-9)."""
    m, a = _frames()
    res = M.build_pairs(m, a, TABLE, NAMES, load_defect_lexicon())          # check_paths 를 주지 않았다
    meta = MM.build_meta(res, m, manifest_digest="d" * 64, date="2026-09-21")
    assert meta["image_paths_checked"]["n"] == 0 and meta["image_paths_checked"]["of"] == res.n_input


def test_본실험_입력은_동결_매니페스트이고_표는_파일럿_초판이다():
    assert MM.SNAP == M.REPO / "data/interim/manifest_v1"
    assert MM.LIMITS_CSV == M.LIMITS_CSV and MM.ASSET == "d4_pairs_main" and MM.VERSION == "v1"


def _frames():
    def row(iid, sha, has_defect, n, split="train"):
        return {"image_id": iid, "group_id": "g-" + iid, "rel_path": f"tiles/{iid}.png", "client": "C1" if split != "eval" else "",
                "split": split, "material": "ST", "has_defect": has_defect, "n_defects": n, "width_px": 100,
                "height_px": 100, "sha256": sha}
    # train·val 안에서만 해시를 겹치게 둔다. eval 과 겹치면 그것은 평가 격리 위반이라 빌드가 멈춘다.
    rows = [row("img-1", "aa", True, 1), row("img-2", "aa", False, 0), row("img-3", "bb", False, 0),
            row("img-4", "bb", False, 0), row("img-5", "cc", False, 0), row("img-9", "ee", False, 0, "eval")]
    anns = [{"image_id": "img-1", "ann_id": "img-1#0", "geom_valid": True, "iso_code": DIM, "bbox_x1_px": 1,
             "bbox_y1_px": 1, "bbox_x2_px": 9, "bbox_y2_px": 9, "major_axis_px": 8.0, "equiv_diameter_px": 8.0}]
    return pd.DataFrame(rows), pd.DataFrame(anns)


def test_같은_바이트인데_라벨이_다른_행을_매니페스트에서_센다():
    """같은 화소에 정상 서술과 결함 서술이 함께 나가는 자리다(06번 §1-나). eval 행은 세지 않는다."""
    m, _ = _frames()
    assert MM.same_bytes_within_train_val(m) == {"n_keys": 2, "n_rows": 4, "n_keys_label_conflict": 1,
                                                 "n_keys_n_defects_differ": 1}


def test_메타가_표의_상태·사람_검증·상속한_한계를_싣는다():
    m, a = _frames()
    res = M.build_pairs(m, a, TABLE, NAMES, load_defect_lexicon())
    meta = MM.build_meta(res, m, manifest_digest="d" * 64, date="2026-09-21")
    assert meta["asset"] == "d4_pairs_main_v1" and meta["built_on"] == "2026-09-21"
    # 실행 신원이 가리키는 것을 산출물이 스스로 말한다 — digest 가 입력만의 함수가 아니라는 사실(검토 L5)
    assert "git_commit" in meta["run_identity"] and "pairs.jsonl" in meta["run_identity"]
    assert meta["input_snapshot"] == {"path": "data/interim/manifest_v1", "snapshot_digest": "d" * 64,
                                      "splits_used": ["train", "val"]}
    assert meta["limits_table"]["sha256"] == hashlib.sha256(M.LIMITS_CSV.read_bytes()).hexdigest()
    assert "이중 검수 전" in meta["limits_table"]["status"]
    assert "known_defects_in_v1" not in meta                       # 파일럿 v1 의 회계는 본실험 자산의 것이 아니다
    assert meta["validated_by"] == "rule" and "형식 통과율" in meta["pass_rate"]["meaning"]
    assert meta["inherited_split_limits"]["same_bytes_within_train_val"]["n_keys_label_conflict"] == 1
    assert meta["inherited_split_limits"]["near_duplicates_with_eval"]["train_val_vs_eval_pixel_twins_lower_bound"] == 33
    assert meta["coord_space"] == "ABS_ORIG" and meta["eval_isolation"]["eval_group_id_overlap"] == 0
    assert "null" in meta["clause_granularity"] and "구조화 JSON" in meta["answer_contract"]
