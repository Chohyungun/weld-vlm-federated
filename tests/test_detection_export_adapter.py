"""본실험 검출 export → 계약 #4 어댑터 (13_spec_D §2-3 · 17번 시드 1 첫 채점).

파일럿에서는 D 가 직접 추론해 레코드를 만들었고(`detect_infer._record_from_result`),
본실험에서는 C 가 낸 jsonl 을 옮긴다. **두 경로가 같은 레코드를 만들어야** 파일럿과
본실험이 같은 채점을 받는다 — 그 등가가 이 파일의 첫 시험이다.

나머지는 조용히 틀릴 수 있는 자리를 막는다.

- 클라이언트 번호 → 이름(`c0`→`C1`): 밀려도 어디서도 안 걸리고 RQ3 이 통째로 뒤바뀐다
- 클래스 인덱스 → ISO 코드: 순서가 갈리면 전 지표가 다른 클래스에 실린다
- 좌표 규약: 어댑터가 정정하면 `coord_space_contract` 게이트가 영원히 통과한다
"""

from __future__ import annotations

import json

import pytest

from data.label_map import load_label_map
from evaluation.adapters import DETECTION_EXPORT_TAGS, adapt_detection_export
from evaluation.detect_infer import CHECKPOINTS, cell_tag
from evaluation.params import CLASS_NAMES

LM = load_label_map()


def _line(image_id: str, boxes: list[dict], **kw) -> str:
    row = {"image_id": image_id, "boxes": boxes,
           "coord_space": "ABS_ORIG", "coord_cfg_hash": "abc123"}
    row.update(kw)
    return json.dumps(row, ensure_ascii=False)


def _adapt(lines, *, cell="sep_fed", client=None, image_size=None):
    return adapt_detection_export(
        lines, cell=cell, client=client, seed=20260828,
        class_names=CLASS_NAMES, iso_code_of=LM.iso_code, image_size=image_size,
    )


# --------------------------------------------------------------------------------------
# 태그·클라이언트 대응 — 밀리면 RQ3 이 뒤바뀐다
# --------------------------------------------------------------------------------------

def test_export_태그가_파일럿_체크포인트_대응과_같다():
    """`sep_local_c0` ↔ `C1` 은 파일럿에서 이미 쓰던 대응이다. 두 표가 갈리면
    같은 npz 가 파일럿에서는 C1, 본실험에서는 다른 클라이언트로 채점된다."""
    # 파일럿 체크포인트 경로의 파일명 → (칸, 클라이언트). ④ 는 라운드 번호가 붙은
    # `global_r003.npz` 라 이름이 다르므로 로컬 3벌만 이름으로 대조하고 나머지는 칸으로 본다.
    from_ckpt = {
        rel.split("/")[-1].removesuffix(".npz"): (cell, client)
        for cell, client, rel in CHECKPOINTS
    }
    by_cell = {(cell, client) for cell, client, _ in CHECKPOINTS}
    for src_tag, cell, client in DETECTION_EXPORT_TAGS:
        if src_tag in from_ckpt:
            assert from_ckpt[src_tag] == (cell, client), f"{src_tag} 대응이 파일럿과 다르다"
        assert (cell, client) in by_cell, f"{src_tag}: 파일럿에 없는 칸 대응"
    assert len(DETECTION_EXPORT_TAGS) == len(CHECKPOINTS)


def test_계약_태그가_채점기가_찾는_이름과_같다():
    tags = [cell_tag(c, cl) for _, c, cl in DETECTION_EXPORT_TAGS]
    assert tags == ["sep_local_C1", "sep_local_C2", "sep_local_C3",
                    "sep_central", "sep_fed"]


def test_클라이언트가_레코드에_실린다():
    rep = _adapt([_line("i1", [])], cell="sep_local", client="C2")
    assert rep.records[0].client == "C2" and rep.records[0].cell == "sep_local"


def test_sep_local_에_클라이언트가_없으면_스키마가_거부한다():
    """계약 #4 의 교차 검증. 어댑터가 빈 값으로 넘기면 여기서 죽어야 한다."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="클라이언트"):
        _adapt([_line("i1", [])], cell="sep_local", client=None)


# --------------------------------------------------------------------------------------
# 파일럿 경로와의 등가
# --------------------------------------------------------------------------------------

def test_파일럿_레코드와_같은_모양을_만든다():
    """`_record_from_result` 가 만드는 것과 필드·값이 같아야 한다(추론 결과 스텁 경유)."""
    from types import SimpleNamespace

    from evaluation.detect_infer import _record_from_result
    from evaluation.params import ScoringParams

    class _Coords(list):
        def tolist(self):
            return list(self)

    class _Box:
        def __init__(self, cls, xyxy, conf):
            self.cls = cls
            self.xyxy = [_Coords(xyxy)]
            self.conf = conf

    xyxy = [10.0, 20.0, 110.0, 220.0]
    res = SimpleNamespace(names=dict(enumerate(CLASS_NAMES)),
                          boxes=[_Box(2, xyxy, 0.4321)])
    params = ScoringParams(snapshot=".", pilot=".", out=".", seed=20260828,
                           profile="main")
    pilot = _record_from_result({"image_id": "i1"}, res, LM, "sep_fed", None, params)
    mine = _adapt([_line("i1", [{"cls": 2, "xyxy_px": xyxy, "conf": 0.4321}])]).records[0]

    a, b = pilot.model_dump(), mine.model_dump()
    b.pop("coord_cfg_hash"), a.pop("coord_cfg_hash")     # export 만 싣는 필드
    assert a == b


def test_클래스_인덱스가_ISO_코드로_순서대로_바뀐다():
    boxes = [{"cls": i, "xyxy_px": [0.0, 0.0, 10.0, 10.0], "conf": 0.5}
             for i in range(len(CLASS_NAMES))]
    rec = _adapt([_line("i1", boxes)]).records[0]
    assert [d.iso_code for d in rec.defects] == [LM.iso_code(n) for n in CLASS_NAMES]


def test_클래스_범위_밖_인덱스는_항목만_버리고_센다():
    rep = _adapt([_line("i1", [
        {"cls": 0, "xyxy_px": [0.0, 0.0, 10.0, 10.0], "conf": 0.5},
        {"cls": 9, "xyxy_px": [0.0, 0.0, 10.0, 10.0], "conf": 0.5},
    ])])
    assert rep.n_unknown_code == 1 and rep.n_boxes == 1
    assert len(rep.records) == 1 and len(rep.records[0].defects) == 1


# --------------------------------------------------------------------------------------
# 버리는 것을 센다 — 레코드는 살린다(80번 D7)
# --------------------------------------------------------------------------------------

def test_퇴화_박스는_항목만_버린다():
    rep = _adapt([_line("i1", [
        {"cls": 0, "xyxy_px": [10.0, 10.0, 10.0, 20.0], "conf": 0.5},   # x1 == x2
        {"cls": 0, "xyxy_px": [10.0, 10.0, 20.0, 20.0], "conf": 0.5},
    ])])
    assert rep.n_bad_items == 1 and rep.n_boxes == 1
    assert len(rep.records) == 1 and rep.records[0].parse_ok


def test_망가진_항목도_레코드를_죽이지_않는다():
    rep = _adapt([_line("i1", [
        {"cls": 0, "xyxy_px": [0.0, 0.0, 10.0], "conf": 0.5},           # 좌표 3개
        {"cls": 0, "conf": 0.5},                                        # 키 부재
        {"cls": 0, "xyxy_px": [0.0, 0.0, 10.0, 10.0], "conf": 0.5},
    ])])
    assert rep.n_bad_items == 2 and rep.n_boxes == 1 and len(rep.records) == 1


def test_boxes_가_리스트가_아니면_그_레코드만_실패다():
    rep = _adapt([_line("i1", None), _line("i2", [])])   # type: ignore[arg-type]
    assert rep.adapter_failures == {"schema_violation": 1}
    bad = next(r for r in rep.records if r.image_id == "i1")
    assert bad.parse_ok is False and bad.parse_error == "schema_violation"
    assert next(r for r in rep.records if r.image_id == "i2").parse_ok is True


def test_경계_이탈은_세기만_하고_고치지_않는다():
    """클리핑은 IoU 를 올리는 방향으로만 작동한다 — 답을 고쳐 주는 셈이다."""
    rep = _adapt([_line("i1", [{"cls": 0, "xyxy_px": [-5.0, 0.0, 1300.0, 10.0],
                                "conf": 0.5}])],
                 image_size={"i1": (1280, 720)})
    assert rep.n_boxes_out_of_bounds == 1 and rep.n_boxes == 1
    assert rep.records[0].defects[0].bbox_px == (-5.0, 0.0, 1300.0, 10.0)


# --------------------------------------------------------------------------------------
# 좌표 규약 — 어댑터가 정정하지 않는다
# --------------------------------------------------------------------------------------

def test_coord_space_는_파일이_말하는_값_그대로_실린다():
    """정정하면 `coord_space_contract` 게이트가 영원히 통과한다 — 함정 #4 의 방어선이 무너진다."""
    rep = _adapt([_line("i1", [], coord_space="NORM_1000")])
    assert rep.records[0].coord_space == "NORM_1000"
    assert rep.records[0].coord_cfg_hash == "abc123"


def test_검출_레코드는_판정불가이고_인용이_없다():
    """판정부(⑤) 미실행 — 검출 축 선채점. 지어낸 판정을 싣지 않는다."""
    rec = _adapt([_line("i1", [])]).records[0]
    assert rec.verdict == "판정불가" and rec.cited_clauses == []
    assert all(d.retrieved is None for d in rec.defects)


def test_빈_줄은_레코드를_만들지_않는다():
    rep = _adapt(["", "  ", _line("i1", [])])
    assert rep.n_lines == 1 and len(rep.records) == 1
