"""독립 대조기의 2 판 갈래 — 판 가림 · 새 칸 읽기 · N1 ~ N7(C 08 2 판 §5-1 · §9-2, D 07 §4 · §16 의 시험 10).

빌더로 지은 1 판 · 2 판 레코드(합성 세계)를 독립 대조기에 대 본다 — 두 구현이 같은 계약을 따로 구현했는지 맞대는 자리다. 독립 대조기는
빌더를 import 하지 않는다(그 규칙은 `tests/test_pairs_independent.py` 가 고정한다). 이 시험은 빌더를 불러 입력을 짓기만 한다.

- 2 판 레코드 + 2 판 표 → 일치(전 범위).
- 1 판 레코드 + 2 판 표(M5 — 표 해시만 바꿔 끼운 메타까지) → 적용 가능한 새 계약 위반으로 거부(N1 ~ N6 의 어긋난 레코드 > 0, N7 도 걸린다).
- 1 판 레코드 + 1 판 표 → 지금처럼 일치(회귀).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "probe"))

import pairs_independent as pi

from corpus.generate import make_pairs_main as MM
from corpus.generate import make_pairs_pilot as P
from corpus.generate.run_cycle_corpus import defect_names
from corpus.rules.limits_loader import load_limits
from corpus.rules.skeleton_gen import load_defect_lexicon
from tests.test_pairs_independent import LABEL_MAP, _ann, _img, _snapshot

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "corpus/rules/limits_v0_pilot.csv"
V2 = REPO / "corpus/rules/limits_v2_pilot.csv"
NEW = {"narrative_measure", "narrative_aggregation", "narrative_mixture", "narrative_distinction",
       "narrative_premise", "narrative_correspondence"}


@pytest.fixture
def world2(tmp_path):
    """정상 · 기공 · 균열+기공 · 슬래그+융합불량(val) · 기공+슬래그(혼재 대상) · AL 기공 · 용입불량 · 폐기 대상. eval 한 장."""
    images = [
        _img("a_normal"), _img("b_poro", codes="2011", n=2), _img("c_crack_poro", client="C2", codes="100;2011", n=2),
        _img("d_slag_lof", split="val", codes="301;401", n=2), _img("e_poro_slag", codes="2011;301", n=2),
        _img("f_al", client="C3", material="AL", codes="2011", n=1), _img("h_lop", codes="402", n=1),
        _img("g_discard", codes="100", n=1),
        {**_img("z_eval", split="eval", codes="2011", n=1, group="g-eval"), "client": ""},
    ]
    anns = [_ann("b_poro", 1, "2011"), _ann("b_poro", 2, "2011", (300, 300, 340, 350)),
            _ann("c_crack_poro", 1, "2011"), _ann("c_crack_poro", 2, "100", (0, 0, 1280, 720)),
            _ann("d_slag_lof", 1, "401"), _ann("d_slag_lof", 2, "301"),
            _ann("e_poro_slag", 1, "2011"), _ann("e_poro_slag", 2, "301", (400, 100, 600, 140)),
            _ann("f_al", 1, "2011"), _ann("h_lop", 1, "402"), _ann("g_discard", 1, "100", valid=False),
            _ann("z_eval", 1, "2011")]
    return _snapshot(tmp_path / "snap", images, anns)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _build(world: Path, out: Path, edition: str, *, meta_edit=None) -> Path:
    """빌더(조립 · 게이트)로 레코드를 짓고, 독립 대조기가 읽는 꼴(레코드 · 폐기 · 회계 · 메타 · 계약서)로 쓴다."""
    ed = MM.EDITIONS[edition]
    table = load_limits(ed.limits_csv, pilot=True)
    m = pd.read_csv(world / "manifest.csv", dtype=str, keep_default_na=False)
    a = pd.read_csv(world / "annotations.csv", dtype=str, keep_default_na=False)
    res = P.build_pairs(m, a, table, defect_names(), load_defect_lexicon(), edition=edition)
    assert res.discarded == [{"image_id": "g_discard", "reasons": [P.NO_VALID_GEOMETRY]}]
    out.mkdir(parents=True)
    P.write_jsonl(out / "pairs.jsonl", res.made)
    P.write_jsonl(out / "discarded.jsonl", res.discarded)
    per: dict[str, dict[str, int]] = {}
    for r in res.made:
        c = per.setdefault(r["client"], {"n_total": 0, "n_defect": 0, "n_normal": 0})
        c["n_total"] += 1
        c["n_defect" if r["skeleton"]["defects"] else "n_normal"] += 1
    P.write_json(out / "counts.json", {"clients": per, "discarded": {"quarantine": len(res.discarded)}})
    body = [x for x in (world / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    limits_table = {"path": str(ed.limits_csv), "sha256": _sha(ed.limits_csv), "status": ed.limits_status}
    meta = {**P.base_meta(res, asset="t"), pi.META_SNAPSHOT: {
        "path": world.name, "snapshot_digest": hashlib.sha256("".join(x + "\n" for x in body).encode()).hexdigest()},
        pi.META_LIMITS: limits_table, pi.META_PATHS_CHECKED: {"n": res.n_input, "of": res.n_input}}
    if ed.text_contract:
        meta.update(identity=MM.identity_block(limits_table), text_contract=ed.text_contract)
    if meta_edit:
        meta_edit(meta)
    P.write_json(out / "PAIRS_META.json", meta)
    P.snapshot_digest(out, ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json"))
    return out


def _compare(world: Path, limits: Path, pairs: Path) -> dict:
    return pi.compare(pi.build_expected(world, limits, LABEL_MAP), pairs)


def test_판은_표의_머리에서_가린다(tmp_path):
    assert pi.limits_edition(V1) == "v1" and pi.limits_edition(V2) == "v2"
    head = V2.read_text(encoding="utf-8").splitlines()
    part = tmp_path / "part.csv"
    part.write_text("\n".join([head[0].rsplit(",", 2)[0], *(h.rsplit(",", 2)[0] for h in head[1:])]) + "\n",
                    encoding="utf-8")
    with pytest.raises(pi.InputContractError, match="일부만"):
        pi.limits_edition(part)


def test_2판_레코드는_2판_표와_전_범위로_일치한다(world2, tmp_path):
    out = tmp_path / "report"
    pairs = _build(world2, tmp_path / "b2", "v2")
    code = pi.main(["--manifest-dir", str(world2), "--limits", str(V2), "--label-map", str(LABEL_MAP),
                    "--pairs", str(pairs), "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mismatch"] == {} and code == 0 and summary["coverage"] == "full"
    assert summary["edition"] == "v2" and summary["conditions"] == {"in_table": True}
    assert summary["text_contract_doc"] == pi.TEXT_CONTRACT_DOC and summary["limits_sha256"] == _sha(V2)
    snap = summary["pairs_snapshot"]
    assert snap["declared"] == snap["computed"] and snap["members_match"]


def test_1판_레코드는_1판_표와_지금처럼_일치한다(world2, tmp_path):
    result = _compare(world2, V1, _build(world2, tmp_path / "b1", "v1"))
    assert result["mismatch"] == {} and result["not_compared"] == []


def test_1판_레코드에_2판_표를_대면_새_계약_위반으로_거부한다(world2, tmp_path):
    """M5 — 메타의 표 해시 · 경로까지 2 판으로 바꿔 끼운 1 판 묶음. 구조 칸 · 수는 같아 옛 검사로는 지나던 길이다(D 07 §4)."""
    def swap(meta):
        meta[pi.META_LIMITS] = {"path": str(V2), "sha256": _sha(V2), "status": "x"}
        meta.update(identity=MM.identity_block(meta[pi.META_LIMITS]), text_contract="v2")

    result = _compare(world2, V2, _build(world2, tmp_path / "m5", "v1", meta_edit=swap))
    assert NEW <= set(result["mismatch"]), result["mismatch"]
    assert result["mismatch"].get("narrative_forbidden_form", 0) > 0          # (가) 1 판 대응 문구는 계약 표 밖이다
    structural = {k for k in result["mismatch"] if not k.startswith("narrative_")}
    assert structural == set(), structural                                    # 바꿔 끼우기는 서술 뜻 검사만이 잡는다


def test_2판_레코드에_1판_표를_대면_메타가_어긋난다(world2, tmp_path):
    result = _compare(world2, V1, _build(world2, tmp_path / "x", "v2"))
    assert {f"meta:{pi.META_LIMITS}.sha256", f"meta:{pi.META_LIMITS}.path", f"meta:{pi.META_IDENTITY}_unexpected"} <= \
        set(result["mismatch"])


@pytest.mark.parametrize("edit, kind", [
    (lambda m: m.pop(pi.META_IDENTITY), f"meta:{pi.META_IDENTITY}"),
    (lambda m: m.update(text_contract="v1"), f"meta:{pi.META_TEXT_CONTRACT}"),
    (lambda m: m[pi.META_IDENTITY].update(limits={"path": "x.csv", "sha256": "0" * 64}), f"meta:{pi.META_IDENTITY}.limits"),
    (lambda m: m[pi.META_IDENTITY].update(conditions={"in_table": False}), f"meta:{pi.META_IDENTITY}.conditions"),
    (lambda m: m[pi.META_LIMITS].update(path="corpus/rules/limits_v0_pilot.csv"), f"meta:{pi.META_LIMITS}.path"),
])
def test_2판_메타의_표_신원이_어긋나면_잡는다(world2, tmp_path, edit, kind):
    assert kind in _compare(world2, V2, _build(world2, tmp_path / "e", "v2", meta_edit=edit))["mismatch"]


def _texts(pairs: Path) -> list[dict]:
    return [json.loads(x) for x in (pairs / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if x]


def _rewrite(pairs: Path, fn) -> None:
    recs = _texts(pairs)
    for r in recs:
        fn(r)
    P.write_jsonl(pairs / "pairs.jsonl", recs)
    P.snapshot_digest(pairs, ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json"))


@pytest.mark.parametrize("seed_group", sorted(pi.N7_SEEDS))
def test_씨앗_문구가_붙으면_금지_꼴이다(world2, tmp_path, seed_group):
    pairs = _build(world2, tmp_path / "s", "v2")
    seed = pi.N7_SEEDS[seed_group][-1].replace(" … ", " 아주 ")

    def add(r):
        if r["image_id"] == "e_poro_slag":
            r["target_text"] = r["target_text"][:-1] + ", " + seed + "."   # 마지막 문장에 이어 붙인다 — 문장 꼴도 바뀐다

    _rewrite(pairs, add)
    assert _compare(world2, V2, pairs)["mismatch"].get("narrative_forbidden_form") == 1


def test_씨앗_문구는_계약_문장_자신을_걸지_않는다():
    for text in pi.V2_FIXED.values():
        assert not pi._seed_hit(text), text


#: 표에 아직 없는 띠 꼴의 합성 행 — 수는 실물 표의 수와 겹치지 않게 골랐다(시험용 합성).
SYNTH_BAND_ROWS = [
    SimpleNamespace(rule_id="synth-const", limit_rule="const", unit="percent", limit_op="lt", limit_value="7", limit_factor=None,
                    limit_cap=None, ratio_basis=None, thickness_min="11", thickness_max="19"),
    SimpleNamespace(rule_id="synth-prop", limit_rule="prop_t", unit="mm", limit_op="le", limit_value=None, limit_factor="0.45",
                    limit_cap=None, ratio_basis="s", thickness_min="0", thickness_max="31.01"),
    SimpleNamespace(rule_id="synth-cap", limit_rule="prop_t_cap", unit="mm", limit_op="lt", limit_value=None,
                    limit_factor="0.15", limit_cap="2.5", ratio_basis="a", thickness_min="41.01", thickness_max=None),
    SimpleNamespace(rule_id="synth-none", limit_rule="none_permitted", unit=None, limit_op=None, limit_value=None,
                    limit_factor=None, limit_cap=None, ratio_basis=None, thickness_min="0", thickness_max=None),
]


def _builder_band(row) -> str:
    """생성 쪽(1 판 `criterion_ko` · `thickness_ko`)이 짓는 띠 하나 — 독립 문법과 맞댈 다른 구현이다."""
    from corpus.rules.clause_text import criterion_ko, thickness_ko

    band = thickness_ko(row.thickness_min, row.thickness_max)
    return f"{band if band == '모든 두께' else '두께 ' + band}: {criterion_ko(row)}"


def test_띠_문법은_생성_쪽_띠를_모두_받는다():
    """09 §10 I1 의 양성 — 생성 쪽이 표의 모든 행과 합성 행(표에 아직 없는 단위 · 부등호 · 비례 기준 · 두께 구간)으로 짓는 띠를 독립
    문법이 그대로 받는다. 띠를 ' / ' 로 이은 목록도 받는다."""
    bands = [_builder_band(r) for r in list(load_limits(V2, pilot=True).rows) + SYNTH_BAND_ROWS]
    assert len(bands) > len(SYNTH_BAND_ROWS)
    for band in bands:
        assert pi._BANDS.fullmatch(band), band
    assert pi._BANDS.fullmatch(" / ".join(bands))


@pytest.mark.parametrize("tail", [", 따라서 합격이다", " 이므로 합격이다", "이다", " (참고)", " / 합격"])
def test_띠에_남는_어구가_있으면_띠가_아니다(tail):
    for row in SYNTH_BAND_ROWS:
        band = _builder_band(row)
        assert pi._BANDS.fullmatch(band) and not pi._BANDS.fullmatch(band + tail), band + tail


@pytest.mark.parametrize("bad", ["두께 11 mm 이상 19 mm 미만: 7 cm 미만", "두께 11 mm 이상 19 mm 미만: 7 mm 이상",
                                 "두께 11 mm 이상 19 mm 미만: 7.50 mm 미만", "두께 19 mm 미만 11 mm 이상: 7 mm 미만",
                                 "모든 두께: 판정 기준은 7 mm 미만", "두께 11 mm 이상 19 mm 미만:7 mm 미만"])
def test_띠의_단위_부등호_수_꼴_접속이_계약_밖이면_받지_않는다(bad):
    assert not pi._BANDS.fullmatch(bad)


def test_띠의_마지막_마침표_앞에_판정_어구를_끼우면_금지_꼴만_걸린다(world2, tmp_path):
    """09 §10 I1 의 음성 — 정상 띠 문장의 마지막 마침표 앞에 ', 따라서 합격이다' 를 끼운다. 수 · 측정량 낱말 · '미만' 수가 그대로라
    N1 ~ N6 · N8 은 못 잡는다. 닫힌 띠 문법이 남는 어구를 잡는다(N7)."""
    pairs = _build(world2, tmp_path / "j", "v2")
    done = []

    def add(r):
        if r["image_id"] != "b_poro":
            return
        text = r["target_text"]
        m = re.search(r"다음과 같다\. (.+?)\.(?= |$)", text)
        new = text[:m.end(1)] + ", 따라서 합격이다" + text[m.end(1):]

        def nums(s: str) -> list[str]:
            return sorted(re.findall(r"\d+(?:\.\d+)?", s))

        assert nums(new) == nums(text) and new.count("미만") == text.count("미만")
        assert all(new.count(w) == text.count(w) for w in pi.MEASURES)
        r["target_text"] = new
        done.append(r["image_id"])

    _rewrite(pairs, add)
    assert done == ["b_poro"]
    assert _compare(world2, V2, pairs)["mismatch"] == {"narrative_forbidden_form": 1}


def test_조건_문장을_빼면_그_항목으로_잡는다(world2, tmp_path):
    pairs = _build(world2, tmp_path / "d", "v2")
    _rewrite(pairs, lambda r: r.update(target_text=r["target_text"].replace(" " + pi.V2_FIXED["C6"], "")))
    got = _compare(world2, V2, pairs)["mismatch"]
    assert got.get("narrative_mixture") == 1 and "narrative_forbidden_form" not in got


def _b_regression() -> dict | None:
    rel = "evaluation/vectors/narrative_b_regression.v1.json"
    p = REPO / rel
    raw = p.read_bytes() if p.is_file() else None
    if raw is None:
        out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"91e83979:{rel}"], capture_output=True, check=False)
        raw = out.stdout if out.returncode == 0 else None
    if raw is None:
        return None
    assert hashlib.sha256(raw).hexdigest() == "d75253aedf68747dd478d02fe7ba7cd8d15fc4dc446b17aaaa02d309593e04c6"
    return json.loads(raw)


def test_N7_씨앗_목록은_평가_쪽_B_회귀_사례의_목록과_같다():
    """09 §6 Q13 — 독립 대조기 상수로 두고, 평가 쪽 고정 사례와의 동일성은 시험이 맞댄다."""
    doc = _b_regression()
    if doc is None:
        pytest.skip("평가 쪽 기대 사례를 읽지 못했다")
    assert {k: list(v) for k, v in pi.N7_SEEDS.items()} == doc["n7_seeds"]


def test_문장_출처_표지는_그_커밋의_blob_을_가리킨다():
    """요약에 싣는 출처(D 07 §18-4) — 상수끼리 맞대지 않고 그 커밋의 blob 을 다시 잰다. 문서에 절이 덧붙어도 옮겨 온 판을 가리킨다."""
    doc = pi.TEXT_CONTRACT_DOC
    out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{doc['commit']}:{doc['path']}"], capture_output=True,
                         check=False)
    if out.returncode != 0:
        pytest.skip("그 커밋이 이 저장소에 없다")
    assert hashlib.sha256(out.stdout).hexdigest() == doc["sha256"]
    assert all(s in out.stdout.decode("utf-8") for s in (pi.V2_FIXED["C5"], pi.V2_FIXED["C8"], pi.V2_FIXED["C12"]))


def test_계약_문장은_평가_쪽_기대_사례의_글자와_같다():
    doc = _b_regression()
    if doc is None:
        pytest.skip("평가 쪽 기대 사례를 읽지 못했다")
    s = doc["contract_sentences"]
    assert {k: pi.V2_FIXED[k] for k in ("C4", "C5", "C6", "C7", "C8", "C9", "C11", "C12")} == \
        {k: s[k] for k in ("C4", "C5", "C6", "C7", "C8", "C9", "C11", "C12")}
    assert pi.V2_FIXED["C13a"] + " " + pi.V2_FIXED["C13b"] == s["C13"]
    assert pi.V2_FIXED["C3b"] + " " == s["C3b"].split("{띠}")[0]
