"""서술 추출 · 상태의 학습 쪽 구현(`vlm/narrative.py`) — 02 2판 §3-5 의 새 시험 6 · 7(지시 1022b, 외부 검토 194 M3).

- 벡터 — D 가 낸 벡터 파일을 **바이트로** 읽고 sha256 을 맞댄 뒤 전 사례의 다섯 값(state · narrative · front_text · stop_kept ·
  upstream_reason)이 값과 자료형까지 기대와 같아야 한다. 정본은 03f-2(`vlm.narrative.VECTOR_SHA256`)다 — 정규 배포 자리 → 지금 자리
  → 그 바이트를 고정한 D 커밋의 blob 순서로 찾고(`vlm.narrative.canonical_vector_bytes`), 셋 다 없으면 **실패**한다(지시 1022c
  덧붙임 E — 정본 대조를 건너뛰지 않는다). 초안 03f 는 따로 본다.
- 정본 파서와의 일치 — 레코드 실패 판정과 사유가 `vlm.gen_parse.parse_generation` 과 같다(해독 성공만으로 유효로 보지 않는다).
- 합성 경계(페어 없이도 늘 돈다) — 유효 JSON 직후에서 끝나면 길이 종료여도 **미기재**다(M3). 종료 사유가 먼저다. `{}` 뒤 설명은
  스키마 위반 · 상자 하나만 깨진 출력은 레코드 실패가 아니다. 펜스 표지 줄은 04 §7-4 의 좁힌 문법으로 가린다.
- 정답 타깃 자기 시험 — 학습 풀의 새 타깃을 종료 토큰으로 넣으면 추출문 = `target_text` · 완결이다. JSON 직후 · 서술 가운데 ·
  JSON 안에서 자르면 미기재 · 잘림 · 레코드 실패다. 페어가 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm import narrative as N
from vlm.gen_parse import parse_generation


def _pairs() -> tuple[Path | None, str | None]:
    """설정의 학습 풀 페어(`uni_pairs`)와 등록 digest. 설정을 읽지 못하거나 페어가 없으면 (None, None)."""
    try:
        from vlm.uni_config import load_uni_config

        cfg = load_uni_config()
    except Exception:                     # noqa: BLE001 — 설정이 없는 사본에서는 합성 경계 시험만 돈다
        return None, None
    pp = Path(cfg.pairs_path) if cfg.pairs_path else None
    return (pp, str(cfg.pairs_digest)) if pp is not None and pp.is_file() else (None, None)


PAIRS, PAIRS_DIGEST = _pairs()
J = '{"defects":[],"verdict":"판정불가","cited_clauses":[]}'


def _draft():
    return N.load_vectors(N.DRAFT_VECTOR_PATH, sha256=N.DRAFT_VECTOR_SHA256)


def _canonical(tmp: Path):
    """정본 벡터 — `canonical_vector_bytes` 의 바이트를 임시 파일에 꺼내 `load_vectors` 로 다시 sha256 을 맞댄다. 없으면
    `VectorRejected` 가 그대로 올라 시험이 실패한다(건너뛰지 않는다)."""
    raw, _where = N.canonical_vector_bytes()
    p = tmp / "canonical_vectors.json"
    p.write_bytes(raw)
    return N.load_vectors(p, sha256=N.VECTOR_SHA256)


# ================================================================ 시험 7 — 벡터
def test_초안_벡터의_전_사례에서_다섯_값이_기대와_같다():
    vecs, sha = _draft()
    assert sha == N.DRAFT_VECTOR_SHA256 and len(vecs) == 36
    res = N.run_vectors(vecs)
    assert res["mismatches"] == [] and res["agree"] == res["n"] == 36
    assert res["extractor_id"] == N.EXTRACTOR_ID and res["extractor_sha256"] == N.source_sha256()


def test_정본_벡터의_전_사례에서_다섯_값이_기대와_같고_초안_사례를_그대로_품는다(tmp_path):
    vecs, sha = _canonical(tmp_path)
    assert sha == N.VECTOR_SHA256
    res = N.run_vectors(vecs)
    assert res["mismatches"] == [] and res["agree"] == res["n"] == len(vecs)
    by_id = {v["id"]: v for v in vecs}
    draft, _ = _draft()
    assert len(by_id) == len(vecs) > len(draft)
    assert all(by_id.get(v["id"]) == v for v in draft)                 # 초안의 사례는 바뀌지 않았다


def test_정본_자리를_찾는_순서와_없으면_실패(tmp_path, monkeypatch):
    """정규 배포 자리 → 지금 자리 → D 커밋의 blob. 자리에 있는 파일도 sha 가 다르면 거부하고, 셋 다 없으면 거부한다."""
    raw, where = N.canonical_vector_bytes()
    assert where in ("release", "tree", "commit")
    repo = tmp_path / "repo"                                            # git 저장소가 아니다 — 커밋 blob 을 읽을 수 없다
    (repo / N.RELEASE_VECTOR_PATH).parent.mkdir(parents=True)
    with pytest.raises(N.VectorRejected):
        N.canonical_vector_bytes(repo)
    (repo / N.RELEASE_VECTOR_PATH).write_bytes(raw)
    assert N.canonical_vector_bytes(repo) == (raw, "release")
    (repo / N.RELEASE_VECTOR_PATH).write_bytes(raw + b" ")
    with pytest.raises(N.VectorRejected):
        N.canonical_vector_bytes(repo)


@pytest.mark.parametrize("text", [None, 3, b'{"defects":[]}', ["{}"], {"text": "{}"}])
def test_문자열이_아닌_생성문은_검증_오류다(text):
    """계약 밖 입력 — 종료 사유가 맞아도 파서에 넘기지 않는다(D 와 같은 꼴, 지시 1022c 덧붙임 D)."""
    for stop in ("eos", "length"):
        assert N.extract(text, stop).as_dict() == {"state": "validation_error", "narrative": None, "front_text": None,
                                                   "stop_kept": None, "upstream_reason": None}


def test_벡터_대조는_값과_자료형을_함께_본다():
    """`0` · `1` 과 `False` · `True` 를 다르게 본다 — 기대가 0 이면 False 를 낸 추출기는 어긋난다(덧붙임 D)."""
    base = {"id": "T1", "text": J + "\n서술.", "gen_stop": "eos",
            "expect": {"state": "complete", "narrative": "서술.", "front_text": False, "stop_kept": "eos",
                       "upstream_reason": None}}
    assert N.run_vectors([base])["mismatches"] == []
    loose = {**base, "expect": {**base["expect"], "front_text": 0}}
    res = N.run_vectors([loose])
    assert res["agree"] == 0 and res["mismatches"][0]["fields"] == ["front_text"]


def test_벡터_파일의_바이트가_다르면_대조하지_않는다(tmp_path):
    p = tmp_path / "v.json"
    p.write_bytes((ROOT / N.DRAFT_VECTOR_PATH).read_bytes().replace(b"\n", b"\r\n"))     # 글 모드 전달 사고
    with pytest.raises(N.VectorRejected):
        N.load_vectors(p, sha256=N.DRAFT_VECTOR_SHA256)
    with pytest.raises(N.VectorRejected):
        N.load_vectors(tmp_path / "없음.json", sha256=N.DRAFT_VECTOR_SHA256)


def test_레코드_실패의_판정과_사유가_정본_파서와_같다(tmp_path):
    vecs = _draft()[0] + _canonical(tmp_path)[0]
    for v in vecs:
        if v.get("gen_stop") not in N.GEN_STOPS:
            continue
        got = N.extract(v["text"], v["gen_stop"])
        err = parse_generation(v["text"]).parse_error
        assert (got.state == "record_fail") == (err is not None), v["id"]
        assert got.upstream_reason == err, v["id"]


# ================================================================ 시험 6 — 합성 경계(늘 돈다)
@pytest.mark.parametrize("text,stop,want", [
    (J, "eos", ("missing", "", False, "eos", None)),
    (J, "length", ("missing", "", False, "length", None)),                    # JSON 직후 길이 종료 = 미기재(M3)
    (J + "\n", "length", ("missing", "", False, "length", None)),             # 줄바꿈만 남아도 미기재
    (J + "\n서술이 한도에서 끊", "length", ("truncated", "서술이 한도에서 끊", False, "length", None)),
    (J + "\n서술이 끝났다.", "eos", ("complete", "서술이 끝났다.", False, "eos", None)),
    ("{} 뒤 설명", "eos", ("record_fail", None, None, None, "schema_violation")),
    ('{"defects":[],"verdict":"판정', "length", ("record_fail", None, None, None, "truncated")),
    ("객체가 없다.", "eos", ("record_fail", None, None, None, "no_json")),
    ('{"defects":[{"iso_code":"2011","bbox_2d":[1,2]}],"verdict":"판정불가","cited_clauses":[]}\n상자 하나가 깨졌다.', "eos",
     ("complete", "상자 하나가 깨졌다.", False, "eos", None)),
])
def test_상태_순서와_경계(text, stop, want):
    assert tuple(N.extract(text, stop).as_dict().values()) == want


@pytest.mark.parametrize("stop", [None, "", "stop", "stop_undetermined", "EOS", 1, ["eos"]])
def test_종료_사유가_없거나_모르면_파싱보다_먼저_검증_오류다(stop):
    for text in (J, "객체가 없다.", J + "\n서술."):
        assert N.extract(text, stop).state == "validation_error"
        assert N.extract(text).state == "validation_error"                    # 칸 없음 = 인자 생략


@pytest.mark.parametrize("line,is_marker", [
    ("```", True), ("```json", True), ("````", True), ("```c++", True), ("```py-3_x", True), ("   ```json  ", True),
    ("``` json", False), ("```이 결함은 합격이다", False), ("```json 합격", False), ("```인라인```", False),
    ("~~~", False), ("``", False), ("서술 ```", False),
])
def test_펜스_표지_줄의_좁힌_문법(line, is_marker):
    assert N.is_fence_marker(line) is is_marker
    got = N.extract(J + "\n" + line + "\n뒤 서술.", "eos")
    assert got.narrative == ("뒤 서술." if is_marker else f"{line.strip()}\n뒤 서술.")


def test_앞_글과_뒤_글은_각각_벗긴_뒤_공백_하나로_잇고_안쪽은_바꾸지_않는다():
    got = N.extract("앞 글이다.\r\n\n" + J + "\n\n뒤  글\r\n둘째 줄.　", "eos")
    assert got.as_dict() == {"state": "complete", "narrative": "앞 글이다. 뒤  글\r\n둘째 줄.", "front_text": True,
                             "stop_kept": "eos", "upstream_reason": None}


# ================================================================ 시험 6 — 정답 타깃 자기 시험(학습 풀)
@pytest.mark.skipif(PAIRS is None, reason="학습 풀 페어가 없다 — 합성 경계 시험은 위에서 돈다")
def test_학습_풀의_새_타깃은_완결이고_추출문이_target_text_와_같다():
    from vlm.coords import CoordCfg, ImageGeom
    from vlm.pilot_vlm import build_target
    from vlm.uni_config import pairs_snapshot_problems

    assert pairs_snapshot_problems(PAIRS, PAIRS_DIGEST) == []          # 학습 코드와 같은 스냅샷 계약 대조
    cfg = CoordCfg(coord_space="ABS_ORIG")
    n = bad = 0
    with PAIRS.open(encoding="utf-8") as fh:
        for i, ln in enumerate(fh):
            r = json.loads(ln)
            if r.get("split") not in ("train", "val"):
                continue
            t = build_target(r, ImageGeom(orig_w=int(r["width_px"]), orig_h=int(r["height_px"])), cfg,
                             recipe="json_line_then_basis")
            line, _sep, basis = t.partition("\n")
            n += 1
            got = N.extract(t, "eos")
            if (got.state, got.narrative, got.front_text) != ("complete", r["target_text"], False):
                bad += 1
                continue
            if i % 25:
                continue
            cut = basis[: max(1, len(basis) // 2)]
            assert N.extract(line, "length").state == "missing", r["image_id"]
            assert N.extract(line + "\n", "length").state == "missing", r["image_id"]
            mid = N.extract(line + "\n" + cut, "length")
            assert (mid.state, mid.narrative) == ("truncated", cut.strip()), r["image_id"]
            assert N.extract(line[: len(line) // 2], "length").as_dict() == {
                "state": "record_fail", "narrative": None, "front_text": None, "stop_kept": None,
                "upstream_reason": "truncated"}, r["image_id"]
            assert N.extract(t).state == "validation_error"
    assert n == 49845 and bad == 0
