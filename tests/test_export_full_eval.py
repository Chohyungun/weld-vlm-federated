"""본실험 모델 묶음의 생성 전 전량 관문 — `vlm.export_preflight.check_generation_list`(지시 20261007e 의 1).

검토 12번(검토 기록 12번(전량평가 등록계약 미니스펙))은 전량 계약을 어기면 **예측 생성 · 본채점 전에** 실패하라고 한다.
채점 쪽은 평가 쪽이 `_read_lists` 에 걸었다. 이 시험은 생성 쪽 — export 가 이미지를 열고 생성기를 올리기 전 — 의 같은 관문을 본다.
합성 매니페스트만 쓴다(id 는 `synth:` — 실물 꼴이 아니다). 실물 평가 자산을 열지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.entry_gate import ManifestMeta
from evaluation.eval_list import canonical_bytes, validate_eval_list
from vlm import export_preflight as XP
from vlm.export_writer import ExportRefused

EVAL = tuple(f"synth:e{i:02d}" for i in range(5))
VAL = tuple(f"synth:v{i:02d}" for i in range(3))
TRAIN = ("synth:t00",)


def meta(eval_ids=EVAL, val_ids=VAL) -> ManifestMeta:
    split = {**{i: "eval" for i in eval_ids}, **{i: "val" for i in val_ids}, **{i: "train" for i in TRAIN}}
    return ManifestMeta(split_of=MappingProxyType(split), sha256_of=MappingProxyType({i: "0" * 64 for i in split}),
                        group_of=MappingProxyType({i: f"g_{i}" for i in split}))


def elist(ids):
    return validate_eval_list(canonical_bytes(list(ids)))


def reg_for(gen_ids=None, echo_ids=VAL):
    """등록의 생성 쪽 — 목록을 주면 그 두 해시를, `None` 이면 미등록(해시 없음)."""
    g = elist(gen_ids) if gen_ids is not None else None
    e = elist(echo_ids)
    return SimpleNamespace(eval_list_file_sha256=g.file_sha256 if g else None, eval_list_set_sha256=g.set_sha256 if g else None,
                           echo_list_file_sha256=e.file_sha256, echo_list_set_sha256=e.set_sha256)


def check(file_ids, reg, *, purpose="main", mode="model", m=None):
    XP.check_generation_list(elist(file_ids), reg, mode=mode, purpose=purpose, meta_of=lambda: m or meta())


def code_of(fn) -> tuple[str, str]:
    with pytest.raises(ExportRefused) as e:
        fn()
    return e.value.code, str(e.value)


# ---------------------------------------------------------------- 검토 12 의 경우 — 생성 쪽

def test_정상_전량은_지난다() -> None:
    check(EVAL, reg_for(EVAL))


def test_한_장_빠진_목록은_등록과_맞아도_생성_전에_거부한다() -> None:
    code, msg = code_of(lambda: check(EVAL[:-1], reg_for(EVAL[:-1])))
    assert code == "main_list_not_full_eval" and "빠짐 1개" in msg
    assert not any(i in msg for i in EVAL), "사유에 평가 id 를 싣지 않는다 — 건수만"


def test_같은_집합의_다른_순서는_등록한_순서가_아니면_거부한다() -> None:
    code, _ = code_of(lambda: check(EVAL[::-1], reg_for(EVAL)))
    assert code == "list_binding"


def test_미등록_목록은_거부한다() -> None:
    code, _ = code_of(lambda: check(EVAL, reg_for(None)))
    assert code == "list_binding"


def test_등록_지문이_다르면_거부한다() -> None:
    code, _ = code_of(lambda: check(EVAL, reg_for(EVAL[:-1])))
    assert code == "list_binding"


def test_평가_행_밖의_장이_섞이면_거부한다() -> None:
    mixed = EVAL[:-1] + (VAL[0],)
    code, _ = code_of(lambda: check(mixed, reg_for(mixed)))
    assert code == "entry_LIST_SPLIT"


def test_기대_집합은_매니페스트에서_온다_목록이_전부_평가여도_모자라면_거부한다() -> None:
    """목록의 모든 id 가 평가 행이라 분할 검사는 지나지만, 매니페스트의 평가 행이 더 많다 — 기대 집합을 목록에서 만들면 지나간다."""
    big = meta(eval_ids=EVAL + ("synth:e99",))
    code, msg = code_of(lambda: check(EVAL, reg_for(EVAL), m=big))
    assert code == "main_list_not_full_eval" and "빠짐 1개" in msg and "평가 행 6개" in msg


def test_등록이_먼저_거부하면_매니페스트를_읽지_않는다() -> None:
    reads = []
    with pytest.raises(ExportRefused) as e:
        XP.check_generation_list(elist(EVAL), reg_for(None), mode="model", purpose="main", meta_of=lambda: reads.append(1))
    assert e.value.code == "list_binding" and reads == []


# ---------------------------------------------------------------- 합법적 부분집합은 그대로

@pytest.mark.parametrize("purpose", ["rehearsal", "frame_diag"])
def test_리허설과_진단의_부분집합은_받는다(purpose) -> None:
    sub = VAL[:-1]
    check(sub, reg_for(sub), purpose=purpose)


def test_본실험의_에코_묶음은_전량_관문을_거치지_않는다() -> None:
    """에코는 목적과 상관없이 val 에코 목록이다(§31-3) — 전량 eval 대상이 아니다."""
    check(VAL, reg_for(EVAL, echo_ids=VAL), mode="echo")


# ---------------------------------------------------------------- 실제 크기

@pytest.mark.parametrize(("n", "ok"), [(12461, True), (12460, False)])
def test_실제_크기의_전량_계약(n, ok) -> None:
    full = tuple(f"synth:e{i:05d}" for i in range(12461))
    ids = full[:n]
    fn = (lambda: check(ids, reg_for(ids), m=meta(eval_ids=full)))
    if ok:
        fn()
    else:
        code, msg = code_of(fn)
        assert code == "main_list_not_full_eval" and "목록 12460개 · 평가 행 12461개" in msg


# ---------------------------------------------------------------- 배선 — export 가 이미지 · 생성기 전에 이 관문을 부른다

# ---------------------------------------------------------------- 평가 쪽 시험 벡터 — 두 구현이 같은 판정을 낸다(지시 20261008d 의 1)

VECTORS = ROOT / "evaluation" / "full_eval_vectors.json"
CODE = {"MAIN_LIST_NOT_FULL_EVAL": "main_list_not_full_eval"}      # 평가 쪽 사유 코드 → 생성 쪽 사유 코드


def _vectors() -> dict:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _vector_meta(v: dict) -> ManifestMeta:
    split = {**{i: "eval" for i in v["eval_ids"]}, **{i: "val" for i in v["other_ids"]}}
    return ManifestMeta(split_of=MappingProxyType(split), sha256_of=MappingProxyType({i: "0" * 64 for i in split}),
                        group_of=MappingProxyType({i: f"g_{i}" for i in split}))


EXPORT_CASES = [c for c in _vectors()["cases"] if c["export_side"]]


def test_벡터_파일의_생성_쪽_경우는_목록이_하나다() -> None:
    """생성 쪽 경우는 채점 목록이 생성 목록과 같다 — 생성 쪽에는 목록이 하나뿐이라 그 경우만 맞댈 수 있다(벡터 파일의 about)."""
    assert len(EXPORT_CASES) >= 4
    assert all(c["pop"] == c["gen"] for c in EXPORT_CASES)
    assert {c["expect"] for c in EXPORT_CASES} <= {"pass", *CODE}


@pytest.mark.parametrize("case", EXPORT_CASES, ids=[c["name"] for c in EXPORT_CASES])
def test_생성_쪽과_채점_쪽_관문이_같은_시험_벡터에서_같은_판정을_낸다(case) -> None:
    from evaluation.entry_gate import EntryRejected, check_main_full_eval

    v = _vectors()
    m, el = _vector_meta(v), elist(case["gen"])
    if case["expect"] == "pass":
        XP.check_full_eval_set(el, m)
        check_main_full_eval(el, el, m)
        return
    with pytest.raises(ExportRefused) as mine:
        XP.check_full_eval_set(el, m)
    assert mine.value.code == CODE[case["expect"]]
    with pytest.raises(EntryRejected) as theirs:
        check_main_full_eval(el, el, m)
    assert case["expect"] in str(theirs.value)


def test_진입점의_순서에서는_분할_검사가_먼저_걸릴_수_있다() -> None:
    """벡터의 판정은 관문 함수만의 것이다 — 생성 쪽 목록 검사는 등록 대조 → 분할 → 전량 순이라 평가 행 밖의 id 는 분할 검사가 먼저 잡는다."""
    v = _vectors()
    case = next(c for c in EXPORT_CASES if c["expect"] != "pass" and set(c["gen"]) - set(v["eval_ids"]))
    code, _ = code_of(lambda: check(case["gen"], reg_for(case["gen"]), m=_vector_meta(v)))
    assert code == "entry_LIST_SPLIT"


def test_export_는_이미지와_생성기보다_먼저_목록_관문을_부른다(tmp_path, monkeypatch) -> None:
    from tests import uni_export_world as W

    w = W.build(tmp_path)
    seen = []

    def spy(el, g, *, mode, purpose, meta_of):
        seen.append((mode, purpose, el.file_sha256))
        raise ExportRefused("spy", "목록 관문에서 멈춘다")

    monkeypatch.setattr(XP, "check_generation_list", spy)
    loader = W.Loader()
    with pytest.raises(ExportRefused) as e:
        W.run(w, loader=loader)
    assert e.value.code == "spy" and loader.calls == 0
    assert seen == [("model", "rehearsal", validate_eval_list(w.lists["model"].read_bytes()).file_sha256)]
