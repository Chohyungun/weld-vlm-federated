"""D1(통합형 1.4 계약) 착수 **전의 기준선**을 고정한다. 07번 미니스펙 §12-4 · §14-2.

D1 은 "기존 것은 이름도 값도 건드리지 않는다"를 약속한다. 그 약속을 diff 가 아니라 시험으로 묶는다.

- **계약 파일의 바이트.** v3 산출물은 채점 코드의 파일별 해시를 싣고, 59번 독립 검산 실행기의 코드 드리프트 검사는
  `cells`·`adapters`·`schema` 를 파일 해시로만 본다. 새 기능은 전부 새 파일에 둔다.
- **이름과 값.** `UNI_TAGS`·`ALL_TAGS`·`UNIFIED_CELLS`·`Cell`·`SCHEMA_VERSION` 은 `score_cells.py`·
  `stratified_compare.py` 와 1.3 레코드 전부가 같이 쓴다. 바꾸면 파일 diff 는 0 인 채로 파일럿 재채점이 죽는다.
- **저장된 1.3 줄.** 되읽어 다시 쓴 바이트가 원본과 같아야 한다 — 선택 필드 하나만 더해도 null 키가 생겨 깨진다.

여기서 시험이 깨지면 **고칠 것은 시험이 아니라 변경**이다. 의도한 변경이면 미니스펙 개정과 검수가 먼저다.

고정값은 **이 브랜치의 바이트**를 가리킨다. 이 브랜치의 정리 커밋이 병합되기 전에 main 체크아웃에서 돌리면
`test_계약_파일의_바이트가_그대로다` 가 실패한다 — 값이 틀린 것이 아니라 대상 트리가 다른 것이다.
주석만 바뀌었다는 증명은 `tests/test_scrub_was_comment_only.py` 가 따로 선다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import get_args

import pytest

from evaluation import gates
from evaluation.adapters import AdaptReport, adapt_unified_generations
from evaluation.cells import ALL_TAGS, DET_TAGS, UNI_TAGS
from evaluation.provenance import SCORER_FILES
from evaluation.schema import SCHEMA_VERSION, UNIFIED_CELLS, Cell, PredictionRecord

REPO = Path(__file__).resolve().parents[1]

FROZEN_SHA256: dict[str, str] = {
    "evaluation/schema.py": "ae6cec81b9692cb68f2eca71d04200a3714904a26715f6eff5f1cf2a99b11642",
    "evaluation/adapters.py": "1145b7848f02dd7bb1db71c9d95296d50c0d94657840733990a66341e2ebacb0",
    "evaluation/cells.py": "9221dd4b47bec64c6e4ebedff5bb37a513c345582dee57d5ea18e55d23311aaf",
    "evaluation/policy.py": "7a86efed857d5f5c4874b33ea9fdae530f7a86001ab764997149ed85241d228f",
    "evaluation/gates.py": "4f0aed78ea175359be850338fd4d337e80917c8571426eb53dee6e8f02d8ca93",
    "evaluation/score.py": "1233d8e67dd7b16ffcb5a16d10da54cd1bd829620a968de31f3798e57817d5d6",
    "evaluation/discrimination.py": "ecc1e39478541acf958d48f770827bb4e55a2e1a32497e51c366743612d46c38",
    "evaluation/eval_set.py": "85ef783c62ae9c5a9afb8e9caede888fafa9a1d24a4e6e0cbbe0046829cd2b38",
    "evaluation/params.py": "07ff761be77d4ff98e8927284948e4752c3d79686ece2016edbf4f44f59ae948",
    "evaluation/prereg.py": "39b1060c0649656009466be047c9a2a9f7f3c16712f9ab39ef870fc54db2fa1a",
    "evaluation/recovery_ci.py": "1d2508e110e4cd2122281644dff904443035abb567d1248f7ccaee379ce193d5",
    "evaluation/metrics/localization.py": "a465ec7152238aea2f77e6949cef0677ab5107f9005822c44dd2cd07925ad12a",
    "evaluation/metrics/detection.py": "0bf7242f7084b30e45785268c91a442406eb471e5a01776ad03358196ccb072f",
    "evaluation/prediction.schema.json": "bf925c6cfa1e59e8e47eae3d717afa23f06ecb91125fea2ebb21d6c23e3b14a6",
    "scripts/probe/score_cells.py": "a5ad44d2ea53d1d300d4f0505ec2fa99c3b28162dce94b94f366177b438338f5",
    "scripts/probe/aggregate_seeds.py": "9cf76496660a3036033346975a19f0b00e65cdb88871dc6e60729ea5a5ebbb2d",
    "scripts/probe/recovery_bootstrap.py": "4e892a10c1a574485f1b807ae5620b5d6bb8db92e2398cf7bda99217b7ee768e",
    "scripts/probe/map50_independent.py": "3f1a6201477be80983300cbcfc37324a4ed8165ddc55eb276435bd4dcbc1537b",
    "evaluation/provenance.py": "3c7550a05995aa2ee22e9348641dd5c1aea63938325fe493530bac960e341f71",
}
"""줄끝을 LF 로 고친 바이트의 sha256 — `evaluation.provenance` 의 코드 지문과 같은 정의다(체크아웃의 줄끝 설정에
값이 흔들리지 않게). 2026-09-21 D1 착수 직전의 값이고, 같은 날 공개본 표현 정리(주석·docstring 만)로
여섯 파일을 다시 고정했다 — 동작·단언·설정값은 그대로다. **2026-10-01 에 열한 파일을 다시 박았다** — 역할 이름을 걷었고,
당시 값은 `BEFORE_ROLE_SCRUB` 에 병기했다. 주석 밖(산출 문자열)이 바뀐 파일은 `tests/test_scrub_was_comment_only.py` 의
`ROLE_SCRUB_DIVERGED` 가 사유와 함께 든다. **같은 날 `evaluation/provenance.py` 를 더했다**(검수 16번 I-3) — 채점 지문을 내는
파일인데 역할 이름 정리로 구문 트리 시험에서 빠져 걸린 고정이 0 이 됐다. 그 정리 직전의 값을 `BEFORE_ROLE_SCRUB` 에 병기했다."""

BEFORE_ROLE_SCRUB: dict[str, str] = {
    "evaluation/discrimination.py": "6cdb267258cf9c9ad1f10227c293eff19fc85609eefd0d60129f6431bfd7f9b0",
    "evaluation/gates.py": "c3ca14e8d77668caa814c79c5060e8f100f93c0da0f7ca4794fb5ae18f52411f",
    "evaluation/metrics/detection.py": "882e1dc644e2c261ac1062a8bf026a42d364bc33ab7f16506658d56e58d2f62e",
    "evaluation/params.py": "256fa89b22a5f4fce4ec395ac8f19e93b846612387fe6271a0354d453e0d76fd",
    "evaluation/prereg.py": "f9387506f3687fbef81fbec27fd04124fe7f104dc3826afbb424e288864840a4",
    "evaluation/recovery_ci.py": "e83e04886a8b5994b05ed1bb4ec47b52c696bd9959ee4e94a5edf47065f6ab68",
    "evaluation/score.py": "ebb02191d26c9276e30b17b57461c37a92c4dc207964e920fa1d1471dd6995d2",
    "scripts/probe/aggregate_seeds.py": "396a5c29fab315f9512f2411c481747adf1ee3b610ea3096e13636eb68f48178",
    "scripts/probe/map50_independent.py": "80d7eceb94d047de84b97bbec478ac289a1e6af50f3006f3260944c9b2663102",
    "scripts/probe/recovery_bootstrap.py": "67866f9b5468d75a14db246b93c0902fa473e5b3b88f00828afd27c44cf8785f",
    "scripts/probe/score_cells.py": "4aee80cb7585fcb0b5adf91db12b6bdf7d872723d0c9540654e30d0fcb847c60",
    "evaluation/provenance.py": "c1cf1c0d72556a7c75244fd6aa426156431e69a8b1e21284e37d75c41e6a31f0",
}
"""**당시 값** — 2026-10-01 역할 이름 정리 직전의 바이트 해시(위 표와 같은 정의). 덮지 않고 병기한다. v3 산출물의 파일별 해시와
맞대는 시험은 이 값을 쓴다 — 산출물이 싣는 파일별 해시는 그때의 기록이라 고치지 않는다."""


def at_d1(rel: str) -> str:
    """D1 착수 직전(그리고 2026-09-21 표현 정리 뒤)의 고정값 — 2026-10-01 정리 전의 바이트."""
    return BEFORE_ROLE_SCRUB.get(rel, FROZEN_SHA256[rel])

SAME_AS_V3 = (
    "evaluation/adapters.py",
    "evaluation/cells.py",
    "evaluation/discrimination.py",
    "evaluation/gates.py",
    "evaluation/metrics/detection.py",
    "evaluation/metrics/localization.py",
    "evaluation/params.py",
    "evaluation/policy.py",
    "evaluation/prereg.py",
    "evaluation/schema.py",
    "evaluation/score.py",
)
"""v3 를 채점한 코드와 **D1 착수 직전(`at_d1`)에 같았던** 파일. 계약 #4 의 세 파일(`schema`·`adapters`·`cells`)이 여기 있다.
2026-10-01 역할 이름 정리로 이 가운데 여섯(`discrimination` · `gates` · `metrics/detection` · `params` · `prereg` · `score`)은
바이트가 달라졌다 — v3 와 맞대는 시험은 당시 값(`at_d1`)을 쓴다. "지금도 같다" 고 적었던 것을 고쳤다(검수 16번 M-9).
eval_set.py 는 2026-09-21 공개본 표현 정리로 주석 한 줄이 달라져 뺐다(동작 불변).
나머지(recovery_ci.py · recovery_bootstrap.py · score_cells.py)는 v3 뒤의 검수 후속(덮어쓰기 차단·결속 보강)으로 이미 달라졌고,
위 고정값은 D1 착수 직전의 값이다."""

REGISTERED_GATES = frozenset({
    "content_free_gate", "coord_space_contract", "macro_ap_baseline_paired", "no_cloud_logging",
    "p9_source_separation", "prereg_constants_reproduced", "recovery_denominator", "required_tags",
    "scoring_population", "stratified_scoring", "sweep_curve_recorded",
})

ADAPT_REPORT_KEYS = frozenset({
    "adapter_parse_failures", "discard_policy", "n_bad_items_dropped", "n_boxes", "n_boxes_out_of_bounds",
    "n_images_with_citation", "n_lines", "n_out_of_scope_dropped", "n_records", "n_unknown_code_dropped",
    "out_of_scope_codes", "upstream_parse_failures",
})


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.parametrize("rel", sorted(FROZEN_SHA256))
def test_계약_파일의_바이트가_그대로다(rel: str) -> None:
    assert _lf_sha256(REPO / rel) == FROZEN_SHA256[rel], (
        f"{rel} 이 바뀌었다. D1 은 새 기능을 새 파일에 둔다(07번 §14-2 의 1). "
        "의도한 변경이면 미니스펙 개정과 게이트가 먼저다"
    )


def test_고정값이_v3_산출물의_파일별_해시와_같다() -> None:
    """고정값이 '지금 트리'가 아니라 **v3 를 채점한 코드**를 가리킨다는 것을 공유 산출물로 확인한다."""
    art = REPO / "outputs/main_d/seed1/score_cells_v3.json"
    if not art.exists():
        pytest.skip(f"공유 산출물 없음: {art.relative_to(REPO)} — 대형 자산이 있는 체크아웃에서 돈다")
    files = json.loads(art.read_text(encoding="utf-8"))["scorer_code"]["files"]
    assert {rel: files[rel] for rel in SAME_AS_V3} == {rel: at_d1(rel) for rel in SAME_AS_V3}


def test_게이트_레지스트리는_열한_개_그대로다() -> None:
    """카나리아를 전역 레지스트리에 더하면 사전실험 재채점의 게이트 수와 종료 코드가 바뀐다."""
    assert frozenset(gates.REGISTRY) == REGISTERED_GATES


def test_1_3_계약의_이름과_값이_그대로다() -> None:
    assert SCHEMA_VERSION == "1.3"
    assert get_args(Cell) == ("uni_central", "uni_fed", "sep_local", "sep_central", "sep_fed")
    assert UNIFIED_CELLS == ("uni_central", "uni_fed")
    assert UNI_TAGS == ("uni_central", "uni_fed")
    assert len(DET_TAGS) == 5
    assert ALL_TAGS == DET_TAGS + UNI_TAGS


def test_어댑터_보고의_키가_그대로다() -> None:
    """`AdaptReport` 는 검출 export 와 파일럿 통합형이 같이 쓴다. 키가 늘면 그쪽 산출물이 달라진다."""
    assert frozenset(AdaptReport().as_dict()) == ADAPT_REPORT_KEYS


PENDING_SCORER_FILES: tuple[str, ...] = ()
"""목록에 먼저 든 경로. 2026-10-01 재고정 때 진입점(`scripts/probe/score_unified.py`)이 여기 있었고, 같은 날 진입점 구현(과제 3-3)이
파일을 만들어 뺐다. 다시 선등록하면 여기에 두고, 파일이 생기면 뺀다(아래 시험이 강제한다)."""


def test_코드_지문의_파일_목록이_전부_실재한다() -> None:
    """`scorer_source_files` 는 없는 경로를 조용히 건너뛴다. 오타와 선등록이 아무 데서도 안 걸린다.

    예외는 `PENDING_SCORER_FILES` 하나다 — 그 파일이 생기면 예외 목록에 남아 있는 것 자체가 실패다."""
    missing = [rel for rel in SCORER_FILES if not (REPO / rel).exists()]
    assert set(missing) <= set(PENDING_SCORER_FILES), f"SCORER_FILES 에 없는 경로: {missing}"
    grown = [rel for rel in PENDING_SCORER_FILES if (REPO / rel).exists()]
    assert not grown, f"이제 있는 파일이다 — PENDING_SCORER_FILES 에서 뺀다: {grown}"


def test_채점_지문의_목록은_당시_값을_병기하고_진입점만_더했다() -> None:
    """2026-10-01 재고정. 사전실험 산출물이 지문을 낸 목록(`SCORER_FILES_AT_V3`)은 그대로 두고 진입점 하나만 뒤에 더했다."""
    from evaluation.provenance import SCORER_FILES_AT_V3

    assert SCORER_FILES == SCORER_FILES_AT_V3 + ("scripts/probe/score_unified.py",)
    assert SCORER_FILES_AT_V3 == (                    # 당시 목록 전체 — 길이와 첫 항목만으로는 덜 묶인다(검수 16번 M-9)
        "scripts/probe/score_cells.py", "scripts/probe/adapt_main_detections.py",
        "scripts/probe/content_free_baselines.py", "data/label_map.py", "data/id_strata.py",
        "data/manifest_io.py", "detection/serialize.py", "scripts/probe/recovery_bootstrap.py",
    )


def test_v3_산출물의_지문은_당시_목록으로_낸_것이다() -> None:
    """당시 값이 옳게 병기됐는지를 공유 산출물로 확인한다 — v3 가 해시한 트리 밖 파일은 옛 목록 안에 있다."""
    from evaluation.provenance import SCORER_FILES_AT_V3

    art = REPO / "outputs/main_d/seed1/score_cells_v3.json"
    if not art.exists():
        pytest.skip(f"공유 산출물 없음: {art.relative_to(REPO)} — 대형 자산이 있는 체크아웃에서 돈다")
    files = json.loads(art.read_text(encoding="utf-8"))["scorer_code"]["files"]
    outside = {rel for rel in files if not rel.startswith("evaluation/")}
    assert outside <= set(SCORER_FILES_AT_V3)
    assert "scripts/probe/score_unified.py" not in files


def test_당시_값과_지금_값은_정리한_열두_파일에서만_다르다() -> None:
    """정리한 열한 파일과 같은 날 고정에 더한 `provenance.py`(검수 16번 I-3). 당시 값이 정리 직전의 바이트인지는
    `tests/test_scrub_was_comment_only.py` 가 커밋된 blob 에서 다시 잰다."""
    assert set(BEFORE_ROLE_SCRUB) <= set(FROZEN_SHA256)
    assert len(BEFORE_ROLE_SCRUB) == 12 and "evaluation/provenance.py" in BEFORE_ROLE_SCRUB
    assert all(BEFORE_ROLE_SCRUB[rel] != FROZEN_SHA256[rel] for rel in BEFORE_ROLE_SCRUB)


# --------------------------------------------------------------------------------------
# 저장된 1.3 줄의 바이트 되읽기 — 공유 자산이 있을 때만 돈다
# --------------------------------------------------------------------------------------

STORED_V13 = (
    *(f"outputs/pilot_d/{tag}_s20260828.jsonl" for tag in (
        "uni_central", "uni_fed", "sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")),
    "outputs/main_d/seed1/sep_fed_s20260828.jsonl",
    "outputs/main_d/seed2/sep_local_C3_s20260829.jsonl",
    "outputs/main_d/seed3/sep_central_s20260830.jsonl",
)


def _lf_lines(path: Path) -> list[bytes]:
    """바이트를 LF 로만 나눈다. `str.splitlines()` 는 U+2028·U+0085 에서도 쪼갠다."""
    raw = path.read_bytes()
    assert b"\r" not in raw, f"{path.name}: CR 바이트가 있다"
    assert raw.endswith(b"\n"), f"{path.name}: 끝 개행이 없다"
    return raw[:-1].split(b"\n")


@pytest.mark.parametrize("rel", STORED_V13)
def test_저장된_1_3_줄을_되읽어_다시_쓴_바이트가_원본과_같다(rel: str) -> None:
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"공유 산출물 없음: {rel}")
    lines = _lf_lines(path)
    assert lines, f"{rel}: 빈 파일"
    for n, raw in enumerate(lines, 1):
        rec = PredictionRecord.model_validate_json(raw)
        assert type(rec) is PredictionRecord
        assert rec.model_dump_json().encode("utf-8") == raw, f"{rel} {n}번째 줄이 되읽기에서 달라졌다"


@pytest.mark.parametrize("cell", ("uni_central", "uni_fed"))
def test_파일럿_통합형을_옛_어댑터로_다시_어댑트한_바이트가_저장본과_같다(cell: str) -> None:
    """옛 결합 규칙(v1) 어댑터의 **거동**을 실물로 고정한다. 바이트 되읽기는 직렬화만 잡는다."""
    import argparse

    from data.label_map import load_label_map
    from evaluation.cells import load_population
    from evaluation.params import add_common_args, params_from_args

    src = REPO / f"outputs/pilot_c/predictions/{cell}.generations.jsonl"
    stored = REPO / f"outputs/pilot_d/{cell}_s20260828.jsonl"
    if not (src.exists() and stored.exists()):
        pytest.skip("파일럿 통합형 자산 없음 — 대형 자산이 있는 체크아웃에서 돈다")
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    params = params_from_args(ap.parse_args([]))
    if not (REPO / params.snapshot / "manifest.csv").exists():
        pytest.skip(f"파일럿 동결본 없음: {params.snapshot}")
    pop = load_population(params)
    rep = adapt_unified_generations(
        src.read_text(encoding="utf-8").splitlines(),
        cell=cell, seed=params.seed, known_iso_codes=set(load_label_map().iso_codes()),
        scoring_iso_codes=pop.classes, image_size=pop.sizes,
    )
    fresh = [r.model_dump_json().encode("utf-8") for r in rep.records]
    assert fresh == _lf_lines(stored)
