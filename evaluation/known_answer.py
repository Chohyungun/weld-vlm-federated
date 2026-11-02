"""M1 기지답 — 저장소에 커밋한 합성 박스 픽스처를 공통 채점기(`build_counts`)로 센 계수표와 그 sha256(등록 `canary.known_answer_sha256`).

픽스처(`FIXTURE_REL`)는 손으로 셀 수 있는 장 아홉이다 — 맞춤 · IoU 정확히 1/2(경계 재판정) · 1/2 미만 · 한 정답에 예측 둘 · 클래스
어긋남 · 정상 이미지의 예측 · 빈 정상 · 탐욕 배정이면 하나만 맞는 교차(최대 짝 수는 둘) · 채점 클래스 밖 코드. 손으로 센 기대 계수를
함께 들고, 시험이 계수표와 맞댄다. 본채점(엄격)은 **영수증 커밋의** 픽스처로 계수표를 다시 내 등록 값과 맞대고, 다르면
`KNOWN_ANSWER_MISMATCH` 로 멈춘다 — 매칭 · 경계 규약(`matching_rule_id`)이 바뀌면 이 값이 바뀐다(07번 §12-5 · 597행).

| 계수표 바이트 | 값 |
|---|---|
| 꼴 | `{"kind", "classes", "image_ids", "tp", "fp", "fn", "boundary", "pred_per_image"}` 의 JSON, 키 정렬 · 구분자 `,` `:` · UTF-8 · 끝 개행 없음 |
| 순서 | `image_ids` 는 `build_counts` 의 순서(정렬), 칸은 `classes` 순서 |
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

from evaluation.metrics.box_f1 import CountTable, build_counts, score_m1

FIXTURE_REL = "evaluation/fixtures/m1_known_answer_v1.json"
FIXTURE_KIND = "m1_known_answer"
TABLE_KIND = "m1_known_answer_table"


class KnownAnswerError(ValueError):
    """기지답 픽스처의 꼴이 틀렸다."""


def read_fixture(raw: bytes) -> dict:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise KnownAnswerError("기지답 픽스처를 읽지 못한다") from None
    if not isinstance(obj, dict) or obj.get("kind") != FIXTURE_KIND or not isinstance(obj.get("classes"), list) \
            or not isinstance(obj.get("images"), list) or not obj["images"]:
        raise KnownAnswerError("기지답 픽스처의 꼴이 아니다")
    ids = [im.get("image_id") for im in obj["images"]]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise KnownAnswerError("기지답 픽스처의 image_id 가 비었거나 겹친다")
    return obj


def count_table(fixture: dict) -> CountTable:
    """모델과 **같은 채점기**로 센다(불변조건 3-7). 정상 이미지도 빈 정답으로 모집단에 든다."""
    gold = {im["image_id"]: [(str(c), [float(x) for x in b]) for c, b in im["gold"]] for im in fixture["images"]}
    pred = {im["image_id"]: [(str(c), [float(x) for x in b]) for c, b in im["pred"]] for im in fixture["images"]}
    return build_counts(pred, gold, [str(c) for c in fixture["classes"]])


def table_bytes(table: CountTable) -> bytes:
    obj = {"kind": TABLE_KIND, "classes": list(table.classes), "image_ids": list(table.image_ids),
           "tp": table.tp.astype(int).tolist(), "fp": table.fp.astype(int).tolist(), "fn": table.fn.astype(int).tolist(),
           "boundary": table.boundary.astype(int).tolist(), "pred_per_image": table.pred_per_image.astype(int).tolist()}
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def known_answer_sha256(raw: bytes) -> str:
    """픽스처 바이트 → 계수표 바이트의 sha256 — 등록 `canary.known_answer_sha256` 의 값."""
    return hashlib.sha256(table_bytes(count_table(read_fixture(raw)))).hexdigest()


def expected_mismatches(fixture: dict, table: CountTable) -> list[str]:
    """픽스처가 손으로 센 기대(계수 · 경계 · 예측 수 · 매크로 · micro)와 계수표가 다른 자리. 비면 같다."""
    exp, out = fixture.get("expected") or {}, []
    idx = table.index_of()
    for r, iid in enumerate(table.image_ids):
        for c, code in enumerate(table.classes):
            want = (exp.get("counts", {}).get(iid, {}).get(code) or [0, 0, 0])
            got = [int(table.tp[r, c]), int(table.fp[r, c]), int(table.fn[r, c])]
            if got != list(want):
                out.append(f"{iid}·{code}: 계수 {got} ≠ 기대 {want}")
            b_want = exp.get("boundary", {}).get(iid, {}).get(code, 0)
            if int(table.boundary[r, c]) != b_want:
                out.append(f"{iid}·{code}: 경계쌍 {int(table.boundary[r, c])} ≠ 기대 {b_want}")
        if int(table.pred_per_image[idx[iid]]) != exp.get("pred_per_image", {}).get(iid, -1):
            out.append(f"{iid}: 예측 수 {int(table.pred_per_image[idx[iid]])} ≠ 기대")
    rep = score_m1(table)
    for name, got in (("m1_macro", rep.macro_f1), ("micro", rep.micro_f1)):
        if got is None or abs(got - float(Fraction(exp.get(name, "0")))) > 1e-12:
            out.append(f"{name}: {got} ≠ 기대 {exp.get(name)}")
    return out
