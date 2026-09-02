"""채점 폐기·범위 정책 — **다섯 칸이 같은 규칙으로 버린다.** 체크리스트 13 (80번 D7·D8).

어댑터는 "칸을 구분해도 되는 유일한 지점"이지만, 그건 **형식**의 차이를 흡수하라는
뜻이지 **정책**을 갈라도 된다는 뜻이 아니었다. 실제로 갈려 있었다.

| | 통합형(어댑터) | 분리형(추론) |
|---|---|---|
| 결함 항목 하나가 깨졌을 때 | 레코드 전체를 `defects=[]` 로 폐기 | 그 박스만 건너뛰고 나머지는 살림 |
| 채점 클래스 밖 코드 | 통과시킴 → `class_jaccard` 분모만 커짐 | nc=4 라 물리적으로 못 냄 |

같은 이미지가 칸에 따라 "전량 미검출" 또는 "일부 검출"이 됐다. 본실험은 이미지당 결함
최대 50개라 발현 확률이 파일럿보다 훨씬 크다.

**여기 상수 하나가 두 경로의 정책이다.** 갈라 쓰려면 이 파일을 고쳐야 하고, 고치면
대칭성 시험이 깨진다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

ITEM_LEVEL = "item"
"""결함 항목 하나가 깨지면 **그 항목만** 버린다. 레코드는 살린다."""

RECORD_LEVEL = "record"
"""결함 항목 하나가 깨지면 레코드 전체를 폐기한다. **쓰지 않는다** — 여기 남긴 이유는
정책이 두 개 있었다는 사실을 코드에 남기기 위해서다."""

DEFECT_ITEM_POLICY = ITEM_LEVEL
"""**다섯 칸 공통.** 분리형이 이미 이렇게 동작했고, 통합형을 여기 맞춘다.

레코드 폐기가 더 엄해 보이지만 실제로는 반대다 — 결함 20개 중 1개가 깨졌을 때
나머지 19개를 미검출로 계상하면 그 칸만 과소평가된다. 항목 폐기는 깨진 것만 버린다.
"""

OUT_OF_SCOPE_CODE_POLICY = "drop_and_count"
"""채점 클래스(4결함) 밖 ISO 코드는 **버리되 센다.**

버리는 이유: `score_detection` 은 `classes` 만 순회해 이 코드를 FP 로도 세지 않는데
`class_jaccard` 는 원시 집합의 합집합을 쓰므로 **분모만 커진다.** 분리형은 nc=4 라
이런 코드를 물리적으로 낼 수 없으니 벌점이 통합형에만 붙는다.

세는 이유: 채점 공간 밖 코드를 낸다는 것은 그 자체로 진단 정보다(환각 신호).
채점에서 빼되 `n_out_of_scope` 로 보고한다 — 조용히 버리면 그 사실이 사라진다.
"""

SCHEMA_FATAL_FIELDS = ("verdict", "cited_clauses", "defects")
"""레코드 수준에서 깨졌다고 보는 필드. 이것들이 무너지면 항목 폐기로 구제할 수 없다."""


@dataclass
class DefectFilterReport:
    """항목 필터 결과. **버린 것을 건수로 증명한다.**"""

    kept: list[dict] = field(default_factory=list)
    n_bad_item: int = 0
    """형식이 깨진 항목 수 (좌표 4개 아님, 퇴화 박스, 비유한값)."""
    n_unknown_code: int = 0
    """`label_map` 에 없는 코드. 스키마 밖이라 형식 오류로 센다."""
    n_out_of_scope: int = 0
    """`label_map` 에는 있으나 채점 4클래스 밖. 정책상 버리되 센다."""
    out_of_scope_codes: dict[str, int] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return self.n_bad_item + self.n_unknown_code + self.n_out_of_scope

    def as_dict(self) -> dict:
        return {
            "n_kept": len(self.kept),
            "n_dropped": self.n_dropped,
            "n_bad_item": self.n_bad_item,
            "n_unknown_code": self.n_unknown_code,
            "n_out_of_scope": self.n_out_of_scope,
            "out_of_scope_codes": dict(sorted(self.out_of_scope_codes.items())),
            "policy": DEFECT_ITEM_POLICY,
        }


def filter_defect_items(
    items: Iterable[dict],
    *,
    known_codes: Iterable[str],
    scoring_codes: Sequence[str],
) -> DefectFilterReport:
    """결함 항목 목록에 **공통 정책**을 적용한다. 두 계열이 같은 함수를 부른다.

    Args:
        items: `{"iso_code", "bbox_px", ...}` 형태의 원시 항목.
        known_codes: `label_map.yaml` 전체 코드.
        scoring_codes: 채점 대상 4클래스. 이 밖은 `OUT_OF_SCOPE_CODE_POLICY` 를 따른다.

    좌표를 고치지 않는다. 경계 이탈도 여기서 버리지 않는다 — 클리핑은 IoU 를 올리는
    방향으로만 작동하고, 이탈은 진단으로 따로 센다.
    """
    import math

    known = set(known_codes)
    scoring = set(scoring_codes)
    rep = DefectFilterReport()
    for d in items:
        code = str(d.get("iso_code", ""))
        if code not in known:
            rep.n_unknown_code += 1
            continue
        box = d.get("bbox_px")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            rep.n_bad_item += 1
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            rep.n_bad_item += 1
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
            rep.n_bad_item += 1
            continue
        if code not in scoring:
            rep.n_out_of_scope += 1
            rep.out_of_scope_codes[code] = rep.out_of_scope_codes.get(code, 0) + 1
            continue
        rep.kept.append({**d, "iso_code": code, "bbox_px": [x1, y1, x2, y2]})
    return rep
