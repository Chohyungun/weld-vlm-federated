"""D4 페어 2 판(부분 B)의 정답 서술 — 문장 계약(C 08 2 판 §4, D 07 의 T5)을 글자 그대로 조립한다.

1 판 서술(`make_pairs_pilot.defect_target_text`)과 상수는 바꾸지 않는다 — 1 판 페어 · 게이트 · 회귀가 그대로 선다. 2 판은 이 모듈의
상수와 함수를 쓴다. 골격(JSON 쪽)은 1 판과 같은 함수로 만든다(T4 — `make_pairs_pilot.assemble_record`).

- **바뀐 것** — 대응 문구(C2), 값 한계 조항의 측정량 머리(C3a), 영상 수준 조건 문장 넷(C5 합계 조건 · C6 혼재 · C7 종별 구별 곤란 ·
  C8 전제 미확인), 미대응 두 꼴(C9 · C10 의 '대응'), 불허 맺음(C12 — 09 §6 Q7). 측정량은 기준 목록 머리에 한 번이고 띠는 1 판
  그대로다(Q3).
- **그대로인 것** — 관찰(C1), 불허 조항의 기준 목록 머리(C3b), 미표현 단서(C4), 치수 맺음(C11), 정상(C13). 1 판 상수를 그대로 쓴다.
- **차례** — 관찰 → 코드마다(첫 등장 순) C2 · C3 · C4 → 영상마다 한 번씩 C5 · C6 · C7 · C8 → C9 또는 C10 → C11 → C12.
- **붙는 규칙(§4 표)** — C5: 인용한 조항 가운데 `군집간격` 을 든 조항이 있으면 한 번(둘 이상이면 멈춘다). C6: 인용한 코드의 종별이
  `혼재:X` 짝을 둘 다 품으면 한 번. C7: 인용한 코드의 행이 `구별곤란:X` 를 들고 그 재질에 X 종별 행이 있으면 한 번. C8: 인용한 조항이
  `무등급전제` 를 들면 한 번.

게이트(`corpus.validate.pair_criteria_gate`)와 독립 대조기(`scripts/probe/pairs_independent.py`)는 이 모듈을 import 하지 않고 같은 문장을
§4 표에서 따로 옮긴다 — 두 곳이 어긋나면 폐기 · 불일치로 드러난다.

순수 함수만 둔다(I/O · 전역 상태 · 난수 없음).
"""

from __future__ import annotations

from corpus.generate import make_pairs_pilot as P

#: C2 — 대응 문구. 주 조항이 있는 코드마다, 코드 덩이의 첫 문장이다(기준 목록의 앵커).
CORRESPOND = "{name}에 현 허용치 표에서 대응되는 조항은 {cid} 이다."
#: C3a 의 머리 — 값 한계 조항(행의 `measure` 가 있음). 띠 목록과 끝 마침표는 1 판 그대로 잇는다.
CRIT_HEAD_MEASURE = "이 조항의 측정량은 {measure}이고, 기준은 다음과 같다. "
#: C3b 의 머리 — 불허 조항(측정량 빈칸). 1 판과 같다.
CRIT_HEAD = "이 조항의 기준은 다음과 같다. "
#: C5 — 합계 조건.
AGGREGATION = "합계길이는 군집이면 간격을 포함해 잰다는 조건이 있고, 이 자료는 그 간격을 재지 않는다."
#: C6 — 혼재.
MIXTURE = "종별이 섞이면 함께 견주는 결합 규칙이 있고, 이 자료는 그 결합을 계산하지 않는다."
#: C7 — 종별 구별 곤란.
DISTINCTION = "종별을 가리기 어려우면 두 종별의 기준으로 각각 평가한다는 조건이 있고, 이 자료는 그 구별을 확인하지 않는다."
#: C8 — 전제 미확인. 판정어(검사기 닫힌 목록의 판정어 갈래 · 생성 쪽 판정 함의 목록)를 쓰지 않는다(T5).
PREMISE = "이 기준은 품질등급이 지정되지 않은 경우의 기준이고, 이 자료는 등급 지정 여부를 확인하지 않는다."
#: C9 — 재질 미대응.
NO_CLAUSE_MATERIAL = "이 재질에 대응되는 조항이 현 허용치 표에 없어 대응 조항을 특정하지 않는다."
#: C10 의 꼬리 — 앞에 `{이름}(ISO 6520-1 코드 {코드})[, …]` 를 붙인다.
NO_CLAUSE_CODE_TAIL = "에 대응되는 조항이 현 허용치 표에 없어 그 결함의 대응 조항을 특정하지 않는다."
#: C12 — 불허 맺음(09 §6 Q7 · 221 M-03). '치수 정보가 필요 없는 기준이지만 실제 적용 전제는 확인하지 않는다' 는 뜻이다.
CLOSE_SIZE_INDEPENDENT = ("크기와 무관한 기준은 치수 정보가 필요 없지만, 이 자료는 그 기준의 실제 적용 전제를 확인하지 않고"
                          " 판정 축을 조항 검색과 기준 서술로 한정해 합부 판정을 싣지 않는다.")
#: 그대로인 문장(C1 · C4 · C11 · C13) — 1 판 상수 그대로다.
NOTE_UNEXPRESSED = P.NOTE_UNEXPRESSED
CLOSE_DIMENSIONAL = P.CLOSE_DIMENSIONAL
NORMAL_TEXT = P.NORMAL_TEXT

#: 조건 토큰의 어휘 — 표의 단일 소스는 `corpus.rules.schema` 다(같은 글자인지는 시험이 맞댄다).
_CLUSTER, _PREMISE, _MIXTURE, _DISTINCTION = "군집간격", "무등급전제", "혼재", "구별곤란"


def _pairs(conditions, kind: str) -> set[str]:
    """`kind:X` 토큰의 X 들."""
    out = set()
    for t in conditions:
        k, sep, cls = t.partition(":")
        if sep and k == kind:
            out.add(cls)
    return out


def augment_basis(base: dict[str, dict], table, material: str) -> dict[str, dict]:
    """1 판 근거(`make_pairs_pilot.clause_basis`)의 코드마다 2 판 칸(측정량 · 집계 · 종별 · 조건)을 더한 **새 사전**을 낸다.

    칸은 그 코드의 행(같은 거름 — active · RT 또는 ALL · 재질이 그 재질 또는 ALL)에서 읽는다. 같은 조항의 행은 네 칸이 같다(V13) —
    다르면 멈춘다. 주 조항은 `limit_eval.derive_clause_roles` 의 식과 같아야 한다(T1 — 생성기 · 게이트 · 독립 대조기 · 검사기가 한
    식을 쓴다). 표가 2 판이 아니면 멈춘다 — 1 판 표에는 칸이 없어 조건 문장이 조용히 빠진다.
    """
    from corpus.rules.clause_text import val
    from corpus.rules.limit_eval import derive_clause_roles

    if getattr(table, "edition", "v1") != "v2":
        raise SystemExit(f"2 판 서술은 2 판 표로만 짓는다 — 표의 판 {getattr(table, 'edition', 'v1')!r}")
    roles = derive_clause_roles(table)
    out: dict[str, dict] = {}
    for code, ent in base.items():
        rows = [r for r in table.rows
                if val(r.scope) == "active" and val(r.inspection_method) in ("RT", "ALL")
                and val(r.material) in (material, "ALL") and r.defect_code == code]
        vals = {(val(r.measure) or None, val(r.aggregation) or None, val(r.defect_class) or None, tuple(r.conditions))
                for r in rows}
        if len(vals) != 1:
            raise SystemExit(f"({material}, {code}) 의 행들이 2 판 칸의 값이 다르다: {sorted(map(str, vals))}")
        main = (roles.get((material, code)) or {}).get("main")
        if main != ent["clause_id"]:
            raise SystemExit(f"({material}, {code}) 의 조항 {ent['clause_id']} 이 주 조항의 식(derive_clause_roles) {main} 과 다르다")
        measure, aggregation, defect_class, conditions = vals.pop()
        out[code] = {**ent, "measure": measure, "aggregation": aggregation, "defect_class": defect_class,
                     "conditions": conditions}
    return out


def defect_target_text(defects: list[dict], names: dict[str, str], basis: dict[str, dict]) -> str:
    """2 판 결함 페어의 정답 서술. 합부는 말하지 않는다(`verdict=null`). `basis` 는 `augment_basis` 의 출력이다."""
    head, codes = P.observation_line(defects, names)
    lines = [head]
    missing = [c for c in codes if c not in basis]
    kinds: set[str] = set()
    cited: list[dict] = []
    for code in codes:
        b = basis.get(code)
        if b is None:
            continue
        cited.append(b)
        lines.append(CORRESPOND.format(name=names.get(code, "결함"), cid=b["clause_id"]))
        crit_head = CRIT_HEAD_MEASURE.format(measure=b["measure"]) if b["measure"] else CRIT_HEAD
        lines.append(crit_head + " / ".join(b["criteria"]) + ".")
        kinds.update(b.get("rule_kinds", ()))
        if b.get("has_unexpressed"):
            lines.append(NOTE_UNEXPRESSED)
    cluster = sorted({b["clause_id"] for b in cited if _CLUSTER in b["conditions"]})
    if len(cluster) > 1:
        # 영상 수준 문장 하나로는 어느 조항의 조건인지 말할 수 없다 — 그때는 문장에 조항을 실어야 한다(지금 표에서는 나지 않는다).
        raise SystemExit(f"군집 조건 조항이 한 영상에 둘 이상 인용된다: {cluster} — 합계 조건 문장의 계약 밖이다")
    cited_classes = {b["defect_class"] for b in cited if b["defect_class"]}
    present_classes = {b["defect_class"] for b in basis.values() if b["defect_class"]}
    if cluster:
        lines.append(AGGREGATION)
    if any(_pairs(b["conditions"], _MIXTURE) & cited_classes for b in cited):
        lines.append(MIXTURE)
    if any(_pairs(b["conditions"], _DISTINCTION) & present_classes for b in cited):
        lines.append(DISTINCTION)
    if any(_PREMISE in b["conditions"] for b in cited):
        lines.append(PREMISE)
    if missing:
        lines.append(NO_CLAUSE_MATERIAL if not basis else
                     ", ".join(f"{names.get(c, '결함')}(ISO 6520-1 코드 {c})" for c in missing) + NO_CLAUSE_CODE_TAIL)
    if missing or kinds - {"none_permitted"}:
        lines.append(CLOSE_DIMENSIONAL)
    if "none_permitted" in kinds:
        lines.append(CLOSE_SIZE_INDEPENDENT)
    return " ".join(lines)
