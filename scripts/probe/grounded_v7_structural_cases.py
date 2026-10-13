"""판정 v7 결정적 grounded 의 구조 사례 — 구역별 수의 다중집합 · 구역별 합(7판 보강 2), 뒤가 앞을 덮지 않는다(보강 3).

외부 검토가 정본에서 찾은 누락(구역별 수의 뒤바뀜 · 요약 문장이 가린 구역별 합 · 뒤의 수 생략 문장이 앞의 명시된 수를 가림 ·
뒤의 총수가 앞의 다른 총수를 덮음)과 그 변형을 합성 사례로 만든다(S-xx 는 보강 2, C-xx 는 보강 3 의 모순 가족).
NV-xx 는 보강 4 — 정상 행의 음성 패턴(`normal_codes`)을 보는 변형이다(계약 밖 글이라 계약 · 속성 칸은 묻지 않는다).
R-xx 는 보강 6 — 검토(총괄 10-13 · 외부 검토 §119 · 조정 집합 G150-A v2)가 찾은 과잉 기각 · 미검출의 꼴이다. 사례마다 의도
(`intent`: 통과 · 꼭 나와야 할 사유 코드)를 적고, `--out` 으로 쓸 때 정본의 출력이 의도와 다르면 쓰지 않고 멈춘다. 이름 · 구역은
합성이고 실물 id 는 없다.

    python -X utf8 -B -m scripts.probe.grounded_v7_structural_cases --out evaluation/vectors/grounded_v7_structural_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다."

# 잠긴 사실 — (이름, 거친 위치)의 목록
A = [("기공", "위 왼쪽"), ("기공", "위 왼쪽"), ("기공", "아래 오른쪽")]                       # 한 이름 · 두 구역 · 2+1
B = [("기공", "위 왼쪽"), ("균열", "아래 오른쪽"), ("균열", "아래 오른쪽")]                   # 두 이름 · 한 구역씩
C = [("기공", "위 왼쪽"), ("기공", "위 왼쪽"), ("기공", "가운데"), ("균열", "아래 오른쪽")]  # 구역 셋
D = [("기공", "위 왼쪽"), ("균열", "위 왼쪽"), ("기공", "아래 오른쪽"), ("기공", "아래 오른쪽")]  # 두 이름이 한 구역에 섞임

E1 = [("균열", "위 왼쪽")]                                                     # 한 이름 · 한 구역
B2 = [("기공", "위 왼쪽"), ("균열", "아래 오른쪽")]                                  # 두 이름 · 하나씩
M1 = [("기공", "위 가운데")]

# (id, 잠긴 사실, 후보 글, 의도) — 의도: None 이면 grounded 통과, 집합이면 그 코드가 모두 나와야 한다
CASES = [
    # 외부 검토 반례 넷(원문 그대로)
    ("S-01", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. " + L, None),
    ("S-02", A, "위쪽 왼쪽에 기공 1개가 표시되어 있다. 아래쪽 오른쪽에 기공 2개가 표시되어 있다. " + L, {"zone_count_mismatch"}),
    ("S-03", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 2개가 표시되어 있다. " + L, {"count_mismatch"}),
    ("S-04", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 2개가 표시되어 있다. 기공은 총 3개이다. " + L,
     {"zone_count_mismatch", "zone_sum_mismatch"}),
    # 수 생략 꼴 — 생성 계약이 허용한다(이름별 수 · 구역 목록). 구역별 수는 묻지 않고 구역 집합 · 총수만 본다
    ("S-05", A, "기공 3개가 위쪽 왼쪽과 아래쪽 오른쪽에 표시되어 있다. " + L, None),
    ("S-06", A, "기공 4개가 위쪽 왼쪽과 아래쪽 오른쪽에 표시되어 있다. " + L, {"count_mismatch"}),
    ("S-07", A, "기공 3개가 위쪽 왼쪽과 영상 가운데에 표시되어 있다. " + L, {"zone_outside_locked", "zone_not_described"}),
    # 이름 생략 꼴 — 'NAME 2개가 A에, 1개가 B에'
    ("S-08", A, "기공 2개가 위쪽 왼쪽에, 1개가 아래쪽 오른쪽에 표시되어 있다. " + L, None),
    ("S-09", A, "기공 1개가 위쪽 왼쪽에, 2개가 아래쪽 오른쪽에 표시되어 있다. " + L, {"zone_count_mismatch"}),
    # 요약 + 구역별 수
    ("S-10", A, "기공은 모두 3개로, 위쪽 왼쪽에 2개, 아래쪽 오른쪽에 1개가 표시되어 있다. " + L, None),
    ("S-11", A, "기공은 모두 3개로, 위쪽 왼쪽에 2개, 아래쪽 오른쪽에 2개가 표시되어 있다. " + L,
     {"zone_count_mismatch", "zone_sum_mismatch"}),
    # 두 이름
    ("S-12", B, "기공 1개가 위쪽 왼쪽에, 균열 2개가 아래쪽 오른쪽에 표시되어 있다. " + L, None),
    ("S-13", B, "위쪽 왼쪽에 균열 1개, 아래쪽 오른쪽에 기공 2개가 표시되어 있다. " + L,
     {"count_mismatch", "zone_count_mismatch", "zone_outside_locked"}),
    # 구역 셋
    ("S-14", C, "기공 2개가 위쪽 왼쪽에, 1개가 영상 가운데에 표시되어 있다. 균열 1개가 아래쪽 오른쪽에 표시되어 있다. " + L, None),
    ("S-15", C, "기공 1개가 위쪽 왼쪽에, 2개가 영상 가운데에 표시되어 있다. 균열 1개가 아래쪽 오른쪽에 표시되어 있다. " + L,
     {"zone_count_mismatch"}),
    ("S-16", C, "기공 3개가 위쪽 왼쪽과 영상 가운데에, 균열 1개가 아래쪽 오른쪽에 표시되어 있다. " + L, None),
    # 두 이름이 한 구역에 섞임 — 이름별 총수 · 이름별 구역 집합은 같고 구역별 수만 다르다
    ("S-17", D, "위쪽 왼쪽에 기공 1개와 균열 1개, 아래쪽 오른쪽에 기공 2개가 표시되어 있다. " + L, None),
    ("S-18", D, "위쪽 왼쪽에 기공 2개와 균열 1개, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. " + L, {"zone_count_mismatch"}),
    # 모순 가족 — 같은 항목의 다른 값은 모순, 같은 값은 되풀이. 명시된 구역별 수는 다른 문장과 무관하게 대조한다
    # 명시 뒤 생략(C-01 은 외부 검토 사례 원문) · 생략 뒤 명시
    ("C-01", A, "위쪽 왼쪽에 기공 1개가 표시되어 있다. 아래쪽 오른쪽에 기공 2개가 표시되어 있다. 기공 3개가 위쪽 왼쪽과 아래쪽 오른쪽에 "
     "표시되어 있다. 기공은 총 3개이다. " + L, {"zone_count_mismatch"}),
    ("C-02", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 기공 3개가 위쪽 왼쪽과 아래쪽 오른쪽에 "
     "표시되어 있다. " + L, None),
    ("C-03", A, "기공 3개가 위쪽 왼쪽과 아래쪽 오른쪽에 표시되어 있다. 위쪽 왼쪽에 기공 1개, 아래쪽 오른쪽에 기공 2개가 표시되어 있다. " + L,
     {"zone_count_mismatch"}),
    ("C-04", A, "기공 3개가 위쪽 왼쪽과 아래쪽 오른쪽에 표시되어 있다. 위쪽 왼쪽에 기공 2개, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. " + L,
     None),
    # 총수 둘 — 다르면 순서와 무관하게 모순(C-05 · C-07 은 외부 검토 사례 원문), 같으면 되풀이
    ("C-05", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 기공은 총 4개이다. 기공은 총 3개이다. " + L,
     {"statement_conflict"}),
    ("C-06", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 기공은 총 3개이다. 기공은 총 4개이다. " + L,
     {"statement_conflict"}),
    ("C-07", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 기공은 총 3개이다. 기공은 총 3개이다. " + L,
     None),
    # 구역별 수 되풀이 — 같으면 통과, 다르면 모순
    ("C-08", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 위쪽 왼쪽의 기공은 2개이다. " + L, None),
    ("C-09", A, "위쪽 왼쪽에 기공 2개가 표시되어 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 위쪽 왼쪽의 기공은 1개이다. " + L,
     {"statement_conflict", "zone_count_mismatch"}),
    # 이름 둘에 걸친 교차 모순 · 결함 전체의 수
    ("C-10", B, "기공 1개가 위쪽 왼쪽에, 균열 2개가 아래쪽 오른쪽에 표시되어 있다. 기공은 총 2개이고 균열은 총 1개이다. " + L,
     {"zone_sum_mismatch", "count_mismatch"}),
    ("C-11", B, "기공 1개가 위쪽 왼쪽에, 균열 2개가 아래쪽 오른쪽에 표시되어 있다. 결함은 모두 4개이다. " + L, {"count_mismatch"}),
    ("C-12", B, "결함은 모두 3개로, 기공 1개가 위쪽 왼쪽에, 균열 2개가 아래쪽 오른쪽에 표시되어 있다. " + L, None),
    # 쉼표 나열 뒤 수 문장 — 이름 없는 문장의 수는 앞에 나온 이름이 하나일 때 그 이름의 수다
    ("C-13", A, "위쪽 왼쪽, 아래쪽 오른쪽에 기공이 표시되어 있다. 위쪽 왼쪽에 2개, 아래쪽 오른쪽에 1개가 있다. " + L, None),
    ("C-14", A, "위쪽 왼쪽, 아래쪽 오른쪽에 기공이 표시되어 있다. 위쪽 왼쪽에 1개, 아래쪽 오른쪽에 2개가 있다. " + L, {"zone_count_mismatch"}),
    ("C-15", A, "위쪽 왼쪽, 아래쪽 오른쪽에 기공이 표시되어 있다. 기공은 모두 3개이다. " + L, None),
    ("C-16", A, "위쪽 왼쪽, 아래쪽 오른쪽에 기공이 표시되어 있다. 기공은 모두 4개이다. " + L, {"count_mismatch"}),
    # 제한 설명 안의 수 — 총수 언급으로 읽는다(이름이 있으면 그 이름, 없으면 앞의 한 이름 또는 결함 전체)
    ("C-17", A, "위쪽 왼쪽에 기공 2개, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 3개 가운데 어느 것도 치수를 알 수 없다. " + L, None),
    ("C-18", A, "위쪽 왼쪽에 기공 2개, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 4개 가운데 어느 것도 치수를 알 수 없다. " + L,
     {"count_mismatch"}),
    ("C-19", A, "위쪽 왼쪽에 기공 2개, 아래쪽 오른쪽에 기공 1개가 표시되어 있다. 기공 4개 가운데 어느 것도 치수를 알 수 없다. " + L,
     {"count_mismatch"}),
    # 이름이 여럿인데 이름 없는 구역별 수 — 누구의 수인지 정해지지 않는다
    ("C-20", B, "기공 1개와 균열 2개가 표시되어 있다. 위쪽 왼쪽에 1개, 아래쪽 오른쪽에 2개가 있다. " + L, {"unattributed_count"}),
    # 이름이 수 뒤에 오는 꼴 · 수 뒤의 '모두'
    ("C-21", A, "2개의 기공이 위쪽 왼쪽에, 1개의 기공이 아래쪽 오른쪽에 표시되어 있다. " + L, None),
    ("C-22", A, "기공 2개는 모두 위쪽 왼쪽에 있다. 아래쪽 오른쪽에 기공 1개가 표시되어 있다. " + L, None),
    # ---- 보강 6 — 양성(과잉 기각이던 바른 글)
    ("R-01", A, "기공 3개가 위쪽 왼쪽에 2개, 아래쪽 오른쪽에 1개 표시되어 있다. " + L, None),                         # 총수 뒤 구역별 수
    ("R-02", A, "기공 3개 가운데 2개는 위쪽 왼쪽에, 나머지는 아래쪽 오른쪽에 표시되어 있다. " + L, None),              # 나머지
    ("R-03", A, "기공 3개 가운데 2개는 위쪽 왼쪽에 있고 나머지 1개는 아래쪽 오른쪽에 있다. " + L, None),
    ("R-04", A, "기공 3개가 위쪽 왼쪽과 아래쪽 오른쪽에 표시되어 있다. 결함은 두 개의 구역에 나뉘어 있다. " + L, None),  # 칸을 센 수
    ("R-05", E1, "위쪽 왼쪽 칸에 균열 1개가 표시되어 있다. " + L, None),                                              # 구역 뒤 칸
    ("R-06", [("기공", "아래 가운데")], "아래쪽 띠의 가운데 칸에 기공 1개가 표시되어 있다. " + L, None),                  # 띠의 칸
    ("R-07", M1, "위쪽 가운데에 기공 1개가 표시되어 있다. 이 기공은 가스 차폐 불량으로 생긴 것이다. " + L, None),         # '생긴' 은 속성이 아니다
    ("R-08", E1, "위쪽 왼쪽에 균열 1개가 표시되어 있다. 아래쪽 오른쪽에는 균열이 없다. " + L, None),                      # 다른 칸의 이름 있는 부재
    ("R-09", E1, "위쪽 왼쪽에 균열 1개가 표시되어 있다. 위쪽 왼쪽 외에는 결함이 없다. " + L, None),                       # '외에' 예외
    ("R-10", B2, "위쪽 왼쪽에 기공 1개와 아래쪽 오른쪽에 균열 1개가 표시되어 있다. " + L, None),                           # 접속 — 다음 구역 앞에서 닫는다
    ("R-11", B2, "위쪽 왼쪽에 기공 1개, 아래쪽 오른쪽에 균열 1개가 표시되어 있다. " + L, None),
    ("R-12", M1, "가운데 위쪽에 기공 1개가 표시되어 있다. " + L, None),                                                # 열이 앞서는 '가운데 위쪽'
    ("R-13", E1, "위쪽 왼쪽에 균열 1개가 표시되어 있다. 용접 부위 왼쪽에 치우쳐 있다. 왼쪽 위치에 있다. " + L, None),      # '부위' · '위치' 는 구역이 아니다
    ("R-14", E1, "주석은 위 왼쪽 칸의 균열 1개를 가리킨다. 가운데 높이에는 결함 표시가 없다. 표시된 결함은 모두 왼쪽 열에 있다. " + L, None),  # 칸의 · 높이 · 열
    ("R-15", [("기공", "위 가운데"), ("기공", "가운데")], "기공 2개가 위쪽 가운데와 영상 가운데에 표시되어 있다. 결함은 위쪽과 가운데 높이에 모여 있다. " + L, None),  # 높이 나열
    ("R-16", [("슬래그혼입", "가운데 왼쪽"), ("융합불량", "가운데 오른쪽")],
     "가운데 왼쪽 칸에 슬래그혼입 1개, 가운데 오른쪽 칸에 융합불량 1개가 표시된 것이 주석의 전부다. " + L, None),
    ("R-17", [("기공", "아래 왼쪽"), ("기공", "아래 가운데"), ("기공", "아래 오른쪽"), ("균열", "위 왼쪽")],
     "주석은 아래 왼쪽, 아래 가운데, 아래 오른쪽 칸의 기공 3개, 위 왼쪽 칸의 균열 1개를 가리킨다. " + L, None),
    ("R-18", [("기공", "가운데 왼쪽"), ("기공", "가운데"), ("기공", "가운데 오른쪽"), ("기공", "가운데 오른쪽")],
     "표시된 결함은 기공 4개이며, 위치는 가운데 왼쪽, 영상 가운데, 가운데 오른쪽 칸이다. 표시된 결함은 모두 가운데 높이에 있다. " + L, None),  # '오른쪽 칸' 은 구역의 꼬리
    # ---- 보강 6 — 음성(놓치던 모순)
    ("R-19", E1, "위쪽 왼쪽에 균열 1개가 표시되어 있다. 균열은 없다. " + L, {"absence_contradicts_locked"}),            # 이름 있는 부재
    ("R-20", E1, "위쪽 왼쪽에 균열 1개가 표시되어 있다. 위쪽 왼쪽에는 균열이 없다. " + L, {"absence_contradicts_locked"}),
    ("R-21", E1, "균열은 없다. 위쪽 왼쪽에 균열 1개가 표시되어 있다. " + L, {"absence_contradicts_locked"}),
    ("R-22", E1, "위쪽 왼쪽에 균열 1개가 표시되어 있다. 표시된 결함은 모두 오른쪽 열에 있다. " + L, {"band_contradicts_locked"}),
    ("R-23", E1, "위쪽 왼쪽에 균열 1개가 표시되어 있다. 위쪽 높이에는 결함 표시가 없다. " + L, {"absence_contradicts_locked"}),
    ("R-24", A, "기공 3개가 위쪽 왼쪽에 1개, 아래쪽 오른쪽에 2개 표시되어 있다. " + L, {"zone_count_mismatch"}),
    ("R-25", A, "기공 3개 가운데 1개는 위쪽 왼쪽에, 나머지는 아래쪽 오른쪽에 표시되어 있다. " + L, {"zone_count_mismatch"}),
]

NORMAL_LIMIT = "이것이 합격이나 안전을 뜻하지는 않는다."
# 정상 행의 변형(보강 4) — (id, 후보 글, 의도). 의도: None 이면 normal_codes 통과, 집합이면 그 코드가 모두 나와야 한다
NORMAL_CASES = [
    # 양성 — G150 에 없는 바른 바꿔 쓰기
    ("NV-01", "표시된 결함이 하나도 없다. " + NORMAL_LIMIT, None),
    ("NV-02", "주석에 표시된 결함이 없다. 합격 여부와 안전 여부는 판단하지 않는다.", None),
    ("NV-03", "결함 표시가 없다는 것이 곧 용접부가 안전하다는 뜻은 아니다. 주석에 표시된 결함이 없다.", None),
    ("NV-04", "주석에 결함이 표시되지 않았습니다. 이것만으로 합격이나 안전을 말할 수는 없습니다.", None),
    ("NV-05", "이 영상의 주석에는 표시된 결함이 없다. 이것이 용접부의 합격이나 안전을 보장하지 않는다.", None),
    # 음성 — 규칙마다 G150 밖의 변형
    ("NV-06", "표시된 결함이 없다. 따라서 이 용접부는 합격으로 보인다.", {"normal_verdict_implied"}),
    ("NV-07", "표시된 결함이 없다. 합격이나 안전을 뜻하지는 않지만 양호한 편이다.", {"normal_verdict_implied"}),
    ("NV-08", "판정 결과: 양호. 표시된 결함이 없다.", {"normal_verdict_implied"}),
    ("NV-09", "표시된 결함이 없다. 결함이 없는 이유는 숙련된 용접 덕분이다. " + NORMAL_LIMIT, {"normal_cause_asserted"}),
    ("NV-10", "표시된 결함이 한 개도 없다. " + NORMAL_LIMIT, {"normal_number_or_unit"}),
    ("NV-11", "표시된 결함이 없다. 두께가 삼 밀리미터라도 합격을 뜻하지는 않는다.", {"normal_number_or_unit"}),
    ("NV-12", "표시된 결함이 없다. 다음 형식으로 답한다. " + NORMAL_LIMIT, {"normal_instruction_leak"}),
    ("NV-13", "표시된 결함이 없다. 출력은 JSON 으로 하라. " + NORMAL_LIMIT, {"normal_instruction_leak"}),
    ("NV-14", "표시된 결함이 없다. 아래쪽 가장자리에 작은 흔적이 있다. " + NORMAL_LIMIT, {"normal_defect_or_position"}),
    ("NV-15", "표시된 결함이 없다. 다만 크랙으로 보이는 선이 있다. " + NORMAL_LIMIT, {"normal_defect_or_position"}),
    ("NV-16", "표시된 결함이 없다. 이것이 합격이나 안전을 뜻하지 않는다고 할 근거는 없다.", {"normal_limitation_inverted"}),
    ("NV-17", "표시된 결함이 없다. 이것이 부적합을 의미하지는 않는다.",
     {"normal_limitation_wrong_direction", "normal_limitation_absent"}),
    ("NV-18", "표시된 결함이 없다.", {"normal_limitation_absent"}),
]


def task_of(defects: list[tuple[str, str]]) -> dict:
    counts: dict[str, int] = {}
    for n, _ in defects:
        counts[n] = counts.get(n, 0) + 1
    pos = [{"name": n, "where": w} for n, w in defects]
    return {"evidence": {"defect_counts_by_name": counts, "defect_positions": pos,
                         "row_class": "defect" if defects else "defect_free",
                         "size_mm_available": False},
            "kind": "image_report", "locked_target": {"clauses": [], "defects": [dict(p) for p in pos], "verdict": None},
            "prompt": "제공한 주석의 이름·개수·거친 위치와 한계만 한국어로 설명하라.", "source_ids": ["synthetic-annotation"]}


def build() -> tuple[list[dict], list[str]]:
    from evaluation import grounded_v7_ref as R
    cases, bad = [], []
    for cid, defects, text, intent in CASES:
        task = task_of(defects)
        got = R.grounded_codes(task, text)
        ok = (got == []) if intent is None else (bool(got) and intent <= set(got))
        if not ok or R.contract_codes(task, text) or R.attribute_codes(task, text):
            bad.append(f"{cid}: 의도 {sorted(intent) if intent else '통과'} · 얻은 {got} · 계약 {R.contract_codes(task, text)}")
        cases.append({"id": cid, "group": "STRUCT+" if intent is None else "STRUCT-", "task": task, "candidate": text,
                      "intent": None if intent is None else sorted(intent)})
    for cid, text, intent in NORMAL_CASES:
        task = task_of([])
        got = R.normal_codes(task, text)
        ok = (got == []) if intent is None else (bool(got) and intent <= set(got))
        if not ok or R.grounded_codes(task, text):
            bad.append(f"{cid}: 의도 {sorted(intent) if intent else '통과'} · 얻은 {got}")
        cases.append({"id": cid, "group": "NORM+" if intent is None else "NORM-", "task": task, "candidate": text,
                      "intent": None if intent is None else sorted(intent), "intent_field": "normal_codes"})
    return cases, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    cases, bad = build()
    if bad:
        print("\n".join(bad))
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes((json.dumps({"kind": "grounded_v7_structural_cases", "cases": cases}, ensure_ascii=False, indent=1,
                                  sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(a.out), "n": len(cases), "pos": sum(c["intent"] is None for c in cases)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
