"""판정 v7 의 결정적 grounded — 평가 쪽 참조 구현(정본). 미니스펙 16번 7판 §13(보강 1 ~ 7).

결함 행: 글에서 사상표 이름 · 개수(아라비아 · 고유어) · 9 구역(어순 · 쪽 · 하이픈 · 띄어쓰기 · 칸 · 띠의 칸 · 영상 가운데 · 중앙)을 뽑아
잠긴 필드와 '이름 × 구역 × 수' 다중집합으로 맞댄다. 언급은 모두 모아 두고 같은 항목의 다른 값은 모순이다(뒤가 앞을 덮지 않는다).
이름 · 영역별 부재 주장도 사실로 남겨 잠긴 입력과 양립해야 한다 — 쉼표 · 연결로 이어진 칸 없는 맨 부재('…칸에서 기공 2개가
확인된다, 균열은 확인되지 않는다')는 영상 전체의 부재이고, 그 칸만의 부재는 부재 토막이나 그것을 한정하는 말에 칸이 적힌 때만
이다. 범위는 시작과 끝을 한 규칙으로 읽는다 — 제 토막에서 시작한 부재는 새 절이 닫으면 거기까지, 앞 칸을 물려받는 부재는 고치기
전처럼 바탕 절 전체. 이름 뒤의 속성 · 관계 판독 불가('균열의 모양은 알 수 없다' · '기공이 서로 이어져 있는지 알 수 없다')는 부재가
아니라 그 이름의 언급이고, 'A 와 B 에 각각
N개씩' 의 수는 그 수 바로 앞 칸들의 칸마다 수다(보강 7 · 독립 검수 반영). 예외어가 든 부재('기공을 뺀 위 왼쪽 칸에는 결함이 없다' ·
'위쪽 왼쪽 칸을 뺀 칸에는 …')는 예외로 든 칸 · 이름을 뺀 나머지로 대조하고, 칸 바로 뒤의 이름은 다른 칸의 나열에 끌려가지 않는다
(2026-10-16). 정상 행 · 제한 문장: 생성 계약(v6 생성 지시의 고정 문장)과 정확 일치, 그리고 정상 행의 음성 패턴(`normal_codes`).
주석 밖 시각 속성의 긍정 서술. 모델은 부르지 않는다.

정식 구현은 이 파일을 **바이트 그대로** 옮긴 `corpus/production/judge_grounded_v7.py` 다. 두 사본은
`evaluation/vectors/grounded_v7_vectors.jsonl` 의 같은 입력에서 같은 출력을 내야 한다 — 하니스
`scripts/probe/check_grounded_v7_equivalence.py`. 벡터는 독립 증거가 아니다(두 구현의 동치만 고정한다).

공개 함수 넷(하니스의 계약): `grounded_codes(task, text)` · `contract_codes(task, text)` · `attribute_codes(task, text)` ·
`normal_codes(task, text)` — 모두 사유 코드의 정렬된 목록을 돌려주고, 빈 목록이 통과다. `grounded_codes` 는 결함 행에서만,
`normal_codes` 는 정상 행에서만 뜻이 있다(그 밖은 빈 목록).
"""

from __future__ import annotations

import re

from corpus.production import generation_policy as G
from corpus.production import judge_policy as P
from corpus.production.prepare import _label_names

NAMES = tuple(sorted(set(_label_names().values()), key=lambda n: (-len(n), n)))
"""긴 이름 먼저, 같은 길이는 이름순 — **결정적**이다. 길이만으로 정렬하면 같은 길이 이름(용입불량 · 융합불량, 기공 · 균열)의 순서가
집합 순회, 곧 PYTHONHASHSEED 를 따라 정책 지문이 프로세스마다 갈린다(7판 보강 5, 2026-10-13)."""
KO = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8,
      "아홉": 9, "열": 10}
ROW = {"위": "위", "위쪽": "위", "아래": "아래", "아래쪽": "아래", "가운데": "가운데"}
COL = {"왼쪽": "왼쪽", "왼편": "왼쪽", "오른쪽": "오른쪽", "오른편": "오른쪽", "가운데": "가운데"}
ALL = {(r, c) for r in ("위", "가운데", "아래") for c in ("왼쪽", "가운데", "오른쪽")}
_R = r"(?<![가-힣])(위쪽|아래쪽|위|아래|가운데)"
"""행 낱말 — 앞에 한글이 붙으면 낱말이 아니다('부위 왼쪽' 의 '위')."""
_C = r"(왼쪽|오른쪽|왼편|오른편|가운데)"
_ROWB = r"(?<![가-힣])(위쪽|아래쪽|가운데)"
_COLB = r"(?<![가-힣])(왼쪽|오른쪽|가운데)"
NEG = re.compile(r"(?:없|않|아니)")
SUMMARY = re.compile(r"(?:모두|총|합쳐|합계)")
ATTR = re.compile(r"(?<![가-힣])(?:크고|작고|둥근|둥글|길쭉|가늘|굵|밝은|어두운|선명|날카로|뾰족|짧은|넓은|좁은|긴\s|큰\s|작은\s)")
"""시각 속성 어휘 — 낱말 앞에 한글이 없을 때만('생긴 · 시작은' 은 속성이 아니다, 7판 보강 6)."""
ABSENCE_EXEMPT = re.compile(r"(?:다른|나머지|외에|밖에|제외|뺀)")
"""부재문의 예외어 — '위쪽 왼쪽 외에는 결함이 없다' · '…을 뺀 칸에는 결함이 없다'. 예외로 든 칸 · 이름(`_excepted`)을 뺀 나머지로
부재를 대조한다. 무엇을 뺐는지 읽지 못하면('다른 칸에는 · 나머지는') 대조하지 않는다."""
ATTR_OK = re.compile(r"(?:알 수 없|확인할 수 없|모른|주어지지 않|판단하지 않)")
_NAMES_ALT = "|".join(map(re.escape, NAMES))
_BAND_EXCEPT = re.compile(r"(?:" + _NAMES_ALT + r")(?:\s*(?:과|와|및|,)\s*(?:" + _NAMES_ALT + r"))*"
                          r"\s*(?:(?:을|를)\s*(?:뺀|제외한|빼면|제외하면)|외의|외에|밖의|이외의|이외에|말고)"
                          r",?\s*(?=(?:(?:다른|나머지)\s*)?결함(?:은|이|들은|들이)\s)")
"""띠 소견의 예외 — '용입불량을 뺀 · 제외한 결함은 · 제외하면 결함은' · '용입불량 외의 · 말고 (다른) 결함은'. 띠 대조는 그 이름을 빼고
한다. 띠 주어('(다른 · 나머지) 결함은 · 이 · 들은') 바로 앞에 붙은 것만이다 — '기공 외에 용입불량도 있으며, 결함은 …' ·
'용입불량 외에도 · 말고도 결함은 …'(덧붙임) · '…있고 기공을 뺀 나머지도 그렇다' 는 예외가 아니다."""
_EXC_ITEM = (r"(?<![가-힣])(?:위쪽|아래쪽|위|아래|가운데|왼쪽|오른쪽|왼편|오른편|영상|중앙|가로|세로|높이|띠|줄|열|칸|구역|영역|부분)(?:의)?"
             r"|(?:" + _NAMES_ALT + r")(?:들|\s*결함(?:들)?)?")
_EXCEPT_WORD = re.compile(r"\s*(?:(?:칸|구역)\s*)?(?:(?:을|를)\s*(?:뺀|제외한|빼고|제외하고|빼면|제외하면)|외의|외에(?:는)?(?!도)"
                          r"|이외의|이외에(?:는)?(?!도)|밖의|밖에(?:는)?(?!도)|말고(?!도))")
_EXCEPT_LIST = re.compile(r"(?:(?:" + _EXC_ITEM + r")(?:\s*(?:과|와|및|,|·|ㆍ|/|이나)\s*|\s+))*(?:" + _EXC_ITEM + r")$")
"""부재의 예외로 든 것 — 예외어(`_EXCEPT_WORD`) 바로 앞의 칸 · 띠 · 이름 나열('위쪽 왼쪽 칸을 뺀' · '기공을 제외한' · '위쪽 높이와
아래 가운데를 뺀'). 이름은 '들 · 결함' 이 붙은 꼴도('기공들을 · 기공 결함을 뺀'), 칸은 '영역 · 부분' 이 붙은 꼴도('위쪽 왼쪽 영역 외에는'),
나열 사이는 '과 · 와 · 및 · 쉼표 · · · ㆍ · / · 이나' 다('기공·균열 · 기공/균열 · 기공이나 균열 외에는'). 덧붙임('외에도 · 말고도')은
예외어가 아니다."""
_ATTR_NOUN = (r"(?:모양|형태|형상|생김새|크기|길이|너비|폭|깊이|두께|굵기|넓이|면적|밝기|명암|농도|색상|색깔|색|윤곽|테두리"
              r"|방향|기울기|각도|간격|끝부분|끝|치수"
              r"|(?<![가-힣])(?:질감|색조|색채|빛깔|농담|음영|그림자|밀도|명도|채도|선명도|투명도|외형|외관|모양새|무늬|패턴|광택"
              r"|윤기|곡률|지름|직경|반지름|굴곡|요철|방향성|꼴(?=이|을|은|의)|결(?=(?:이|은|을|도)\s)))"
              r"(?=[\s,·ㆍ/]|과|와|이나|나|및|은|는|이|가|을|를|도|만|에|의|$)")
"""주석에 없는 속성의 이름. 위치 · 수 · 존재 · 분포는 넣지 않는다 — 주석이 정한 것을 모른다는 말은 모순이다. 낱말 끝을 본다('끝내').
뒤에 더한 이름(질감 · 색조 · 그림자 · 무늬 …)은 낱말 머리에서만 읽고, '꼴 · 결' 은 토씨가 붙은 꼴로만 읽는다('결함 · 결과' 의 '결' 은
아니다). '모습 · 형체' 는 넣지 않는다 — '균열의 모습은 영상에서 드러나지 않는다' 는 그 결함이 없다는 말로 읽힌다."""
_MOD = r"(?:(?:정확한|실제|실제의|구체적인|구체적|자세한|세부|세밀한|정밀한|전체|전체적인|개별|그)\s*)"
_ADV = (r"(?:(?:주석|영상|사진|이미지|자료|이것|이\s*정보|주어진\s*정보|현재\s*정보)"
        r"(?:만으로는|만으로|만|으로는|으로|로는|로|에서는|에서|에는|에)?|정확히|정확하게|따로|직접|현재로서는|지금으로서는"
        r"|현재|지금|아직|자세히|구체적으로|모두|각각|둘\s*다)\s*")
"""속성 낱말과 앎의 부정 사이에 올 수 있는 말 — 닫힌 목록이다('크기는 물론 존재 여부도' · '크기와 상관없이' 처럼 다른 말이 끼면 아니다)."""
_KNOW_NEG = (r"(?:(?:알|확인할|판단할|단정할|말할|파악할|가늠할|측정할|특정할|짐작할|추정할|구별할|구분할|잴|(?<![가-힣])볼)\s*수\s*"
             r"(?:는|가|도)?\s*없"
             r"|모르|모른"
             r"|(?:주어지|나와\s*있|적혀\s*있|담겨\s*있|기록되어\s*있|포함되어\s*있|표시되어\s*있|표시되|제공되|알려져\s*있|알려지|드러나"
             r"|확인되|나타나\s*있|알려\s*주)지\s*(?:는|도)?\s*않"
             r"|(?:판단|단정|추정|측정)하지\s*(?:는|도)?\s*않"
             r"|(?:정보|자료|기록)(?:가|는|도)?\s*(?:(?:주석|영상)에(?:는|도)?\s*)?없)")
"""앎 · 정보의 부정 — 그것이 없다는 말이 아니라 그것을 알 수 없다는 말. '볼' 은 낱말 머리일 때만이다('찾아볼 · 알아볼 · 살펴볼 수
없다' 는 앎의 부정이 아니다)."""
_REL_STEM = (r"(?:이어져|이어진|이어지|연결되|연결돼|연결된|붙어|닿아|맞닿|겹쳐|겹치|모여|몰려|떨어져|나란히|인접해|인접한|이웃해"
             r"|이웃한)")
"""결함 사이 관계의 말 — 닫힌 목록이다. 이 말 뒤의 '…있는지 · …있다고' 가 앎의 부정으로 끝나면 관계를 모른다는 말이다."""
ATTR_UNKNOWN = re.compile(
    r"(?:\s*(?:과|와|및|,|·|이나|나)\s*(?:" + _NAMES_ALT + r"))*"
    r"(?:\s*(?:결함|부위|부분|사이|각각))?\s*(?:들)?(?:의|은|는|이|가|도|에\s*대한|에\s*대해서는|에\s*대해|에\s*관한|에\s*관해서는)?\s*"
    r"(?:" + _ADV + r")*"
    r"(?:" + _MOD + "?" + _ATTR_NOUN
    + r"(?:\s*(?:과|와|이나|나|및|,|·|ㆍ|/)\s*(?:(?:" + _NAMES_ALT + r")\s*의\s*)?" + _MOD + "?" + _ATTR_NOUN + r")*"
    r"\s*(?:정보)?(?:은|는|이|가|을|를|도|만|까지|조차)?\s*(?:에\s*(?:대한|관한)\s*)?"
    r"(?:서로\s*(?:같은지|다른지|비슷한지)(?:는|를|도)?\s*)?"
    r"|얼마나\s*(?:깊|길|긴|넓|크|큰|작|굵|가늘|두껍|두꺼|얇|밝|어두)[가-힣\s]{0,8}?지(?:는|를|도)?\s*"
    r"|(?:(?:끝|끝부분)(?:이|은|는)?\s*)?어디까지\s*(?:이어|뻗어|이르|퍼져|연결되)[가-힣\s]{0,6}?지(?:는|를|도)?\s*"
    r"|어느\s*방향으로\s*[가-힣\s]{0,8}?지(?:는|를|도)?\s*"
    r"|(?:서로\s*)?" + _REL_STEM + r"(?:(?!않|없|아니)[가-힣\s]){0,8}?(?:지|다고|라고)(?:는|를|도)?\s*"
    r"|(?:어떤\s*(?:모양|형태|형상|방향|색)(?:인지|이었는지)|어떻게\s*생겼는지)(?:는|를|도)?\s*"
    r"|(?:큰지\s*작은지|긴지\s*짧은지|깊은지\s*얕은지|굵은지\s*가는지|넓은지\s*좁은지|밝은지\s*어두운지)(?:는|를|도)?\s*)"
    r"(?:" + _ADV + r")*" + _KNOW_NEG + r"[가-힣]{0,3}\s*$")
"""이름(또는 '결함') 바로 뒤에서 **절 끝까지**가 속성 판독 불가 — '균열의 모양은 주석만으로 알 수 없다' · '균열의 길이와 폭은 …' ·
'균열이 얼마나 깊은지는 주석에 나와 있지 않다' · '기공과 균열의 모양과 크기는 …'. 관계 판독 불가('기공들이 서로 이어져 있는지는
알 수 없다' · '…이어져 있다고 볼 수는 없다')도 같다 — 관계의 말과 '지 · 다고' 사이에 부정이 끼면 아니다. 부재 주장이 아니라 그
이름의 언급이다(잠기지 않은 이름이면 이름 불일치). 존재 · 위치 · 수를 모른다는 말, 닫힌 목록 밖의 말이 끼는 꼴, 절이 끝나기 전에 다른 서술이 이어지는 꼴
('모양을 알 수 없을 뿐 기공은 없다')은 아니다 — 부재 주장으로 남는다(보강 7)."""
_GLOBAL = re.compile(r"(?:어디에도|어디서도|어디에서도|어느\s*(?:칸|곳)에(?:서)?도|영상\s*전체|영상\s*어디|모든\s*칸|전체\s*영상|아무\s*데도"
                     r"|아무\s*곳에도)")
"""영상 전체를 가리키는 말 — 정형 부재 주장에 붙으면 앞 절의 구역을 물려받지 않는다('위쪽 왼쪽에는 기공이 있다, 균열은 어디에도
없다')."""
_NONE_AT_ALL = re.compile(r"\s*(?:도|조차|마저|라도)\s*(?:없|보이지\s*않|확인되지\s*않|발견되지\s*않|표시되어\s*있지\s*않|표시되지\s*않"
                          r"|존재하지\s*않|드러나지\s*않"
                          r"|나타나지\s*않|관찰되지\s*않|찾을\s*수\s*없)")
"""수 뒤의 '도 · 조차 + 부정' — '균열이 1개도 없다' 는 1개의 언급이 아니라 부재다('1개도 크기는 알 수 없다' 의 '도' 는 아니다)."""
_ABS_PRED = (r"(?:(?:보이|표시되|확인되|발견되|관찰되|나타나|존재하|드러나)지\s*(?:는\s*)?않|표시되어\s*있지\s*(?:는\s*)?않"
             r"|찾을\s*수\s*(?:는\s*)?없|없)"
             r"(?:는다|다|았다|었다|습니다|는|았|었)?")
"""평범한 부재 술어(토막 끝) — 앎의 부정 · 의심 · 인용('알 수 없다' · '확실하지 않다' · '있는지' · '없다고')은 넣지 않는다. '보이지는
않는다' 는 부재 술어다 — 전칭어가 걸린 부분 부정('모든 칸에서 보이지는 않는다')만 범위를 고치기 전처럼 읽는다(`_PARTIAL`)."""
_PARTIAL = re.compile(r"(?:모든|전부|모두|전체|(?<![가-힣])다(?=\s))[^,]*?지\s*는\s*않")
"""전칭어가 걸린 부분 부정('균열은 모든 칸에서 보이지는 않는다' — 어느 칸에는 있을 수 있다는 말) — 그 범위는 고치기 전처럼 바탕
절이다(`_absence_scope`)."""
_N = (r"(?:" + "|".join(map(re.escape, NAMES)) + r")(?:\s*(?:과|와|이나|나|및|,)\s*(?:" + "|".join(map(re.escape, NAMES)) + r"))*"
      r"(?:은|는|이|가)")
_FILL = r"\s*(?:(?:전혀|아예|단)\s*)?(?:(?:1|한)\s*개\s*도\s*|하나\s*도\s*)?"
_ZL = r"§(?:\s*(?:칸|구역))?(?:\s*(?:와|과|및|,)\s*§(?:\s*(?:칸|구역))?)*\s*(?:에는|에서는|에서|에)\s*"
_GL = r"(?:" + _GLOBAL.pattern + r")(?:에서도|에서는|에서|에는|에도|에)?\s*"
_CANON = {"bare": re.compile(r"\s*" + _N + _FILL + _ABS_PRED + r"\s*"),
          "global": re.compile(r"\s*(?:" + _N + r"\s*" + _GL + "|" + _GL + _N + ")" + _FILL + _ABS_PRED + r"\s*"),
          "zoned": re.compile(r"\s*(?:" + _ZL + _N + "|" + _N + r"\s*" + _ZL + ")" + _FILL + _ABS_PRED + r"\s*")}
"""정형 부재 주장 — 토막이 이름 + 은 · 는 · 이 · 가 + 평범한 부재 술어뿐인 꼴(bare), 전역어가 붙은 꼴(global), 구역과 처소 조사가
붙은 꼴(zoned — 구역은 § 로 바꿔 본다). '도' · 되짚는 말 · 관형절 · 대조('…과 달리') · 의심이 끼면 정형이 아니다(고치기 전처럼 읽는다)."""
_REL = re.compile(r"\s*(?:이|가|은|는|도)?\s*(?:함께\s*|같이\s*|몰려\s*)?(?:있는|표시된|나타난|보이는|위치한|놓인|확인된)\s")
"""관형절의 이름('기공 1개가 있는 …칸' · '기공 1개가 표시된 …칸') — 그 이름은 새 절의 주어가 아니다."""
_LIMIT = re.compile(r"(?:한한|한정|두고|경우다|그러하다|결과다|이야기다|말이다|제외|외에는|말고|밖에서|살펴|보면"
                    r"|마찬가지|같다|매한가지|해당|그런\s*셈|똑같|역시|또한|동일)")
"""범위를 한정하는 말('…칸에 한한 이야기다' · '…칸을 두고 하는 말이다' · '…칸을 제외하면' · '…칸만 보면 그러하다')과 되받는
말('…칸도 마찬가지다' · '…칸 또한 그러하다') — 이 말이 든 토막은 새 절이 아니라 앞 부재의 범위다."""
_EMPTY = re.compile(r"\s*(?:(?:칸|구역)\s*)?(?:은|는|이|가)\s*(?:비어|깨끗|아무\s*(?:것|결함)도\s*없)")
"""빈 칸 서술('영상 가운데 칸은 비어 있다') — 그 칸은 어떤 이름에도 붙이지 않는다."""
_LIST_GAP = r"\s*(?:(?:\d+|" + "|".join(KO) + r")\s*개\s*)?(?:와|과|및|,)\s*"
"""칸 바로 뒤 이름 나열의 이음('기공과 균열' · '기공 1개와 균열')."""
_NAME_TAIL = r"\s*(?:(?:\d+|" + "|".join(KO) + r")\s*개)?\s*(?:만이|만|이|가|은|는|도)?\s*"
"""이름과 그 바로 뒤 칸 사이에 올 수 있는 말 — 수와 조사뿐('기공 1개가 ZONE에' · '기공과 균열은 ZONE에')."""
_ONLY = re.compile(r"(?:개|" + "|".join(map(re.escape, NAMES)) + r")\s*만")
"""'만'(그것뿐) — '아래쪽 오른쪽 칸에는 기공 1개만 있다' 는 그 칸에 다른 결함이 없다는 말이다."""
_POS_END = re.compile(r"(?:(?:따로|함께|각각|모두|같이|나란히)\s*)?(?:(?:(?:표시되어|나타나|위치해)\s*)?있(?:다|었다|(?=\s*$))|위치한다|위치했다"
                      r"|확인된다|확인되었다|확인됐다|나타난다|나타났다|관찰된다|관찰되었다|관찰됐다"
                      r"|존재한다|존재했다|보인다|보였다)")
"""토막을 끝내는 긍정 서술 — 'ZONE에 있다 · 표시되어 있다' 의 구역은 처소 주제가 아니라 그 서술의 자리다('ZONE에 표시된 결함만 보면
…' 의 '표시된' 은 아니다)."""
_ZL_ANY = r"§(?:\s*(?:칸|구역))?(?:\s*(?:와|과|및|,)\s*§(?:\s*(?:칸|구역))?)*\s*"
_ZLOC = _ZL_ANY + r"(?:에는|에서는|에서|에|의)\s*"
_ZLOC_PLAIN = _ZL_ANY + (r"(?:에는|에서는|에서|엔|에선|의\s*경우(?:에는|에)?|안에서는|안에서|안에는|에|의)\s*")
_FRAME = r"(?:(?:이\s*)?(?:영상의\s*)?주석(?:에는|에서는|에|상|에\s*따르면)\s+|(?:이\s*)?영상(?:에는|에서는)\s+)?"
_NCNT = r"(?:" + "|".join(map(re.escape, NAMES)) + r")\s*(?:(?:\d+|" + "|".join(KO) + r")\s*개)?"
_NLIST = _NCNT + r"(?:\s*(?:와|과|및|,)\s*" + _NCNT + r")*\s*(?:만이|만|이|가|은|는|도)?\s*"
_POSITIVE = re.compile(r"\s*(?:(?:그리고|또한|한편|다만|아울러)\s+)?" + _FRAME + r"(?:(?:" + _ZLOC + r")?" + _NLIST + r"(?:" + _ZLOC + r")?"
                       r"(?:,|(?:표시되어\s*)?있(?:고|으며))?\s*)+" + _POS_END.pattern + r"\s*")
"""엄격한 긍정 서술 — 토막(구역은 § 로 바꿔 본다)이 구역 · 이름(수) · 긍정 술어뿐인 꼴('아래쪽 오른쪽에 기공 1개가 표시되어
있다' · '기공 1개가 아래쪽 오른쪽에 있다' · '아래쪽 오른쪽의 기공 1개만 표시되어 있다' · '…에 기공 1개, …에 기공 1개가 있다'). 관형절
('기공 1개가 있는 …칸') · 되받는 말('…도 같다') · 틀('…만 보면') · 예외어 · 한정('…칸에 한한 이야기다')이 끼면 맞지 않는다."""
_POSITIVE_PLAIN = re.compile(r"\s*(?:(?:그리고|또한|한편|다만|아울러)\s+)?" + _FRAME + r"(?:(?:" + _ZLOC_PLAIN + r")?" + _NLIST
                             + r"(?:" + _ZLOC_PLAIN + r")?(?:,|(?:표시되어\s*)?있(?:고|으며))?\s*)+" + _POS_END.pattern + r"\s*")
"""앞 구역 판정용 엄격한 긍정 서술 — 처소 주제('…에는 · 에서 · 엔 · 칸의 경우 · 칸 안에서는')도 받는다. 쉼표 · 연결로 이어진 칸 없는
맨 부재('위쪽 왼쪽 칸에서 기공 2개가 확인된다, 균열은 확인되지 않는다')는 영상 전체의 부재다 — 그 칸만의 부재는 부재 토막에 칸이
적힌 때만이다('그 칸에는 · 같은 칸에서 · 거기에는', 전제 2)."""
_ASSERTED = re.compile(r"(?:(?:과|와|및|,)\s*(?:" + "|".join(map(re.escape, NAMES)) + r")\s*)*(?:은|는|이|가|도)(?=[\s,]|$)")
"""이름 바로 뒤의 주어 · 주제 꼴('균열은' · '기공과 균열은') — 그 이름은 제 토막에서 긍정 · 부재를 가른다. '슬래그혼입처럼' · '균열로'
같은 다른 언급은 고치기 전처럼 바탕 절에서 뒤의 부정을 본다."""


def zone_key(r: str, c: str) -> tuple[str, str]:
    return ROW[r], COL[c]


def zone_spans(text: str) -> list[tuple[int, int, tuple[str, str]]]:
    """두 낱말 구역(어순 무관 · 쪽 · 하이픈 · 띄어쓰기), '띠의 칸'(아래쪽 띠의 가운데 칸), '영상 가운데 · 중앙' 의 자리.
    겹치면 먼저 찾은 것을 쓴다 — 띠의 칸 → 행 열 → 열 행 → 중앙 순."""
    spans: list[tuple[int, int, tuple[str, str]]] = []

    def add(m, z):
        if not any(s < m.end() and m.start() < e for s, e, _ in spans):
            spans.append((m.start(), m.end(), z))
    for m in re.finditer(_ROWB + r"\s*(?:가로\s*)?띠의\s*" + _C + r"\s*(?:구역|칸)", text):
        add(m, zone_key(m.group(1), m.group(2)))
    for m in re.finditer(_COLB + r"\s*세로\s*띠의\s*(위쪽|아래쪽|가운데)\s*(?:구역|칸)", text):
        add(m, zone_key(m.group(2), m.group(1)))
    for m in re.finditer(_R + r"[ -]?" + _C + r"(?!\s*(?:세로\s*)?띠)", text):
        add(m, zone_key(m.group(1), m.group(2)))
    for m in re.finditer(r"(?<![가-힣])(왼쪽|오른쪽|왼편|오른편|가운데)[ -]?(위쪽|아래쪽|위|아래|가운데)(?!치)", text):
        add(m, zone_key(m.group(2), m.group(1)))   # 열이 앞서는 꼴 — '왼쪽 가운데' 는 (가운데, 왼쪽), '가운데 위쪽' 은 (위, 가운데). '왼쪽 위치' 는 구역이 아니다
    for m in re.finditer(r"(?:영상 가운데|중앙|가운데 구역|가운데 칸)", text):
        add(m, ("가운데", "가운데"))
    return sorted(spans)


def zones_in(text: str) -> tuple[set, str]:
    return {z for _, _, z in zone_spans(text)}, text


def regions_in(text: str) -> set:
    """띠 · 높이 · 열 · 구역 · 가장자리 같은 넓은 영역(칸의 집합). 두 낱말 구역에 든 낱말은 띠로 읽지 않는다 —
    '가운데 오른쪽 칸' 의 '오른쪽 칸' 은 세로 띠가 아니라 구역의 꼬리다(7판 보강 6)."""
    zs = [(s, e) for s, e, _ in zone_spans(text)]

    def free(m) -> bool:
        return not any(s < m.end() and m.start() < e for s, e in zs)
    out = set()
    # 띠 · 높이 · 열은 나열될 수 있다 — '위쪽과 가운데 높이' 는 두 행의 합이다
    for m in re.finditer(r"((?:(?<![가-힣])(?:위쪽|아래쪽|가운데)\s*(?:과|와|및|,)\s*)*)" + _ROWB + r"\s*(?:가로\s*)?(?:띠|높이|줄)", text):
        if free(m):
            for r in re.findall(r"위쪽|아래쪽|가운데", m.group(1)) + [m.group(2)]:
                out |= {z for z in ALL if z[0] == ROW[r]}
    for m in re.finditer(r"((?:(?<![가-힣])(?:왼쪽|오른쪽|가운데)\s*(?:과|와|및|,)\s*)*)" + _COLB + r"\s*(?:세로\s*띠|열)", text):
        if free(m):
            for c in re.findall(r"왼쪽|오른쪽|가운데", m.group(1)) + [m.group(2)]:
                out |= {z for z in ALL if z[1] == COL[c]}
    for m in re.finditer(r"(?<![가-힣])(왼쪽|오른쪽)\s*(?:구역|칸)", text):
        if free(m):
            out |= {z for z in ALL if z[1] == COL[m.group(1)]}
    for m in re.finditer(r"(?<![가-힣])(위쪽|아래쪽)(?:\s*(?:구역|칸))?(?:에는|에도|은|는)", text):
        if free(m):
            out |= {z for z in ALL if z[0] == ROW[m.group(1)]}
    if "가장자리" in text:
        out |= ALL - {("가운데", "가운데")}
    return out



def _zone_led(s: str, at: int, spans: list) -> bool:
    """s[at:] 의 이름이 칸 바로 뒤('ZONE에 · 의 (는)' 다음)에서 시작하는가 — 그 이름은 그 칸의 것이다."""
    return any(ze <= at and re.fullmatch(r"\s*(?:(?:구역|칸)\s*)?(?:에|의)\s*(?:는\s*)?", s[ze:at]) for _, ze, _ in spans)


def _excepted(s: str, lo: int = 0, hi: int | None = None) -> tuple[set, set, list] | None:
    """s[lo:hi] 의 부재 예외로 든 이름 · 칸과 그 나열의 자리 — 예외어 바로 앞의 칸 · 띠 · 이름 나열(`_EXCEPT_LIST`). 예외어가 없거나
    그 앞이 칸 · 이름이 아니면 None 이다 — 무엇을 뺐는지 모르니 고치기 전처럼 대조하지 않는다."""
    hi = len(s) if hi is None else hi
    names, zones, spans = set(), set(), []
    for m in _EXCEPT_WORD.finditer(s, lo, hi):
        lst = _EXCEPT_LIST.search(s, lo, m.start())
        if lst:
            t = lst.group(0)
            names |= {n for n in NAMES if n in t}
            zones |= {z for _, _, z in zone_spans(t)} | regions_in(t)
            spans.append(lst.span())
    return (names, zones, spans) if spans else None


def _excepted_absence(claim: set, lzones: dict, but: tuple[set, set, list]) -> bool:
    """예외가 든 부재 주장이 잠긴 입력과 모순인가 — 예외로 든 칸을 뺀 범위(남은 칸이 없으면 뺀 칸 밖 모두: '위쪽 왼쪽 칸을 뺀 칸에는',
    뺀 칸도 없으면 영상 전체: '기공을 뺀 결함은 없다')에 예외로 들지 않은 이름의 잠긴 칸이 있으면 모순이다('기공을 뺀 위 왼쪽 칸에는
    결함이 없다' — 그 칸에 용입불량이 있다). 뺀 것이 칸도 이름도 아니면('영상 밖에는') 무엇을 뺐는지 모르니 대조하지 않는다."""
    names, zones, _ = but
    claim = claim - zones
    if not claim:
        if not (names or zones):
            return False
        claim = ALL - zones
    return bool(claim & set().union(set(), *(z for n, z in lzones.items() if n not in names)))


def locked_of(task: dict) -> tuple[dict, dict]:
    g = P.grouped_locked_fields(task)
    counts = dict(g["defect_counts_by_name"])
    zones: dict[str, set] = {}
    zc: dict[tuple, int] = {}
    for zname, by in g["defects_by_zone"].items():
        parts = zname.split()
        z = ("가운데", "가운데") if parts == ["가운데"] else (parts[0], parts[1])
        for n, k in by.items():
            zones.setdefault(n, set()).add(z)
            zc[(n, z)] = zc.get((n, z), 0) + int(k)
    return counts, zones, zc


COUNT_AFTER_NAME = (r"(?:은|는|이|가)?\s*(모두\s*|총\s*)?(?:(\d+)\s*개|(" + "|".join(KO) + r")\s*개|(하나|둘|셋|넷)(?=가|이|만|\s|$))")
_UNIT_NOT_DEFECT = r"(?!\s*(?:의\s*)?(?:구역|칸|영역|곳|위치|띠|줄|부분|열|높이))"
"""'두 개의 구역' · '세 칸' — 자리를 센 수는 결함의 수가 아니다."""
BARE_COUNT = re.compile(r"(?<![가-힣\d])(?:(\d+)|(" + "|".join(KO) + r"))\s*개(?!수)" + _UNIT_NOT_DEFECT)
_EACH_AFTER = re.compile(r"\s*씩")
_EACH_BEFORE = re.compile(r"(?<![가-힣])각(?:각)?\s*$")
"""배분 표지 — 수 뒤의 '씩'('각각 1개씩' · '1개씩'), 수 바로 앞의 '각각 · 각'('각각 1개'). 그 수는 칸마다의 수다(보강 7)."""


def _per_zone(s: str, start: int, end: int, lead: int | None = None) -> bool:
    """s[start:end] 의 수가 칸마다의 배분 수인가 — 배분 표지(`_EACH_AFTER` · `_EACH_BEFORE`)가 있고, 수(이름 붙은 수면 그 이름, lead)
    바로 앞이 '칸 + 에(는 · 서)' 일 때('위쪽 왼쪽과 아래쪽 오른쪽에 각각 1개씩'). '두 결함은 각각 2개씩' · '균열과 기공 각 2개' 처럼
    수 앞이 칸이 아니면 이름마다의 배분이라 펼치지 않는다."""
    if not (_EACH_AFTER.match(s, end) or _EACH_BEFORE.search(s, 0, start)):
        return False
    head = re.sub(r"\s*(?:각각|각)?\s*$", "", s[:start if lead is None else lead])
    named = re.search(r"(?:" + "|".join(map(re.escape, NAMES)) + r")(?:이|가|은|는|도)?$", head)
    if named and not re.search(r"(?:과|와|및|,)\s*$", head[:named.start()]):    # '…칸에 기공이 각각 1개씩' — 이름 나열은 아니다
        head = head[:named.start()].rstrip()
    m = re.search(r"(?:\s*(?:칸|구역))?\s*에(?:는|서)?$", head)
    return bool(m) and any(ze == m.start() for _, ze, _ in zone_spans(s))


def _per_zone_list(s: str, start: int, end: int, lead: int | None = None) -> tuple:
    """배분 수(`_per_zone`)이면 그 수 바로 앞의 칸 목록('위쪽 왼쪽과 아래쪽 오른쪽에' — 와 · 과 · 및 · 쉼표로 이어진 칸), 아니면 빈 꼴."""
    if not _per_zone(s, start, end, lead):
        return ()
    head = re.sub(r"\s*(?:각각|각)?\s*$", "", s[:start if lead is None else lead])
    named = re.search(r"(?:" + "|".join(map(re.escape, NAMES)) + r")(?:이|가|은|는|도)?$", head)
    if named and not re.search(r"(?:과|와|및|,)\s*$", head[:named.start()]):
        head = head[:named.start()].rstrip()
    m = re.search(r"(?:\s*(?:칸|구역))?\s*에(?:는|서)?$", head)
    spans = zone_spans(s)
    i = next(j for j, (_, ze, _) in enumerate(spans) if ze == m.start())
    out = [spans[i][2]]
    while i > 0 and re.fullmatch(r"\s*(?:(?:칸|구역)\s*)?(?:와|과|및|,)\s*", s[spans[i - 1][1]:spans[i][0]]):
        i -= 1
        out.insert(0, spans[i][2])
    return tuple(dict.fromkeys(out))


def _in_name_list(s: str, at: int, n: str) -> bool:
    """s[at:] 의 이름 n 이 이름 나열('균열과 기공' · '균열 2개와 기공 1개') 안인가 — 그 뒤의 배분 수는 이름마다일 수 있다."""
    names = "|".join(map(re.escape, NAMES))
    return bool(re.search(r"(?:" + names + r")\s*(?:(?:\d+|" + "|".join(KO) + r")\s*개)?\s*(?:와|과|및|,)\s*$", s[:at])
                or re.match(r"\s*(?:(?:\d+|" + "|".join(KO) + r")\s*개)?\s*(?:와|과|및)\s*(?:" + names + r")", s[at + len(n):]))


_KO_ALT = "|".join(KO)
GRAND = re.compile(r"결함(?:은|는|이|가|들은|들이)?\s*(?:모두\s*|총\s*|합쳐\s*)?(?:(\d+)|(" + _KO_ALT + r"))\s*개(?!수)" + _UNIT_NOT_DEFECT
                   + r"|(?<![가-힣\d])(?:(\d+)|(" + _KO_ALT + r"))\s*개의\s*결함")


SUBSET = re.compile(r"(?:그중|그 중|그 가운데|이 가운데|이 중|이 가운데)\s*$")


def add_zone_counts(n: str, ks: list, zl: list, zone_st: dict, group_st: dict, total_st: dict) -> None:
    """한 문장에서 한 이름의 수들과 구역들을 언급으로 쌓는다. 구역 하나에 수 하나, 수 하나가 구역 개수와 같으면 하나씩(적힌 구역마다
    적어도 하나라 하나로 정해진다), 수와 구역이 같은 개수면 차례대로. 그 밖(수 생략 꼴 등)은 '그 구역 묶음의 합' 언급이고,
    구역이 없으면 이름의 총수 언급이다."""
    if not zl:
        total_st.setdefault(n, []).append(sum(ks))
    elif len(ks) == 1 and len(zl) == 1:
        zone_st.setdefault((n, zl[0]), []).append(ks[0])
    elif len(ks) == 1 and ks[0] == len(zl):
        for z in zl:
            zone_st.setdefault((n, z), []).append(1)
    elif len(ks) == len(zl):
        for z, k in zip(zl, ks):
            zone_st.setdefault((n, z), []).append(k)
    else:
        group_st.setdefault((n, frozenset(zl)), []).append(sum(ks))


_BASE_CUT = re.compile(r"(?:지만|으나|는데|그러나|그런데|(?<=[없않])고)[,\s]+")
"""결함 행의 바탕 경계 — 고치기 전 정본의 결함 행이 실제로 쓰던 경계(정상 행 규칙 `_CLAUSE_SPLIT` 과 같은 꼴). 늘 나눈다."""
_NEW_CUT = re.compile(r"(?<=다),\s*|(?:(?<=[없않])으며|(?<=않았|없었)(?:으며|고)|(?<=[없않])으므로|(?<=[없않])(?:어서|아서)"
                      r"|(?<=[없않])(?:어|아)(?=,)|(?<=[없않])기\s*때문에)[,\s]+|(?<=\s없이),\s*|(?<=[없않]는)\s(?:반면에|대신에|반면|대신),\s*"
                      r"|(?<=있)고,\s*")
"""결함 행의 새 경계 — 완결된 서술 뒤의 쉼표('균열은 없다, …' · '…표시되어 있다, 균열은 없다'), 부정 서술 뒤의 '으며'(없으며 ·
않았으며) · 까닭(없으므로 · 없어서 · 없어, · 없기 때문에), 쉼표가 붙은 '없이,' · '없는 반면,' · '없는 대신,'. `_cut_ok` 일 때만 나눈다.
긍정 서술 뒤의 이음(있고 · 보이며 …) · 계사(이고 · 이며) · 정체 부정(아니고 · 아니며), 쉼표 없는 '없이 · 반면 · 대신' 은 나누지 않는다 —
그 뒤의 서술이 앞 부재와 같은 자리를 말하는 꼴('균열 없이 기공 1개만 위쪽 왼쪽 칸에 있다')을 갈라 고치기 전에 맞던 글을 새로 틀리게
했다(37 §2, 보강 7)."""
_SUBJECT = re.compile(r"(?:" + _NAMES_ALT + r")(?:(?:은|는|이|가|도)(?=[\s,]|$)|" + COUNT_AFTER_NAME + ")")
"""주어나 수가 붙은 이름('균열은 · 균열이 · 기공 1개가') — 오른쪽 토막이 그 이름의 자기 서술이라는 표지('기공과 이어져 · 기공의 크기는' 은 아니다)."""


def _attr_unknown_clause(c: str) -> bool:
    """이름 없는 절 전체가 속성 판독 불가인가 — '크기는 알 수 없다' · '그 결함의 모양은 주석만으로 알 수 없다'."""
    lead = re.match(r"\s*(?:그\s*)?(?:결함)?", c)
    return bool(ATTR_UNKNOWN.match(c, lead.end()))


def _cut_ok(s: str, b: int, end: int) -> bool:
    """새 경계에서 나눌지 — 오른쪽 토막(s[b:end], 다음에 실제로 나눈 자리까지)에 주어로 쓴 이름이나 수가 붙은 이름이 있을 때만
    나눈다. 이름 없는 토막은 앞 절과 떼지 않는다 — 고치기 전처럼 한 절로 읽혀 부정의 주어가 이어진다. 토막은 주어로 쓴 이름의 긍정 ·
    부재, 속성 판독 불가의 절 끝, 정형 부재 주장, 부재 범위의 창을 정하는 데 쓴다(`_absence_scope` · `defect_grounded`)."""
    if _SUBJECT.search(s[b:end]):
        return True
    lead = re.match(r"\s*(?:" + _NAMES_ALT + r")", s[b:end])        # '균열의 길이도 알 수 없다' — 그 이름의 속성 판독 불가뿐인 토막
    return bool(lead) and bool(ATTR_UNKNOWN.match(s[b:end], lead.end()))


def _base_clause_spans(s: str) -> list[tuple[int, int]]:
    """바탕 경계로만 나눈 절 자리 — 고치기 전 정본의 결함 행이 실제로 쓰던 절."""
    out, start = [], 0
    for m in _BASE_CUT.finditer(s):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(s)))
    return [(x, y) for x, y in out if s[x:y].strip()]


def _defect_clause_spans(s: str) -> list[tuple[int, int]]:
    """결함 행 문장의 절 자리(시작, 끝) — 부재 주장의 범위(이름 · 구역)는 절 안이다. 정상 행의 절(`_normal_clauses`)과 다른 함수다.
    바탕 경계(`_BASE_CUT`)는 늘 나누고, 새 경계(`_NEW_CUT`)는 `_cut_ok` 일 때만 나눈다. 오른쪽 토막은 다음에 실제로 나눈 자리까지라서
    뒤에서부터 정한다."""
    cuts = [(m.start(), m.end()) for m in _BASE_CUT.finditer(s)]
    cand = [(m.start(), m.end()) for m in _NEW_CUT.finditer(s) if not any(x < m.end() and m.start() < y for x, y in cuts)]
    for a, b in sorted(cand, reverse=True):
        if _cut_ok(s, b, min([x for x, _ in cuts if x >= b] + [len(s)])):
            cuts.append((a, b))
    out, start = [], 0
    for a, b in sorted(cuts):
        out.append((start, a))
        start = b
    out.append((start, len(s)))
    return [(x, y) for x, y in out if s[x:y].strip()]


def _mask_attr_unknown(s: str, words: tuple = ("결함",)) -> str:
    """`words` 뒤에서 절 끝까지의 속성 판독 불가를 같은 길이의 빈칸으로 가린다 — 남은 부정만 부재로 읽는다."""
    out = s
    for a, b in _defect_clause_spans(s):
        c = s[a:b]
        hits = [m.start() for w in words for m in re.finditer(re.escape(w), c) if ATTR_UNKNOWN.match(c, m.end())]
        if hits:
            k = a + min(hits)
            out = out[:k] + " " * (b - k) + out[b:]
    return out


def _none_at_all(tail: str, at: int) -> bool:
    """수 뒤의 '도 · 조차 + 부정'('균열이 1개도 없다')이 부재인가 — 그 부정이 토막(tail 은 토막 안)을 평서로 끝낼 때만. 인용 · 이중
    부정 · 조건 · 추정('1개도 없다고' · '1개도 없는 것은 아니다' · '1개도 없다면' · '1개도 없을 것으로')이면 아니다(고치기 전처럼 그
    수의 언급이다)."""
    m = _NONE_AT_ALL.match(tail, at)
    return bool(m) and bool(re.fullmatch(r"(?:다|는다|았다|었다|습니다)?[\s,]*", tail[m.end():]))


def _zone_marked(s: str, a: int, b: int) -> str:
    """s[a:b] 의 구역을 § 로 바꾼 글 — 정형 부재 주장 · 긍정 서술 틀을 맞대는 꼴."""
    t, last = [], a
    for zs, ze, _ in zone_spans(s):
        if a <= zs and ze <= b:
            t += [s[last:zs], "§"]
            last = ze
    return "".join(t) + s[last:b]


def _canonical_absence(s: str, a: int, b: int) -> str | None:
    """토막 s[a:b] 가 정형 부재 주장이면 그 꼴(bare · global · zoned), 아니면 None(`_CANON`)."""
    t = _zone_marked(s, a, b)
    return next((k for k, pat in _CANON.items() if pat.fullmatch(t)), None)


def _plain_before(s: str, ba: int, a: int, pieces: list) -> bool:
    """바탕 절에서 토막 s[a:…] 앞의 토막이 모두 엄격한 긍정 서술인가(`_POSITIVE_PLAIN`) — 그 구역은 처소 주제가 아니라 그 서술의
    자리다('아래쪽 오른쪽에 기공 1개가 표시되어 있다, 균열은 없다'). 띠 · 높이 · 열이 있거나 틀 · 관형절이 끼면 아니다."""
    if regions_in(s[ba:a]):
        return False
    return all(_POSITIVE_PLAIN.fullmatch(_zone_marked(s, x, y)) for x, y in pieces if ba <= x < a)


def _closes(s: str, b: int, end: int) -> bool:
    """다음 토막(s[b:end])이 앞 부재 주장을 닫는가 — 엄격한 긍정 서술 틀(`_POSITIVE`)에 통째로 맞는 새 절일 때만('균열은 없다,
    아래쪽 오른쪽에 기공 1개가 표시되어 있다'). 관형절('…기공 1개가 있는 아래쪽 오른쪽 칸에서도 보이지 않는다'), 서술 없이 끝나는 토막
    ('…, 기공 1개가 있는 위쪽 왼쪽 칸에는'), 되받는 말('…칸도 같다'), 한정 · 예외('…칸에 한한 이야기다' · '…칸을 제외한 이야기다')은
    앞 부재의 범위를 잇는다."""
    return bool(_POSITIVE.fullmatch(_zone_marked(s, b, end)))


def _new_clause(s: str, x: int, y: int) -> bool:
    """토막 s[x:y] 가 제 주어를 가진 긍정 서술의 새 절인가 — '…칸에는 슬래그혼입 1개뿐이다' · '…칸에는 기공 1개만 표시된다'. 범위를
    한정하는 말(`_LIMIT`)이 있거나, 이름이 관형절 안('기공 1개가 있는 …칸')뿐이거나, 부정이 있거나 '다' 로 끝나지 않으면 아니다."""
    t = s[x:y]
    if _LIMIT.search(t) or NEG.search(t) or not re.search(r"다\s*$", t):
        return False
    return any(not _REL.match(t, m.end()) for m in _SUBJECT.finditer(t))


def _scope_end(s: str, a: int, b: int, base: list, pieces: list, n: str | None = None, kind: str | None = None) -> int:
    """부재 주장(s[a:b])이 범위를 읽는 끝 — 다음 토막이 엄격한 긍정 서술(`_closes`)이거나 제 주어를 가진 새 절(`_new_clause`)이면
    토막 끝, 아니면 바탕 절 끝('…, 기공 1개가 있는 위쪽 왼쪽 칸에는.' · '…칸에 한한 이야기다' 는 범위를 잇는다). 맨 · 전역 부재가
    아닌데 닫는 토막이 '…칸에는 기공 1개만 있다'(`_ONLY`)이면 그 칸에 다른 결함이 없다는 말이라 그 토막 끝까지 잇는다 — 그 토막이
    긍정한 이름(n)이면 잇지 않는다."""
    bb = next((y for x, y in base if x <= a < y), len(s))
    nxt = next(((x, y) for x, y in pieces if b <= x < bb), None)
    if not (nxt and (_closes(s, nxt[0], nxt[1]) or _new_clause(s, nxt[0], nxt[1]))):
        return bb
    if kind not in ("bare", "global") and _ONLY.search(s, nxt[0], nxt[1]) and not (n and n in s[nxt[0]:nxt[1]]):
        return nxt[1]
    return b


def _absence_scope(s: str, a: int, b: int, base: list, pieces: list, n: str | None = None) -> tuple[int, int]:
    """토막 s[a:b] 의 부재 주장이 범위(구역 · 영역)를 읽는 자리 — 시작과 끝을 한 규칙으로 정한다. 제 토막에서 시작하는 부재(정형
    부재 주장 — 제 구역 · 전역어가 있거나 앞 토막이 모두 엄격한 긍정 서술(처소 주제 포함)인 맨 부재 — 또는 바탕 절 첫 토막의
    부재)는 끝을 `_scope_end` 로 읽는다: 새 절이 닫으면 거기까지('균열은 없다, 아래쪽 오른쪽 칸에는 슬래그혼입 1개뿐이다' 의 칸은
    균열의 범위가 아니다), 한정하는 말이면 잇는다('균열은 보이지 않는다, 기공 1개가 있는 위쪽 왼쪽 칸에는.'). 앞 칸을 물려받는
    부재('…, 기공도 없다' · '…, 거기에 균열은 없다')와 전칭 부분 부정은 시작 · 끝 모두 고치기 전처럼 바탕 절이다."""
    ba, bb = next(((x, y) for x, y in base if x <= a < y), (0, len(s)))
    if _PARTIAL.search(_zone_marked(s, a, b)):
        return ba, bb                                               # 전칭 부분 부정 — 고치기 전처럼 바탕 절
    kind = _canonical_absence(s, a, b)
    if kind == "global" or (kind == "bare" and _plain_before(s, ba, a, pieces)) or kind == "zoned" or a == ba:
        return a, _scope_end(s, a, b, base, pieces, n, kind)        # 제 토막에서 — 새 절이 닫으면 거기까지, 한정하는 말이면 잇는다
    return ba, bb                                                   # 앞 칸을 물려받는 부재는 시작 · 끝 모두 고치기 전처럼 바탕 절


def grand_counts(s: str) -> list[tuple[tuple[int, int], int]]:
    """'결함 N개' · '결함은 모두 N개' · 'N개의 결함' — 이름 없이 결함 전체의 수를 말하는 언급."""
    out = []
    for m in GRAND.finditer(s):
        g = next(x for x in m.groups() if x)
        out.append((m.span(), int(g) if g.isdigit() else KO[g]))
    return out


def count_of(m: re.Match) -> tuple[int, bool]:
    g = m.group(2) or m.group(3) or m.group(4)
    return (int(g) if g.isdigit() else KO[g]), bool(m.group(1))


def defect_grounded(task: dict, text: str) -> list[str]:
    counts, lzones, lzc = locked_of(task)
    locked_all = set().union(*lzones.values()) if lzones else set()
    sents = [s.rstrip(".?!") for s in P.sentences(text)]
    said: set = set()
    strict: set = set()                                             # 수 · 긍정 서술로 언급된 이름(속성 판독 불가만이 아닌)
    absent_names: set = set()                                       # 영상 전체의 부재라고 밝힌 이름(범위가 빈 부재 주장)
    mzones: dict[str, set] = {}
    # 언급은 문장 순서와 무관하게 모두 모아 둔다 — 뒤 문장이 앞 문장을 덮지 않는다. 같은 항목의 같은 값은 되풀이, 다른 값은 모순
    zone_st: dict[tuple, list] = {}                                 # (이름, 구역) → 수들
    group_st: dict[tuple, list] = {}                                # (이름, 구역 묶음) → 수들 — 짝지을 수 없는 꼴 · 수 생략 꼴
    total_st: dict[str, list] = {}                                  # 이름 → 총수들(요약 · 구역 없는 수)
    grand_st: list = []                                             # 결함 전체의 수들
    out = []
    for s in sents:
        names_here = [n for n in NAMES if n in s]
        z, rest = zones_in(s)
        grands = grand_counts(s)
        grand_st += [k for _, k in grands]
        if not names_here:
            bare_ks = [(m, int(g) if (g := m.group(1) or m.group(2)).isdigit() else KO[g]) for m in BARE_COUNT.finditer(s)
                       if not any(a <= m.start() < b for (a, b), _ in grands)]
            if bare_ks:
                zl = list(dict.fromkeys(zz for _, ze, zz in zone_spans(s) if not _EMPTY.match(s, ze)))
                ks = [k for _, k in bare_ks]
                each = _per_zone_list(s, bare_ks[-1][0].start(), bare_ks[-1][0].end())
                if any(_per_zone(s, m.start(), m.end()) for m, _ in bare_ks[:-1]):
                    each = ()                                        # 배분 수가 둘 이상인 꼴은 펼치지 않는다
                ant = sorted(said - (((said - strict) & absent_names) - set(mzones)))   # 밝힌 부재의 속성 판독 불가는 앞 이름이 아니다
                if zl and len(ant) == 1:                            # '위쪽 왼쪽에 2개, 아래쪽 오른쪽에 1개가 있다' — 앞의 한 이름의 구역별 수
                    mzones.setdefault(ant[0], set()).update(zl)
                    if len(each) >= 2:                              # '…와 …에 각각 1개씩' — 그 칸들만 칸마다 그 수, 남은 수는 남은 칸에
                        for zz in each:
                            zone_st.setdefault((ant[0], zz), []).append(ks[-1])
                        if ks[:-1]:
                            add_zone_counts(ant[0], ks[:-1], [zz for zz in zl if zz not in each], zone_st, group_st, total_st)
                    else:
                        add_zone_counts(ant[0], ks, zl, zone_st, group_st, total_st)
                elif zl:
                    out.append("unattributed_count")                # 구역별 수인데 앞 이름이 없거나 여럿 — 누구의 수인지 정해지지 않는다
                elif not SUBSET.search(s[:bare_ks[0][0].start()]):  # '그중 1개는' 같은 부분 수는 읽지 않는다
                    if len(ant) == 1:
                        total_st.setdefault(ant[0], []).append(sum(ks))
                    else:
                        grand_st.append(sum(ks))                     # 이름이 여럿(또는 없음)이면 결함 전체의 수
            reg = regions_in(s)
            if NEG.search(_mask_attr_unknown(s)) and "결함" in s:   # 부재문 — 그 영역에 잠긴 결함이 없어야 한다('결함의 모양은 알 수 없다' 는 아니다)
                if not ABSENCE_EXEMPT.search(s):
                    if (z | reg) & locked_all:
                        out.append("absence_contradicts_locked")
                elif (but := _excepted(s)) is not None and _excepted_absence(z | reg, lzones, but):
                    out.append("absence_contradicts_locked")        # 예외어 — 예외로 든 칸 · 이름을 뺀 나머지로 대조한다
            elif z - locked_all:                                    # 이름 없는 관찰 — 칸이 잠긴 칸 안에
                out.append("observation_zone_outside_locked")
            continue
        # 이름이 모두 예외로만 든 부재문('기공을 뺀 위 왼쪽 칸에는 결함이 없다')은 이름 없는 부재문처럼 대조한다 — 그 이름은 아래에서
        # 긍정 · 부재 어느 쪽도 아니다
        but = _excepted(s) if NEG.search(_mask_attr_unknown(s)) and "결함" in s and ABSENCE_EXEMPT.search(s) else None
        if but is not None and all(any(a <= m.start() < b for a, b in but[2]) for n in names_here
                                   for m in re.finditer(re.escape(n), s)) and _excepted_absence(z | regions_in(s), lzones, but):
            out.append("absence_contradicts_locked")
        # 이름이 든 부재 주장('균열은 없다' · '위쪽 왼쪽에는 균열이 없다')은 버리지 않고 사실로 남긴다 — 잠긴 입력과 양립해야 한다.
        # 이름 자리마다 긍정 언급과 부재 주장을 가른다. 수가 붙은 이름 · 속성 판독 불가가 따르는 이름(토막 끝까지)은 긍정 언급이고, 수
        # 없이 뒤에 부정이 따르는 이름('균열은 없다' · '균열이 1개도 없다')은 부재 주장이다 — 주어로 쓴 이름은 제 토막에서, 다른 언급은
        # 고치기 전처럼 바탕 절에서 뒤의 부정을 본다. 부재의 범위는 `_absence_scope`. 이름 없는 토막은 새로 해석하지 않는다
        clauses = _defect_clause_spans(s)
        # 속성 판독 불가(이름 뒤에서 절 끝까지)는 부재가 아니라 그 이름의 언급이다. 같은 문장의 다른 절에 정형 부재 주장이 아닌 부정이나
        # 띠 · 높이 소견이 있으면 받지 않고 문장 전체를 고치기 전 절로 읽는다 — 그 절의 주어가 앞 이름일 수 있고, 띠 대조는 '없' 이 든
        # 문장을 묻지 않는다
        unknown = {a + m.start() for a, b in clauses for n in names_here for m in re.finditer(re.escape(n), s[a:b])
                   if not re.match(COUNT_AFTER_NAME, s[a + m.end():b]) and ATTR_UNKNOWN.match(s[a + m.end():b])}
        if unknown and any(not any(a <= p < b for p in unknown) and (regions_in(s[a:b]) or (
                _canonical_absence(s, a, b) is None and NEG.search(s[a:b]) and not ABSENCE_EXEMPT.search(s[a:b])
                and not _attr_unknown_clause(s[a:b]))) for a, b in clauses):
            clauses, unknown = _base_clause_spans(s), set()
        base = _base_clause_spans(s)
        positive: set = set()
        firm: set = set()                                           # 수 · 긍정 서술로 언급된 이름(이 문장)
        absent: list = []                                           # (자리, 이름, 토막) — 부재로 읽은 이름 자리
        for ba, bb in base:
            for n in [n for n in names_here if n in s[ba:bb]]:
                for m in re.finditer(re.escape(n), s[ba:bb]):
                    p, e = ba + m.start(), ba + m.end()
                    a, b = next(((x, y) for x, y in clauses if x <= p < y), (ba, bb))
                    own = _ASSERTED.match(s, e) and not any(NEG.search(s, x, y) and _canonical_absence(s, x, y) is None
                                                            for x, y in clauses if b <= x < bb)
                    tail = s[e:b] if own else s[e:bb]                # 뒤 토막의 부정이 정형 부재 주장의 것이 아니면 생략된 주어일 수 있다
                    cm = re.match(COUNT_AFTER_NAME, s[e:b])
                    if p in unknown:
                        positive.add(n)
                    elif (cm and not _none_at_all(s[e:b], cm.end())) or not NEG.search(tail):
                        positive.add(n)
                        strict.add(n)
                        firm.add(n)
                    elif not ABSENCE_EXEMPT.search(s, ba, scope_hi := _absence_scope(s, a, b, base, clauses, n)[1]):   # 예외어는 범위 끝까지
                        absent.append((p, n, (a, b), None))
                    elif (but := _excepted(s, ba, scope_hi)) is not None and n not in but[0]:
                        absent.append((p, n, (a, b), but[1]))      # 예외로 든 칸을 뺀 나머지가 범위('위쪽 왼쪽 칸을 뺀 칸에는 균열이 없다')
        absent_at = {p for p, _, _, _ in absent}                   # 부재로 읽은 이름의 자리(문장 안)
        for p, n, (a, b), but_zones in absent:
            lo, hi = _absence_scope(s, a, b, base, clauses, n)
            claimed = {zz for zs, _, zz in zone_spans(s) if lo <= zs < hi} | regions_in(s[lo:hi])
            if but_zones is not None:
                claimed -= but_zones
                if but_zones and not claimed:
                    claimed = ALL - but_zones                       # 뺀 칸 밖 모두
            if not claimed:
                absent_names.add(n)                                 # 영상 전체의 부재라고 밝힌 이름
            if (not claimed and n in counts) or (claimed and (lzones.get(n, set()) & claimed)):
                out.append("absence_contradicts_locked")
        names_here = [n for n in names_here if n in positive]
        if not names_here:
            continue                                                # 부재 주장만 있는 문장 — 이름을 세지 않는다
        name_pos = sorted((m.start(), m.end(), n) for n in names_here for m in re.finditer(re.escape(n), s))
        for _, _, n in name_pos:
            said.add(n)
        spans = zone_spans(s)
        canon = [(a, b) for a, b in clauses if _canonical_absence(s, a, b) == "zoned"]
        target: list = [None] * len(spans)                          # 구역마다 붙일 이름들 — 이어진 구역 묶음은 마지막 구역을 따른다
        for i in range(len(spans) - 1, -1, -1):
            zs, ze, z = spans[i]
            after = s[ze:]
            if i + 1 < len(spans) and re.fullmatch(r"\s*(?:와|과|및|,)\s*", s[ze:spans[i + 1][0]]):
                target[i] = target[i + 1]
                continue
            if any(a <= zs < b for a, b in canon) or _EMPTY.match(s, ze):
                target[i] = []                                      # 정형 부재 주장('아래쪽 오른쪽에는 균열이 없다')의 칸은 부재의 범위다
                continue
            nxt = re.match(r"\s*(?:(?:구역|칸)\s*)?(?:에|의)\s*(?:는\s*)?", after)
            follow = []
            if nxt:
                stop = (spans[i + 1][0] - ze) if i + 1 < len(spans) else len(after)
                seg = after[nxt.end():stop].split(",")[0]            # 다음 구역 앞에서 닫는다 — '균열 1개와 아래쪽 오른쪽에 기공' 의 기공은 뒤 구역의 것
                if any(seg.startswith(n) for n in names_here):  # 'ZONE에 · 의 이름 …' — 바로 뒤의 이름들에
                    # 속성 판독 불가로만 언급한 이름은 칸 바로 뒤가 아니면 끼지 않는다 — '위쪽 왼쪽 칸에는 기공 1개가 있고 균열의 길이는 알 수
                    # 없다' 의 칸은 균열의 것이 아니다(고치기 전처럼)
                    hits = sorted((m.start(), m.end(), n) for n in names_here for m in re.finditer(re.escape(n), seg))
                    lead, last = set(), None                # 칸 바로 뒤에서 와 · 과 · 및 · 쉼표로 이어진 이름 나열
                    for at, e, n in hits:
                        if at == 0 or (last is not None and re.fullmatch(_LIST_GAP, seg[last:at])):
                            lead.add(at)
                            last = e
                        elif last is not None:
                            last = None
                    # 나열 밖의 이름은 수 · 긍정 서술로 언급된 것만, 그리고 제 칸이 바로 뒤에 따로 붙은 이름은 빼고('위 왼쪽에는 기공 1개가
                    # 있고 균열 1개가 아래 오른쪽에 …' 의 균열은 뒤 칸의 것)
                    base_at = ze + nxt.end()
                    follow = [n for at, e, n in hits if at in lead or (n in firm and not (
                        i + 1 < len(spans) and re.fullmatch(_NAME_TAIL, s[base_at + e:spans[i + 1][0]])))]
            if follow:
                target[i] = follow
            else:                                                   # 'NAME … ZONE' — 앞의 가장 가까운 이름에
                before = [(st, e, n) for st, e, n in name_pos if st < zs]
                prev = [n for _, _, n in before]
                target[i] = prev[-1:] if prev else []
                if before and re.fullmatch(_NAME_TAIL, s[before[-1][1]:zs]):
                    # 'X 1개와 Y 1개가 ZONE에' · 'X과 Y은 ZONE에' — 칸 바로 앞의 이름 나열은 모두 그 칸에 있다
                    # — 앞 이름이 이미 제 칸을 가졌으면(앞 칸 바로 뒤의 이름: '위 왼쪽에 기공 1개, 균열 1개가 ZONE에') 거기서 멈춘다
                    j = len(before) - 1
                    while (j > 0 and re.fullmatch(_LIST_GAP, s[before[j - 1][1]:before[j][0]])
                           and not _zone_led(s, before[j - 1][0], spans)):
                        j -= 1
                    target[i] = list(dict.fromkeys(n for _, _, n in before[j:]))
                if prev and re.search(r"(?:와|과)\s*(?:함께|같이)\s*$", s[max(0, zs - 24):zs]):
                    # '균열의 모양은 알 수 없지만 기공 1개와 함께 ZONE에 있다' — 동반 꼴의 칸은 앞의 속성 판독 불가 이름에도
                    target[i] = list(dict.fromkeys([n for st, _, n in name_pos if st < zs and n not in firm] + target[i]))
        for (zs, ze, z), names_for in zip(spans, target):
            for n in names_for or []:
                mzones.setdefault(n, set()).add(z)
        counted = [(n, m) for n in NAMES for m in re.finditer(re.escape(n) + COUNT_AFTER_NAME, s)]
        named = {n: [m for k, m in counted if k == n and m.start() not in absent_at] for n in names_here}   # '1개도 없다' 는 수가 아니다
        taken = [m.span() for _, m in counted] + [sp for sp, _ in grands]
        bare: dict[str, list] = {}                                  # 'NAME 2개가 A에, 1개가 B에' — 이름을 생략한 수는 앞 이름에
        for m in BARE_COUNT.finditer(s):
            if any(a <= m.start() < b for a, b in taken):
                continue                                            # 이름 붙은 수 · 결함 전체의 수의 일부
            g = m.group(1) or m.group(2)
            k, summ = int(g) if g.isdigit() else KO[g], bool(SUMMARY.search(s[max(0, m.start() - 4):m.start()]))
            after = re.match(r"\s*의\s*(" + "|".join(map(re.escape, NAMES)) + ")", s[m.end():])
            if after:                                               # 'N개의 NAME' — 뒤의 이름의 수
                bare.setdefault(after.group(1), []).append((m.start(), k, summ, _per_zone_list(s, m.start(), m.end()), False))
                continue
            if summ:
                if len(names_here) == 1:                            # '…, 모두 3개다' — 이름이 하나면 그 이름의 총수
                    total_st.setdefault(names_here[0], []).append(k)
                continue                                            # '합쳐 3개' — 여러 이름을 합친 총수는 이름에 세지 않는다
            prev = [n for st, _, n in name_pos if st < m.start()]
            if prev:
                at = max(st for st, _, n in name_pos if st < m.start() and n == prev[-1])
                bare.setdefault(prev[-1], []).append((m.start(), k, False, () if _in_name_list(s, at, prev[-1])
                                                      else _per_zone_list(s, m.start(), m.end()), False))
        for n in names_here:
            found = [(m.start(), *count_of(m), () if _in_name_list(s, m.start(), n) else _per_zone_list(s, m.start(), m.end(), m.start()),
                      True) for m in named[n]] + bare.get(n, [])
            found.sort()
            ks, per = [], []
            for pos, k, is_sum, each, own in found:
                # 요약어가 수보다 앞에 있는 문장에서 이름의 수가 하나뿐이면 그 수가 총수다('결함은 모두 3개로, 기공 3개가 …').
                # 둘 이상이면 '모두 · 총' 이 붙은 수만 총수다. '기공 2개는 모두 위쪽 왼쪽에' 의 '모두' 는 수 뒤라 총수 표시가 아니다
                if is_sum or (SUMMARY.search(s[:pos]) and len(found) == 1 and not each):
                    total_st.setdefault(n, []).append(k)
                else:
                    ks.append(k)
                    per.append(each)
            # 구역별 수 — 구역 하나에 수 하나, 수가 구역 개수와 같으면 하나씩, 수와 구역이 같은 개수면 차례대로 짝짓는다.
            # 그 밖(수 생략 꼴 · 수 셋에 구역 둘 같은)은 '그 구역 묶음의 합' 언급이다. 구역이 없으면 이름의 총수 언급이다.
            zl = []
            for (zs, ze, z), names_for in zip(spans, target):
                if n in (names_for or []) and z not in zl:
                    zl.append(z)
            if sum(1 for e in per if e) == 1 and len(per[-1]) >= 2:
                # 'A 와 B 에 각각 N개씩' — 그 수 바로 앞의 칸들만 칸마다 N(배분). 앞 수는 남은 칸의 수, 남은 칸이 없으면 총수
                mzones.setdefault(n, set()).update(per[-1])
                for zz in per[-1]:
                    zone_st.setdefault((n, zz), []).append(ks[-1])
                if ks[:-1]:
                    add_zone_counts(n, ks[:-1], [zz for zz in zl if zz not in per[-1]], zone_st, group_st, total_st)
                continue
            if ks and zl and len(ks) == len(zl) + 1 and ks[0] == sum(ks[1:]):
                total_st.setdefault(n, []).append(ks[0])            # '기공 3개가 A에 2개, B에 1개' — 앞 수가 뒤 수들의 합이면 총수
                ks = ks[1:]
            elif ks and len(zl) >= 2 and len(ks) == len(zl) and re.search(r"나머지", s) and ks[0] > sum(ks[1:]):
                total_st.setdefault(n, []).append(ks[0])            # '3개 가운데 2개는 A에, 나머지는 B에' — 나머지 = 총수 − 앞 수들
                ks = ks[1:] + [ks[0] - sum(ks[1:])]
            if ks:
                add_zone_counts(n, ks, zl, zone_st, group_st, total_st)
        # 띠 소견 — '결함은 X 띠를 따라' 는 (그 이름의) 잠긴 칸이 모두 그 띠 안이라는 말이다. '대부분 · 일부 · 주로' 면 묻지 않는다
    for s in sents:
        reg = regions_in(s)
        if not reg or re.search(r"(?:대부분|일부|주로|없)", s):
            continue
        subj = [n for n in NAMES if re.search(re.escape(n) + r"(?:은|는|이|가)\s", s)]
        if subj:
            if any(n in lzones and not lzones[n] <= reg for n in subj):
                out.append("band_contradicts_locked")
        elif re.search(r"결함(?:은|이|들은)\s", s):
            # '용입불량을 뺀 결함은 모두 …' — 예외로 든 이름을 뺀 나머지의 잠긴 칸이 그 띠 안이어야 한다(건너뛰지 않는다)
            but = {n for m in _BAND_EXCEPT.finditer(s) for n in NAMES if n in m.group(0)}
            if not set().union(set(), *(z for n, z in lzones.items() if n not in but)) <= reg:
                out.append("band_contradicts_locked")
    # 부재라고 밝힌 이름의 속성 판독 불가('균열은 표시되어 있지 않아서 균열의 길이도 알 수 없다')는 그 이름의 언급이 아니다
    said -= ((said - strict) & absent_names) - set(mzones)          # 칸이 붙은 속성 판독 불가('위쪽 왼쪽 칸의 균열은 …')는 그 칸의 언급이다
    if said != set(counts):
        out.append("names_mismatch")
    out += statement_checks(counts, lzc, mzones, zone_st, group_st, total_st, grand_st)
    for n, zs in mzones.items():
        if n in lzones and not zs <= lzones[n]:
            out.append("zone_outside_locked")
            break
    for n, zs in lzones.items():
        if not zs <= mzones.get(n, set()):
            out.append("zone_not_described")
            break
    return sorted(set(out))


def statement_checks(counts: dict, lzc: dict, mzones: dict, zone_st: dict, group_st: dict, total_st: dict,
                     grand_st: list) -> list[str]:
    """모아 둔 언급의 대조. 뒤가 앞을 덮지 않는다 — 같은 항목에 다른 값이 둘 이상이면 모순(statement_conflict)이고,
    명시된 구역별 수는 다른 문장에 무엇이 있든 잠긴 값과 맞아야 한다."""
    out = []
    for vals in (*zone_st.values(), *group_st.values(), *total_st.values(), grand_st):
        if len(set(vals)) > 1:
            out.append("statement_conflict")
    for (n, z), vals in zone_st.items():                             # 수가 적혔으면 반드시 맞아야 한다
        if any(v != lzc.get((n, z), 0) for v in vals):
            out.append("zone_count_mismatch")
    for (n, zs), vals in group_st.items():                           # 구역 묶음의 합도 잠긴 값의 합과
        if any(v != sum(lzc.get((n, z), 0) for z in zs) for v in vals):
            out.append("zone_count_mismatch")
        if all((n, z) in zone_st for z in zs) and any(v != sum(zone_st[(n, z)][0] for z in zs) for v in vals):
            out.append("zone_sum_mismatch")
    for n in sorted(set(counts) | set(total_st) | {n for n, _ in zone_st} | {n for n, _ in group_st}):
        told = set(total_st.get(n, [])) | {v for (m, zs), vals in group_st.items() if m == n and zs == mzones.get(n) for v in vals}
        zkeys = [z for (m, z) in zone_st if m == n]
        zsum = sum(zone_st[(n, z)][0] for z in zkeys) if zkeys else None
        if zkeys and set(zkeys) == mzones.get(n, set()):              # 구역별 수가 모든 구역을 덮으면 그 합이 총수다
            if told and told != {zsum}:
                out.append("zone_sum_mismatch")
            told = told | {zsum}
        elif not told and zkeys:
            told = {zsum}                                            # 일부 구역의 수만 있으면 그 합(모자라면 총수 불일치)
        if n in counts and (not told or told != {counts[n]}):
            out.append("count_mismatch")
    if grand_st and set(grand_st) != {sum(counts.values())}:
        out.append("count_mismatch")
    return out


def contract(task: dict, text: str) -> list[str]:
    """v6 생성 계약의 고정 문장 — 정상 행은 두 문장이 정해진 문장, 결함 행은 마지막 문장이 정해진 제한 문장."""
    sents = [s if s.endswith((".", "?", "!")) else s + "." for s in P.sentences(text)]
    if P.row_class(task) == "defect_free":
        ok = len(sents) == 2 and sents[0] in G.NORMAL_ABSENCE_FORMS and sents[1] in G.NORMAL_LIMITATION_FORMS
        return [] if ok else ["normal_not_contract_form"]
    return [] if sents and sents[-1] in G.LIMITATION_FORMS else ["limitation_not_contract_form"]


def attributes(task: dict, text: str) -> list[str]:
    """주석 밖 시각 속성의 긍정 서술(크고 둥근 · 밝은 …) — '알 수 없다' 류의 한계 서술은 세지 않는다."""
    if P.row_class(task) != "defect":
        return []
    bad = [s for s in P.sentences(text) if ATTR.search(s + " ") and not ATTR_OK.search(s)]
    return ["unannotated_attribute"] if bad else []


def grounded_codes(task: dict, text: str) -> list[str]:
    return defect_grounded(task, text) if P.row_class(task) == "defect" else []


def contract_codes(task: dict, text: str) -> list[str]:
    return contract(task, text) if P.row_class(task) is not None else []


def attribute_codes(task: dict, text: str) -> list[str]:
    return attributes(task, text)


# ---------------------------------------------------------------- 정상 행의 음성 패턴(7판 보강 4)
# 정상 행의 수락 = 생성 계약 일치(엄격) ∧ 음성 패턴 없음. 아래는 계약 밖 글도 잡기 위한 보강이다 — 결함 행에는 걸지 않는다.

_V_POS = r"(?:(?<!불)합격|(?<!불)안전|양호|(?<!부)적합|건전|우수|문제(?:가|는)?\s*없)"
"""긍정 판정어. '불합격 · 부적합' 은 뺀다(방향이 반대). '문제(가) 없' 도 양호 단정이다."""
_V_LIMIT = r"(?:(?<!불)합격|(?<!불)안전|합부|양호|(?<!부)적합)"
"""제한 문장이 다루는 판정어 — 합부(합격 · 불합격)도 든다."""
_NEGATION = re.compile(r"(?:않|아니|아닙|아님|아닌|없|못|모르|모른|어렵)")
_CLAUSE_SPLIT = re.compile(r"(?:지만|으나|는데|그러나|그런데|(?<=[없않])고)[,\s]+")
_INVERTED = re.compile(r"(?:않|아니)\S*(?:다고|라고)\s*(?:볼|할|말할|여길)\s*(?:이유|근거|까닭|필요)\S*\s*(?:없|않)")
_WRONG_DIR = re.compile(r"(?:불합격|부적합|불량|위험)\S*\s*(?:을|를|이|가)?\s*(?:뜻|의미|보장|나타내|말하)")
_VERDICT_LABEL = re.compile(r"(?:판정|결론|결과|평가)\s*[:：]?\s*(?:은|는)?\s*(?:합격|안전|양호|적합)")
_CAUSE = re.compile(r"(?:때문(?:이다|입니다|이에요|이라)|덕분|덕택|탓(?:이다|입니다)|원인(?:은|이))")
_KO_NUM = r"(?:영|공|하나|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|스무)"
_NUMBER_WORD = re.compile(r"(?<![가-힣])" + _KO_NUM + r"\s*(?:개|건|곳|군데|가지)(?=[이가은는의도다만,.\s]|$)"
                          r"|(?:영|공|일|이|삼|사|오|육|칠|팔|구)점(?:영|공|일|이|삼|사|오|육|칠|팔|구)+"
                          r"|밀리미터|센티미터|마이크로미터|(?<![가-힣])미터|㎜|(?<![A-Za-z])(?:mm|cm|μm|px)(?![A-Za-z])|픽셀")
_INSTRUCTION = re.compile(r"(?:마라|마시오|하라|하시오|해라|말\s*것|쓸\s*것|적을\s*것)(?=[.\s]|$)"
                          r"|규칙\s*[:：]|형식\s*[:：]|다음\s*(?:형식|규칙|지시|조건)|지시문|프롬프트|출력\s*형식|응답\s*형식|(?i:json)")
_POSITION = re.compile(r"(?:위쪽|아래쪽|왼쪽|오른쪽|왼편|오른편|가운데|중앙|가장자리|모서리|구석|상단|하단|좌측|우측|윗부분|아랫부분"
                       r"|(?<![가-힣])위(?=\s*(?:왼|오른|가운데|쪽))|(?<![가-힣])아래(?=\s*(?:왼|오른|가운데|쪽))|띠|구역|(?<![가-힣])칸(?![가-힣]))")
_DEFECT_WORD = re.compile(r"(?:크랙|갈라짐|기포|구멍|언더컷|오버랩|스패터|터짐|이물|함몰|용입\s*부족|용입불량|슬래그|텅스텐)")
_DEFECT_EXISTS = re.compile(r"결함(?:이|은|들이|들은)?\s*(?:\S+\s+)?(?:보인다|보입니다|보이는|있다|있습니다|있는|관찰|확인된다|확인됩니다|나타난다|존재)")


def _normal_clauses(sentence: str) -> list[str]:
    """정상 행 문장의 절 — 판정어와 그 부정이 같은 절에 있는지 본다. 결함 행의 절(`_defect_clause_spans`)과 다른 함수다."""
    return [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]


def _negated_after(clause: str, start: int) -> bool:
    return bool(_NEGATION.search(clause, start))


def normal_negative(task: dict, text: str) -> list[str]:
    """정상 행(`defect_free`)의 음성 패턴. 사유 코드 목록 — 빈 목록이 통과. 결함 행은 빈 목록."""
    if P.row_class(task) != "defect_free":
        return []
    sents = [s.strip() for s in P.sentences(text) if s.strip()]
    out = []
    has_limit = False
    for s in sents:
        body = s.rstrip(".?!")
        if _INVERTED.search(body):
            out.append("normal_limitation_inverted")
        if _VERDICT_LABEL.search(body):
            out.append("normal_verdict_implied")
        if _CAUSE.search(body):
            out.append("normal_cause_asserted")
        if _NUMBER_WORD.search(body):
            out.append("normal_number_or_unit")
        if _INSTRUCTION.search(body):
            out.append("normal_instruction_leak")
        if _POSITION.search(body) or _DEFECT_WORD.search(body):
            out.append("normal_defect_or_position")
        for m in _DEFECT_EXISTS.finditer(body):
            if not _NEGATION.search(body, m.start()):
                out.append("normal_defect_or_position")
        for c in _normal_clauses(body):
            for m in re.finditer(_V_POS, c):            # 긍정 판정어 뒤에 같은 절의 부정이 없으면 단정이다
                if not _negated_after(c, m.end()):
                    out.append("normal_verdict_implied")
            for m in re.finditer(_V_LIMIT, c):          # 판정어 뒤에 부정이 따르는 절이 제한 문장이다(N5 의 넓은 꼴)
                if _negated_after(c, m.end()):
                    has_limit = True
        if _WRONG_DIR.search(body) and not re.search(_V_LIMIT, body):
            out.append("normal_limitation_wrong_direction")
    if not has_limit:
        out.append("normal_limitation_absent")
    return sorted(set(out))


def normal_codes(task: dict, text: str) -> list[str]:
    return normal_negative(task, text)
