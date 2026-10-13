# 구절 은행 v2 — 원문 QA 생성기의 입력 계약

2026-10-07. v1(`passage_bank_v1.*`)을 대체한다. v1 파일은 기록으로 남기고 고치지 않는다.
소비자는 생산 파이프라인의 원문 QA 생성과 `corpus/generate/run_cycle_corpus.py` · `run_pilot.py` 의 QA 단계다.
`run_cycle_corpus` · `run_pilot` 은 4절의 소비 경계만 거친다. 생산 파이프라인 쪽 연결은 그 구현이 정한다 — 이 은행이 강제하지 않는다.

**이 은행은 학습 투입 승인이 아니다.** 행의 `rights_status` 는 전부 `pending` 이고 사람이 본 행은 없다.
본문 은행에 소비 경계를 실제로 돌리면 지금 넘어가는 구절은 0 이다(`passage_bank_v2.capacity.json` 의 `passing_generation_gate_now`).

## 1. v1 에서 바뀐 것

| 무엇 | v1 | v2 |
|---|---|---|
| 원문 | 다섯 문서 | 다섯 문서 + 쪽 화면과 대조해 옮긴 두 문서의 14구절(TM 1-430 6 · MDPI 리뷰 8) |
| 유료 표준 수치 문장 | 이름 + mm · inch · ° F/C · % 꼴 | 하이픈 단위(`7-inch`) · 도 · 피트 · psi · 유니코드 분수(¾) · 두께 배수(`5t`), AASHTO · CSA · AS/NZS · DIN 이름, FHWA 의 "the Code" · 해설 조항(`clause C-…`) 을 더 센다. AWS 인용 변형(`(AASHTO/AWS, …)` · `(AWS, …c.)` · `as defined by AWS`) |
| 표의 절 | 앞 절을 부모로 붙였다(`2 / Table 8.5`) | IACS · ABS · W33 · FHWA 는 표 번호가 곧 절이다. ABS 표 안의 번호 줄(각주)은 절 머리로 읽지 않는다 |
| 쪽 범위 | 문장의 시작 쪽만 | 문장이 걸친 쪽 전체. 모든 구절의 모든 문장이 [시작 쪽, 끝 쪽] 텍스트 층 안에 있음을 확인했다 |
| 같은 본문 | 되풀이 허용 | 같은 지문의 구절은 처음 것만 남긴다(문서 안 · 문서 사이) |
| 계보 미확인 블록 | KR 물리 327~328(RT) | + KR 물리 328~329(UT, KS B 시험편 · 방법 인용) — `UT_lineage_unverified` |
| 소비 경계 | 다섯 칸 포인터 대조, 기록만 보면 통과 | 본문 말고 **모든 칸** 포인터 대조, 계보 표지 해제 · 위험 표지 확인을 따로 요구, 승인 용도의 꼴 검사, 은행 지문 고정 |
| 역사 자료 | — | `historical_only` 표지, 원인 · 방지 · 보수 · 기준 면 표지를 달지 않는다 |

## 2. 행

v1 의 칸(`passage_bank_v1.md` 2절)에 더해:

| 칸 | 뜻 |
|---|---|
| `schema_version` | 2 |
| `scope` | 문서(KR 은 부록)가 다루는 구조물 · 재료 · 검사 |
| `flags` | 목록. 3절 |
| `draft_passage_id` · `use_boundary` | 옮겨 온 구절만. 초안의 구절 번호와 쓰임 경계(예: 역사적 서술만) |
| `extraction.text_layer_check` | 옮겨 온 구절만. 그 쪽의 텍스트 층에 정규화 후 그대로 있으면 `exact`, 글자 · 숫자만 같으면 `alnum`, 없으면 `failed` |
| `locator.section_kind` | `clause` · `table` · `lead_in` · `curated` |

## 3. 원문 · 범위 · 표지

| 문서 | 판본 | 원문 sha256 | 넣은 물리 쪽 | 구절 |
|---|---|---|---|---|
| `KR-RULES-P2` | 2025 | `46c4ee18…552b` | 123~220, 264~276, 291~293, 307~331, 351~361 | 291 |
| `IACS47` | Rev.10 / Corr.1 2025 | `ac54b790…15a6` | 4~48, 51~67 | 87 |
| `ABS-PUB87-2007` | 2007 | `be55985f…0565` | 7~44 | 35 |
| `IACS-URW33-R1C1` | Rev.1 / Corr.1 2021 | `1edc750f…fee3` | 1~14 | 74 |
| `FHWA-HIF-19-088` | 2019-09, 2020-09 정오표 | `d317b8e9…4046` | 25~321 | 755 |
| `US-WAR-DEPT-TM-1-430-1942` | 1942 판(1943 Change 1 포함) | `7742fb0b…78ee` | 옮긴 구절 6 | 6 |
| `MDPI-APPLSCI-2020-10-1878` | Applied Sciences 10(5):1878 | `2ff799fa…288b` | 옮긴 구절 8 | 8 |

합계 1,256 구절이다. 옮긴 14구절은 텍스트 층 대조에서 `exact` 13 · `alnum` 1 이었다.

**빼는 규칙.** v1 과 같고, 1절의 유료 표준 규칙이 넓어졌다. 문장 단위로 덜어 낸 수는 FHWA 가
유료 표준 수치 41 · AWS 인용 101, KR 3, IACS47 1, W33 1 이다(`passage_bank_v2.build_report.json`).

**표지.** v1 의 표지에 더해:

| 표지 | 뜻 | 소비 경계 |
|---|---|---|
| `L2_lineage_unverified` · `UT_lineage_unverified` | KR 부록 2-7 의 RT 판정 블록(물리 327~328) · UT 판정 블록(물리 328~329)이 든다. 쪽 단위 표지라 같은 쪽의 표 12(표면 결함) 구절에도 붙는다 | 권리 결정 기록의 `cleared_lineage_flags` 에 그 표지가 있어야 넘긴다 |
| `historical_only` | 1942년 교범. 역사적 명칭 · 개념만 | 사람 검수의 `flags_checked` 에 있어야 넘긴다 |
| `cc_by_4_0_attribution_required` | 파생 QA 에 저자 · DOI · 라이선스 · 변경 사실 표시가 필요하다 | 경계는 보지 않는다 — 산출물 쪽의 의무다 |
| `paid_clause_reference_without_numeric` | FHWA 에서 D1.5 조항 · "the Code" 가 수치 없이 나온다 | 사람 검수의 `flags_checked` 에 있어야 넘긴다 |
| `text_layer_match_alnum_only` · `text_layer_cross_check_failed` | 옮긴 구절의 텍스트 층 대조 결과 | 실패는 `flags_checked` 에 있어야 넘긴다 |

사람 검수가 확인해야 하는 위험 표지는 `passage_bank.RISK_FLAGS`, 권리 쪽이 풀어야 하는 계보 표지는 `LINEAGE_FLAGS` 가 정본이다.

## 4. 소비 경계 — `corpus.parse.passage_bank.select_for_generation`

v1 의 검사(등록 · 본문 지문 · 권리 결정 · 사람 검수 · 유료 표준 문장)에 더해:

| 검사 | 사유 키 |
|---|---|
| 포인터 은행의 같은 행과 본문 말고 **모든 칸**이 같다 | `pointer_mismatch:<칸>` |
| 권리 기록의 `approved_uses` 는 문자열 목록이고 원소가 `qa_generation` · `rag_index` · `training_text` 가운데 있다 | `rights:approved_uses_malformed` |
| 행의 계보 표지가 권리 기록의 `cleared_lineage_flags` 에 있다 | `rights:lineage_not_cleared:<표지>` |
| 행의 위험 표지가 사람 검수 기록의 `flags_checked` 에 있다 | `review:flag_unchecked:<표지>` |
| 행 꼴 2판(`schema_version` 2, `passage_id` 문자열, 쪽 번호는 bool 이 아닌 정수, 역사 행에 현대 면 없음) | `shape:*` |

기록 파일은 같은 키가 두 번 있는 줄을 거부한다(`load_jsonl`). 권리 결정 기록과 사람 검수 기록은 은행 빌더가 만들지 않는다.

`run_cycle_corpus.load_passages` 는 포인터 · 본문 은행의 sha256 을 `passage_bank_v2.build_report.json` 의 값과 대조하고,
다르면 구절을 넘기지 않는다(포인터 은행을 바꿔 끼우는 길을 막는다). QA 레코드의 `source` 에 범위 · 표지 · 쓰임 경계를 싣는다.

## 5. 용량 — `passage_bank_v2.capacity.json`

| 구분 | 구절 | QA 상한(구절당 3) | 하한 10,000 까지 | 고유 구절 하한 3,334 까지 |
|---|---|---|---|---|
| 전체 | 1,256 | 3,768 | 6,232 | 2,078 |
| 역사 자료 제외 | 1,250 | 3,750 | 6,250 | 2,084 |
| `welding_general` 제외 | 757 | 2,271 | 7,729 | 2,577 |
| 결함 이름이 나오는 구절만 | 217 | 651 | 9,349 | 3,117 |

계보 묶음은 여섯이다. 가장 작은 묶음을 val 로 두면 train 은 1,250 구절이다. 판정추론 상한은 v1 과 같은 480 이다(허용치 표 v0 의
RT 11 · VT 1 조합 × 40). RT 11 조합은 전부 KR RT 블록(L2)이다. 반복으로 채우지 않는다.

## 6. 재현

```
python -m corpus.parse.build_passage_bank \
  --source KR-RULES-P2=corpus/parse/raw/KR-RULES-P2_2025.pdf \
  --source IACS47=corpus/parse/raw/IACS47_Rev10Corr1_2025.pdf \
  --source ABS-PUB87-2007=<ABS Pub 87 PDF> --source IACS-URW33-R1C1=<UR W33 Rev.1 Corr.1 PDF> \
  --source FHWA-HIF-19-088=<FHWA-HIF-19-088 PDF> \
  --source US-WAR-DEPT-TM-1-430-1942=<TM 1-430 PDF> --source MDPI-APPLSCI-2020-10-1878=<applsci-10-01878 PDF> \
  --curated <옮긴 구절 JSONL, sha256 0c0b6d80…8017> \
  --out-full <저장소 밖>/passage_bank_v2_full.jsonl \
  --out-pointer corpus/parse/passage_bank_v2.jsonl --report corpus/parse/passage_bank_v2.build_report.json
python -m corpus.parse.passage_bank_capacity --bank corpus/parse/passage_bank_v2.jsonl \
  --full-bank <저장소 밖>/passage_bank_v2_full.jsonl --registry corpus/parse/sources.yaml \
  --limits corpus/rules/limits_v0_pilot.csv --out corpus/parse/passage_bank_v2.capacity.json
```

원문 URL 은 `sources.yaml` 에 있다. pypdfium2 5.13.0 · pdfium 153.0.7999.0 에서 두 번 돌려 같은 바이트를 냈다.
구절 길이의 상한은 900자이고, 120자 미만 꼬리를 앞 구절에 붙이는 경우가 있어 실제 상한은 1,020자다.

## 7. 보증하지 않는 것

- 면 · 결함 · 관련성 표지가 맞다는 것. 키워드 규칙이고 사람이 보지 않았다. 표본에서 콘크리트 균열 · 프로펠러 주물 균열에 `crack` 이 붙은 예를 봤다.
- 유료 표준 유래 문장을 모두 걸렀다는 것. 규칙은 "이름 · 조항 참조 + 수치 · 단위가 한 문장" 과 출처 인용 꼴만 잡는다. 참조 없이 수치만 옮긴 문장, 다음 문장에 걸친 참조는 못 잡는다.
- 계보 미확인 블록이 둘뿐이라는 것. KR 의 다른 쪽 · 다른 문서의 계보는 따로 보지 않았다.
- 옮긴 14구절의 사실성 · 권리. 텍스트 층 대조는 글자가 있다는 확인일 뿐이다.
- 권리 · 사실. 권리는 사용자가, 사실은 사람 검수가 정한다.
