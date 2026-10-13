# 구절 은행 v1 — 원문 QA 생성기의 입력 계약

2026-10-07. 소비자는 생산 파이프라인(`corpus/production/`)의 원문 QA 생성과 `corpus/generate/run_cycle_corpus.py` 의
QA 단계다. 둘 다 이 은행을 4절의 소비 경계로 거른 뒤에만 쓴다.

**이 은행은 학습 투입 승인이 아니다.** 행의 `rights_status` 는 전부 `pending` 이고 사람이 본 행은 없다.
지금 생성기로 넘어갈 수 있는 구절은 0 이다.

## 1. 두 벌

| 파일 | 추적 | 본문 | 쓰임 |
|---|---|---|---|
| `corpus/parse/passage_bank_v1.jsonl` (포인터 은행) | 추적 | 없음(`text: null`) | 문서 · 판본 · 원문 지문 · 쪽 · 절 · 면 · 결함 · 권리 · 본문 지문의 정본 |
| `passage_bank_v1_full.jsonl` (본문 은행) | 저장소 밖 | 원문 텍스트 층에서 기계로 뽑은 본문 | 생성기 입력. 포인터 은행의 같은 행과 지문이 맞아야 쓴다 |

본문을 추적하지 않는 까닭은 원문의 이용 조건이다. KR 은 규칙 PDF 판권면과 누리집 저작권 보호정책에서 무단 전재 ·
재배포를 막는다. IACS 누리집 약관은 개인 참조용 내려받기만 허용한다. ABS 약관은 사내 복사 · 사용을 허락하되 제3자
게시와 자동 수집을 막는다. 저장소는 공개다.

본문은 **재서술이 아니다**(`text_kind: verbatim_extract`). 텍스트 층의 줄을 읽은 순서대로 잇고 줄끝 하이픈만 붙였다.
상용 모델이 쓴 문장을 생성기 입력에 넣지 않으려는 것이다(규약 2-4). 검토자가 원문을 요약한 근거 카드도 넣지 않았다.

## 2. 행

| 칸 | 뜻 |
|---|---|
| `passage_id` | `문서:p물리쪽:소속_절:창번호`. 같은 원문 · 같은 빌더면 같은 값 |
| `doc_id` · `registry` | `corpus/parse/sources.yaml` 의 `doc_id`. 소비 경계가 등록 여부와 등록 sha256 을 대조한다 |
| `edition` · `source_file_sha256` | 판본 · 원문 파일 sha256. 빌더는 받은 파일의 sha256 이 다르면 멈춘다 |
| `locator` | `pdf_page_start/end`(물리 쪽, 1부터) · `printed_page_start/end`(인쇄 쪽, 없으면 null) · `part`(KR 만 — `규칙 2장` · `지침 2장` · `지침 부록 2-7` 꼴) · `section_id` · `section_kind`(`clause` · `table` · `lead_in`) · `carried_heading_ids`(앞에 붙인 짧은 절) |
| `text` · `text_sha256` · `text_chars` | 본문(포인터 은행은 null) · NFC + 공백 한 칸 정규형의 sha256 · 글자 수 |
| `facets` | 지식 면 후보: `definition` 정의 · `appearance` 소견 · `cause` 원인 · `prevention` 방지 · `repair` 보수 · `criteria` 기준 · `measurement` 측정 · `applicability` 적용 조건. **키워드 규칙(`keyword_rule_v1`)이 붙였고 사람이 보지 않았다** |
| `defect_classes` | 결함 후보(`crack` · `porosity` · `lack_of_fusion` · `lack_of_penetration` · `slag_inclusion` · `undercut` · `overlap`). 같은 키워드 규칙. ISO 코드 사상은 `configs/label_map.yaml` 이 정한다 |
| `relevance` | `defect_specific`(결함 이름이 나온다) · `weld_quality_general`(검사 · 보수 · 합부 같은 품질 낱말) · `welding_general`(용접만 나온다). 셋 다 아닌 구절은 넣지 않았다 |
| `scope` | 문서(KR 은 부록)가 다루는 구조물 · 재료 · 검사. 원문의 적용 범위 절에서 옮긴 짧은 설명이다. KR 부록 2-4 · 2-6 은 기관 주물 보수, FHWA 는 강 도로교다 — 선체 용접이 아니다 |
| `lineage_group_id` | 계보 묶음. train · val 은 묶음 단위로 가른다. ABS Pub 87 은 서문이 IACS Rec.47 Rev.3 을 바탕으로 했다고 적어 Rec.47 과 같은 묶음이다 |
| `rights_status` · `use_check` | 전부 `pending` — 이용 조건(L1)과 유료 계보(L2) 결정 전이다 |
| `human_review` | 전부 null |
| `flags` | 추출 오류 · 계보 표지(3절) |

L1 은 KR · IACS · ABS 원문을 어떤 꼴로 쓸 수 있는가(재서술만 · 인용 가능 · 쓰지 않음), L2 는 KR 부록 2-7 의
RT 판정 블록 수치가 유료 표준 계보인지 확인되지 않은 상태에서 그 수치를 쓰는가다. 둘 다 사용자가 정한다.

## 3. 대상 · 범위 · 제외

| 문서 | 판본 | 원문 sha256 | 넣은 물리 쪽 | 뺀 범위 | 구절 |
|---|---|---|---|---|---|
| `KR-RULES-P2` | 2025 | `46c4ee18…552b` | 123~220(규칙 2장), 264~276(지침 2장), 291~293(부록 2-4), 307~331(부록 2-6 · 2-7), 351~361(부록 2-12) | 재료 장 · 다른 부록 | 291 |
| `IACS47` | Rev.10 2021 / Corr.1 2025 | `ac54b790…15a6` | 4~48, 51~67 | 1~3(표지 · 목차 · 참고문헌), 49(Part B 목차), 50(참고문헌) | 87 |
| `ABS-PUB87-2007` | 2007 (Pub 87) | `be55985f…0565` | 7~44 | 1~6(표지 · 서문 · 목차) | 40 |
| `IACS-URW33-R1C1` | Rev.1 2020 / Corr.1 2021 | `1edc750f…fee3` | 1~14 | 개정 이력 문장 | 74 |
| `FHWA-HIF-19-088` | 2019-09, 2020-09 정오표 | `d317b8e9…4046` | 25~321(1~12장) | 1~24(앞붙이), 322~(부록 A · B · 참고문헌). 부록 B 는 D1.1 과 D1.5 의 차이라 유료 표준 비교다 | 761 |

ABS 의 NDI 지침(Guide for Nondestructive Inspection of Hull Welds)은 넣지 않았다. 그 4절은 참고문헌이 AWS D1.1 이다.
UR W33 은 지금 건조되는 선박에 적용되는 Rev.1 Corr.1 을 넣었다. Rev.2(2028-01-01 이후 건조계약)를 함께 넣으면
거의 같은 구절이 겹친다.

**구절을 만드는 규칙.** 절 머리 줄과 표 표제 줄에서 끊고, 머리 줄도 본문에 남긴다. 문장(마침표 · 물음표 · 느낌표 뒤
공백)을 900자 넘지 않게 묶는다. 마침표 없는 표 본문은 900자에서 공백 자리에 자르고 `hard_split_mid_sentence` 를 단다.
120자에 못 미치는 절(머리글뿐인 절 등)은 버리지 않고 같은 쪽 구간의 다음 절 앞에 붙인다. 쪽 구간 사이에서는 절을 잇지 않는다.

**빼는 규칙.** 구절 단위 — 점선 목차, `(see Table` 이 셋 이상인 표 목록, 개정 이력, 관련성 셋 어디에도 들지 않는 구절.
문장 단위 — AWS 를 출처로 단 인용(`third_party_quote_aws`), 유료 표준 이름이 수치 · 단위와 한 문장에 있는 것
(`paid_standard_numeric`). FHWA 매뉴얼은 물리 25쪽에서 "(clause x)" 를 Code(AWS D1.5) 조항 번호로 정하므로,
그 문서에서는 이름 없이 "clause 6.26.1.5" · "table 6.1" 처럼 점 번호로 D1.5 를 가리킨 것도 유료 표준 참조로 센다.
문장을 덜어 낸 절의 구절에는 `sentences_omitted_in_section` 을 단다. 사유별 수는
`passage_bank_v1.build_report.json` 에 있다.

**표지.**

| 표지 | 뜻 |
|---|---|
| `table_text_layer` | 표 본문이다. 칸 구조가 없어 행 · 열 대응이 텍스트에서 풀리지 않는다 |
| `pua_unmapped:N` | 대응하지 못한 사용자 영역 글자 N 개를 `□` 로 남겼다(KR 수식 글꼴의 변수 · 단위). **수치 기준이 빠졌을 수 있다** |
| `pua_symbol_mapped` | IACS47 표의 Symbol 글꼴 글자(≤ · ≥ · θ · ° 등)를 Adobe Symbol 대응으로 되돌렸다 |
| `kr_lt_glyph_is_angle` · `kr_angle_glyph_means_lt` | KR 조판은 부등호 `<` 자리에 `∠` 를 쓴다(문서 표지 · 그 글자가 실제로 든 구절) |
| `L2_lineage_unverified` | KR 부록 2-7 의 RT 판정 블록(물리 327~328)이 든다 |
| `paid_standard_name_without_numeric` | 유료 표준 이름이 수치 없이 나온다(적용 범위 · 인용 설명) |
| `paid_clause_reference_without_numeric` | FHWA 에서 D1.5 조항 번호가 수치 없이 나온다. 요구 내용을 풀어 쓴 문장일 수 있다 |
| `archived_edition` · `contractor_authored` · `third_party_figures_present` | 문서 단위: ABS 보관판 · FHWA 용역 저작 · 제3자 그림 |
| `section_started_before_range` | 쪽 구간 앞에서 시작한 절의 이어지는 부분 |

## 4. 소비 경계 — `corpus.parse.passage_bank.select_for_generation`

한 구절이라도 사유가 있으면 넘기지 않는다. 사유는 모두 센다.

| 검사 | 사유 키 |
|---|---|
| 행 꼴(필수 칸 · 열거값 · 지문 꼴) | `shape:*` |
| 등록부에 있고 등록 sha256 이 행의 원문 지문과 같다 | `registry:unregistered` · `registry:sha256_mismatch` |
| 본문이 있고 정규형 지문이 `text_sha256` 과 같다, `verbatim_extract` 다 | `no_text` · `text_sha256_mismatch` · `text_kind_not_verbatim` |
| 추적 포인터 은행의 같은 행과 문서 · 판본 · 원문 지문 · 본문 지문 · 위치가 같다 | `no_pointer_row` · `pointer_mismatch:*` |
| 사용자의 권리 결정 기록: `(doc_id, source_file_sha256)` 마다 하나, `rights_status: approved`, 그 용도가 `approved_uses` 에 있다, `decided_by` · `decided_at` | `rights:*` |
| 사람 검수 기록: `passage_id` 마다 하나, `decision: approved`, 검수한 본문 지문이 같다, `reviewer` · `reviewed_at` | `review:*` |
| 유료 표준 문장 규칙(빌드와 같은 함수) | `third_party_quote_aws` · `paid_standard_numeric` |

등록부의 `license_class` · `allowed_use` 는 보지 않는다. KR · IACS47 의 `OPEN` 은 약관 확인(2026-10-04)보다 먼저
적힌 값이고, 새로 등록한 세 문서는 `PENDING_DECISION` 이다. 권리는 결정 기록에서만 온다. 은행 행의
`rights_status` 로도 넘기지 않는다. 권리 결정 기록과 사람 검수 기록은 은행 빌더가 만들지 않는다.

`run_cycle_corpus.py` 는 `--passage-bank-full` · `--rights-record` · `--review-record` 로 세 파일을 받는다.
본문 은행이 없거나 아무 구절도 경계를 지나지 못하면 QA 구절은 0 이다. 옛 조사 마크다운으로 되돌아가지 않는다.

## 5. 용량 — `passage_bank_v1.capacity.json`

생산 설정 초안(`configs/corpus_production.json`, 2026-10-07 시점, 미추적)은 train 원문 QA 하한 10,000 · 목표 20,000 ·
고유 train 구절 하한 3,334 · 구절당 질문 상한 3 이다. 수는 산출 파일에서 읽는다. 아래는 요약이다.

- 은행 1,253 구절 → QA 상한 3,759. 하한에 6,241, 목표에 16,241 모자란다. 고유 구절은 하한에 2,081 모자란다.
- 계보 묶음은 통째로 한쪽에 간다. 가장 작은 묶음(UR W33, 74 구절)을 val 로 두면 train 은 1,179 구절 · 3,537 QA 다.
- `welding_general` 을 빼면 747 구절 · 2,241, 결함 이름이 나오는 구절만이면 214 구절 · 642 다.
- 판정추론: 허용치 표 v0 의 조합은 RT 11 · VT 1, 조합 상한 40 에서 480 이다. 1만에 9,520 모자란다. 상한 40 에서
  1만을 내려면 서로 다른 규칙 단위가 250 개 있어야 한다. **RT 11 조합은 전부 KR 부록 2-7 표15 · 표16 · 본문(d)(물리 328),
  곧 L2 블록이다.** L2 를 쓰지 않는 쪽으로 정하면 RT 판정추론은 0 이다.

반복으로 채우지 않는다. 늘릴 수 있는 원천과 같은 빌더 규칙으로 잰 크기:

| 원천 | 상태 | 더해지는 구절(QA 상한) |
|---|---|---|
| ABS NDI 지침 1~3절 · 부록(4절 제외) | 파일 있음, ABS 약관 · 미등록 | 165 (495) — 같은 규칙으로 잼 |
| IACS UR W33 Rev.2 | 파일 있음 | 80 가운데 44 가 Rev.1 Corr.1 과 본문 지문이 같다. 고유분은 최대 36 이고 대부분 판 차이만 있는 근사 중복 |
| 그 밖(MIL-STD-2035A, 학회지 논문, 미 육군 교범) | 원문 미확보 또는 이용 조건 미확인 | 재지 않았다 |

위 표의 앞 두 줄을 다 더해도 1,454 구절 · QA 상한 4,362 다. 하한 10,000 은 고유 구절 3,334 가 필요하고, 그만큼은
이 다섯 문서 계열 밖에서 와야 한다.

## 6. 재현

```
python -m corpus.parse.build_passage_bank \
  --source KR-RULES-P2=corpus/parse/raw/KR-RULES-P2_2025.pdf \
  --source IACS47=corpus/parse/raw/IACS47_Rev10Corr1_2025.pdf \
  --source ABS-PUB87-2007=<ABS Pub 87 PDF> \
  --source IACS-URW33-R1C1=<UR W33 Rev.1 Corr.1 PDF> \
  --source FHWA-HIF-19-088=<FHWA-HIF-19-088 PDF> \
  --out-full <저장소 밖>/passage_bank_v1_full.jsonl \
  --out-pointer corpus/parse/passage_bank_v1.jsonl \
  --report corpus/parse/passage_bank_v1.build_report.json
python -m corpus.parse.passage_bank_capacity --bank corpus/parse/passage_bank_v1.jsonl \
  --limits corpus/rules/limits_v0_pilot.csv --out corpus/parse/passage_bank_v1.capacity.json
```

원문 URL 은 `sources.yaml` 에 있다. pypdfium2 5.13.0 · pdfium 153.0.7999.0 에서 두 번 돌려 같은 바이트를 냈다.
다른 판의 pdfium 은 텍스트 층을 다르게 줄 수 있다.

## 7. 보증하지 않는 것

- 면 · 결함 · 관련성 표지가 맞다는 것. 키워드 규칙이고 사람이 보지 않았다.
- 본문이 원문과 글자 단위로 같다는 것. 텍스트 층의 읽기 순서 · 사용자 영역 글자 · 표 구조 손실이 있다(표지로 남겼다).
- 절 귀속이 모두 맞다는 것. 따로 작성된 FHWA 근거 카드 10장과는 쪽 · 절이 모두 맞았다(두 장은 절 둘을 함께 적었다). 그 밖은 표본으로만 봤다.
- 유료 표준 유래 문장을 모두 걸렀다는 것. 규칙은 "이름(FHWA 는 D1.5 조항 번호 포함) + 수치 · 단위가 한 문장" 과 "AWS 출처 인용" 만 잡는다. 이름도 조항 번호도 없이 수치만 옮긴 문장은 못 잡는다.
- 결함 표지가 용접 결함만 가리킨다는 것. 표본 12구절에서 콘크리트 균열 · 프로펠러 주물의 피로균열을 `crack` 으로 단 구절이 있었다.
- 권리 · 사실. 권리는 사용자가, 사실은 사람 검수가 정한다. 파일럿의 형식 통과 188건과 허용치 12행을 금본으로 올리지 않는다.
