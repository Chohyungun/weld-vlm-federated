# 구절 은행 v3 — 원문 보강분 합침

2026-10-07. v2 위에 원문 보강분 다섯 편 58구절을 합쳤다. v1 · v2 파일은 기록으로 남기고 고치지 않는다.
행 꼴 · 소비 경계 · 표지의 정의는 `passage_bank_v2.md` 를 따르고, 이 문서는 달라진 것만 적는다.

**학습 투입 승인이 아니다.** 1,314 행 모두 `rights_status: pending` 이고 사람 검수가 없다.
본문 은행에 소비 경계를 실제로 돌리면 지금 넘어가는 구절은 0 이다(`passage_bank_v3.capacity.json`).

## 1. 합친 것

| 문서 | 다루는 것 | 구절 | 추출 |
|---|---|---|---|
| `mdpi_met11060904_2021` | S960QL 협개선 GMAW 의 수소 유발 균열 | 11 | 출판사 JATS XML 연속 구간 |
| `mdpi_met11020293_2021` | 보수한 필릿 용접부의 피로 | 12 | 출판사 JATS XML 연속 구간 |
| `MDPI-CRYSTALS-2022-12-832` | X80 협개선 레이저-CMT 의 측벽 융합 불량 | 10 | PDF 텍스트 층 |
| `MDPI-METALS-2024-14-857` | CP780 아연도금강 GMAW-P 의 기공 | 10 | PDF 텍스트 층 |
| `MDPI-METALS-2020-10-1121` | 6061 알루미늄 아크 각도와 기공(와이어식 CMT/GMAW) | 15 | PDF 텍스트 층 |

다섯 편 모두 출판사 표시 CC BY 4.0 이다. 등록부에는 결정 대기(`PENDING_DECISION`, `allowed_use: []`)로 넣었다.

**합칠 때 확인한 것**(행마다, 본문은 바꾸지 않는다):

- 등록부의 sha256 · 행의 `source_file_sha256` · 실제 원문 파일이 같다. XML 추출 행은 XML 의 sha256 도 등록부 · 행과 같다.
- 본문 지문이 본문과 맞는다.
- 본문이 원문에 있다. XML 행은 그 XML 본문(`body`)에, PDF 행은 그 쪽 범위의 텍스트 층에 있어야 한다. 쪽을 넘는 구절은 학술지 쪽 머리글 줄을 빼고 잰다.
  결과는 정규화 후 그대로 36 · 글자와 숫자만 같음 22 · 못 찾음 0 이다. 행의 `extraction.bank_v3_source_check` 에 남겼다.
- id · 본문이 기존 1,256 구절과 겹치지 않는다(겹침 0).

**바꾼 칸은 둘뿐이다.** `registry` 를 등록부 경로로 고쳤고, 원래 값과 입력 파일 지문은 `merged_from` 에 남겼다.
`scope`(이 묶음은 문자열 또는 객체) · `generation_constraints` · `required_context_passage_ids` · `related_passage_ids` ·
보류 칸 · XML 추출 근거(XPath · 문단 순서 · 문자 위치 · XML 지문)는 받은 그대로 보존했다. v2 의 1,256 행은 빌더 판 칸 말고 v2 와 같다.

## 2. 보류 — 소비 경계가 강제한다

| 칸 | 뜻 | 사유 키 |
|---|---|---|
| `source_policy_holds` | 원문 정책에 걸린 구절. `mdpi_met11060904_2021_p03` 이 유료 표준 수치 규칙에 걸린다 | `hold:source_policy:<사유>` |
| `context_dependency_policy_holds` | 보류된 구절을 필수 문맥으로 삼는 구절. 같은 논문의 11 구절 | `hold:context:<구절>` |

보류는 권리 결정 · 사람 검수 기록으로 풀리지 않는다. 보류를 푼 판의 은행이 새로 나와야 한다. 유료 표준 문장 규칙은 그대로이고,
p03 도 그 규칙(`paid_standard_numeric`)에 따로 걸린다.

## 3. 필수 문맥 — 경계는 강제하고, 생성 지시는 소비자 몫이다

`required_context_passage_ids` 는 58 가운데 56 행에 있다. 소비 경계(`select_for_generation` · 단일 진입점 `select_bank_for_generation`)는
문맥 구절이 같은 계보이고 같은 용도로 함께 넘어갈 때만 의존 구절을 넘긴다. 문맥의 사슬 · 고리는 더 빠지는 구절이 없을 때까지 되풀이해 닫는다.
품질 검사(`corpus.production.quality`)는 작업의 근거에 그 문맥 본문이 들어 있는지 보지 않는다. 문맥 본문을 생성 지시에 싣는 것은 작업 명부의 일이다.
`run_cycle_corpus` 는 이 칸과 `generation_constraints` 를 QA 레코드의 `source` 에 실어 보낼 뿐 문맥 본문을 붙이지 않는다.

## 3-1. 합친 행의 상대 경로 — 기준점은 지문이다

합친 58 행의 `extraction.source_pdf` · `source_xml` · `draft_input` · `rights_receipt` 는 `provenance_relative_paths_base: this_delivery_folder`, 곧 **납품 폴더 기준** 상대 경로다.
은행 폴더에서 풀면 다른 곳을 가리킨다. 행을 고치면 은행 지문이 바뀌므로 행은 그대로 두고, 기준점과 파일 지문을 `passage_bank_v3.provenance.json` 에 적었다:
기준 폴더 = 납품 은행 파일 `passage_bank_full.jsonl`(sha256 `319ee08505507514fb584f5b8bed733c1d48adb8efcc92b5c9ded5161d170d4c`, 행의 `merged_from.bank_sha256`)이 든 폴더.
그 아래 파일 13 개의 sha256 · 바이트 수와 행마다 칸별 지문이 있다. 위치가 바뀌어도 지문으로 찾고 대조한다. 행마다 원문 PDF 지문은 행의 `source_file_sha256` 과 같다 — 등록부 키는 바뀌었어도 원문은 같은 파일이다.
다음 판을 빌드할 때는 이 기준점을 행에 직접 싣는다 — 빌더는 아직 고치지 않았다.

## 4. 용량 — `passage_bank_v3.capacity.json`

| 구분 | 구절 | QA 상한(구절당 3) | 하한 10,000 까지 | 고유 구절 하한 3,334 까지 |
|---|---|---|---|---|
| 전체 | 1,314 | 3,942 | 6,058 | 2,020 |
| 보류 구절 제외 | 1,303 | 3,909 | 6,091 | 2,031 |
| 역사 자료 제외 | 1,308 | 3,924 | 6,076 | 2,026 |
| 결함 이름이 나오는 구절만 | 271 | 813 | 9,187 | 3,063 |

새 58 구절의 상한 174(58 × 3)는 상한일 뿐 생산량이 아니다. 원인 · 예방 · 보수 · 결함의 분류 수는 겹쳐 세어지므로 독립 문항 수로 더하지 않는다.
이번 합침으로 결함 · 재질 전체의 커버리지가 완성된 것이 아니다. 계보 묶음은 열하나다(새 다섯 편이 각각 하나).

## 5. 재현

`passage_bank_v2.md` 6절의 명령에 다음을 더한다.

```
  --merge-rows <원문 보강분 본문 은행 JSONL, sha256 319ee085…0d4c> \
  --merge-source <DOC_ID>=<PDF> (다섯) --merge-xml <DOC_ID>=<XML> (둘) \
  --out-full <저장소 밖>/passage_bank_v3_full.jsonl \
  --out-pointer corpus/parse/passage_bank_v3.jsonl --report corpus/parse/passage_bank_v3.build_report.json
```

두 번 돌려 같은 바이트를 냈다.

## 6. 보증하지 않는 것

- 합친 58 구절의 사실성 · 분류. 면 · 결함 표지는 그 묶음이 정한 값이고(`facet_method: source_context_curated_not_human_approved`) 사람이 보지 않았다.
- 실험 조건과 표준 규정 전재의 구분. p03 은 판정만 기록했고 보류는 풀지 않았다.
- 필수 문맥이 생성에 함께 쓰인다는 것(3절).
- 합친 행의 원문 · 영수증 파일이 지금도 그 폴더에 있다는 것. 지문은 2026-10-08 에 13 개 모두 대조했다(3-1절) — 그 뒤의 이동 · 삭제는 지문 대조로만 알 수 있다.
- 계보 묶음 이름의 일관성. 새 다섯 편 가운데 둘은 `doi:…`, 셋은 `LG-…` 꼴이다. 받은 값을 그대로 두었다 — 한 논문이 한 묶음이라는 뜻은 같다.
