# 코퍼스 원문 출처와 재현 절차

이 저장소는 규정 원문 전문을 추적하지 않는다. 무료 공개 문서라도 재배포는 별개
권리이고, 저장소가 공개로 전환되면 이력에 남은 전문이 곧 재배포가 된다. 아래 절차로
누구든 동일 산출물을 로컬에서 재현할 수 있다. 기계 판독용 등록부는 `sources.yaml`이
단일 진실이고, 이 문서는 사람이 읽는 재현 안내다.

## 원문 확보 (→ `corpus/parse/raw/`, git 미추적)

| doc_id | 판본 | 취득 | sha256 |
|---|---|---|---|
| KR-RULES-P2 | 2025 선급 및 강선규칙/적용지침 제2편 재료 및 용접 (합본, 362쪽) | krs.co.kr 공개 PDF (sources.yaml의 url) | `46c4ee189f4adef29e42320f9301d8db577946ec96f893be9ff2f1683879552b` |
| IACS47 | IACS Rec. No.47 Rev.10 Corr.1 (Oct 2025, CLN, 67쪽) | iacs.org.uk 경유 공식 S3 (sources.yaml의 url) | `ac54b7904b3f27d8af830ef2f4ff0ca18967a407aeead67c481b136b542f15a6` |

다운로드 후 `corpus/parse/raw/RAW.sha256`(추적됨)과 대조한다. 해시가 다르면 판본이
다른 것이고, 이후 모든 산출물 재현이 성립하지 않는다.

## 파싱 재현 (→ `corpus/parse/survey/`, meta_*.json 만 추적)

```
uv sync --extra corpus
uv run python -m corpus.parse.survey_textscan KR-RULES-P2          # 키워드·텍스트레이어 스캔
uv run python -m corpus.parse.survey_docling KR-RULES-P2 --pages 316-336   # 부록 2-7
uv run python -m corpus.parse.survey_docling IACS47                # 전문 67쪽
```

- 고정 조건: docling 2.120.1 (uv.lock), TableFormer ACCURATE, OCR off,
  `TORCHDYNAMO_DISABLE=1` (스크립트에 내장. sm_120/Windows에 triton 휠이 없어 필요하다).
- 재현 검증: 산출 표 수·형상·페이지가 추적되는 `meta_full.json` /
  `meta_p316-336.json`과 일치해야 한다.

## 무엇이 추적되고 무엇이 안 되나

| 추적 ○ | 추적 × (로컬 전용) |
|---|---|
| 파싱 스크립트, `sources.yaml`, `RAW.sha256`, `meta_*.json` | 원본 PDF (`raw/`) |
| 파생 사실 정보: 허용치 수치·조항번호·결함코드 (`corpus/rules/limits*.csv`) | 변환 markdown·표 CSV·textscan/peek (원문 표현 포함) |

원문 인용이 필요한 검증(전문가 검토·앵커 대조)은 로컬 산출물 또는 원문 PDF를 직접
참조한다. 저장소에는 넣지 않는다.
