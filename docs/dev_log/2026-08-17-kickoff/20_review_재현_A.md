# 재현 검토: 데이터 파이프라인

작성 2026-08-17 · 대상 `10_spec_A_데이터파이프라인.md` 및 그 구현 (커밋 `1af1d03`)
원저자가 아닌 사람이 검토한다.

**요약: Critical 0 / Important 1 / Minor 2.** 조건부 후속 2건.

구현은 재현된다. 82건 전부 통과했고, 스냅샷 해시는 독립 계산으로 전건 일치했으며,
누수 불변식(묶음이 split·client를 가로지르지 않음)은 실제 데이터에서 위반 0건이었다.
Important 1건은 구현의 결함이 아니라 데이터 파이프라인 쪽 명세와 채점 쪽 명세가 서로 다른
기본값을 가정하고 있는 접점 문제다. 확정이 필요하다.

---

## 1. 재현 절차와 결과

전부 실제로 실행했다. 검증 스크립트 경로는 말미 §5에 적었다.

### 1-1. 테스트 재실행: 재현됨

```
uv run python -m pytest -q
→ Creating virtual environment at .venv (CPython 3.11.13)
→ Installed 113 packages in 1m 38s
→ 82 passed in 55.33s
```

깨끗한 새 가상환경에서 잠금 파일만으로 재구성됐다. 게이트가 기록한 82건과 정확히 일치한다.
파일별 분포: `test_dirichlet` 14 / `test_geometry` 11 / `test_invariants` 21 /
`test_label_map` 17 / `test_manifest_io` 18 = 81 + 파라미터화 1건.

### 1-2. 스냅샷 해시 독립 계산: 전건 일치

`SNAPSHOT.sha256`을 읽지 않고 파일 바이트에서 직접 sha256을 계산해 대조했다.

| 스냅샷 | manifest.csv | annotations.csv | data_capabilities.yaml | snapshot_digest |
|---|---|---|---|---|
| `mock_aihub_v1` | 일치 | 일치 | 일치 | 재구성 성공 |
| `mock_riawelc_v1` | 일치 | 일치 | 일치 | 재구성 성공 |

`snapshot_digest`는 선언 파일에 규칙이 적혀 있지 않아 후보 4종을 시도해 역산했다.
**`sha256("{hash}  {name}\n" × 파일 수)`** 로 두 스냅샷 모두 재구성됐다(§3 Minor-1).

파일에 CRLF는 없었다. `write_snapshot`이 바이트 정규 CSV를 쓴다는 A 스펙 §의 주장이
실제로 지켜져, Windows 체크아웃에서도 해시가 흔들리지 않는다. **이 점은 명시적으로
확인해 둔다.** `.gitattributes` 없이 CRLF가 섞였다면 `check-manifest`가 새 환경에서
거짓 실패했을 수 있다.

### 1-3. 분할 불변식: 위반 0건

채점 쪽에서 가장 신경 쓰는 항목이다. 여기가 깨지면 다섯 칸 채점이 통째로 무의미해진다.

| 검사 | `mock_aihub_v1` (1,000장) | `mock_riawelc_v1` (300장) |
|---|---|---|
| `group_id`가 둘 이상의 split에 걸침 | **0건** | **0건** |
| train 내 `group_id`가 둘 이상의 client에 걸침 | **0건** | **0건** |
| eval 비율 | 0.200 (200/1,000) | 0.200 (60/300) |
| eval 행에 client 배정 | 없음 (전부 공백). **선분리 순서 준수** | 없음 |
| annotations → manifest 참조 무결성 | 고아 0건 | 고아 0건 |
| `n_defects` 선언값 ↔ 실제 annotation 수 | 불일치 0건 | 불일치 0건 |
| `has_localization=false`인데 bbox가 채워짐 | 해당 없음 | **0건** |
| 퇴화 bbox(x1≥x2) / 이미지 범위 이탈 | 0건 / 0건 | 해당 없음 |
| 두 어댑터의 manifest·annotations 컬럼 집합 동일 | **동일** | (해당 없음) |

eval 행의 `client`가 비어 있다는 것은 **평가셋을 회사별 분할보다 먼저 뗐다**는 순서
불변조건이 데이터에 실제로 남아 있다는 뜻이다. 문서 주장과 산출물이 일치한다.

### 1-4. Dirichlet 재현성: 재현됨, 입력 순서에도 불변

`data/split/dirichlet.py`를 직접 호출해 확인했다(묶음 300, 4클래스).

| 검사 | 결과 |
|---|---|
| 동일 입력·동일 시드 2회 호출 | 완전 일치 |
| **입력 순서를 무작위 치환한 뒤 되돌림** | 완전 일치. 순서 독립성 주장이 실측으로 성립 |
| 시드 변경 시 배정 변화 | 변화함 (시드가 실제로 작동) |
| 전 묶음 배정, 값 ∈ {0,1} | 성립 (미배정 0건) |
| `partition_with_acceptance` 2회 호출 | `seed_used`·배정 모두 일치 |

순서 독립성은 코드가 각 클래스 내에서 `group_ids`를 사전순 정렬한 뒤에 셔플하기 때문에
성립한다(`dirichlet.py:85`). **호출자가 정렬해 넘길 것을 문서로만 요구하지 않고 함수가
다시 강제**한 설계가 옳다. 호출자가 바뀌어도 결과가 안 바뀐다.

시드 파생도 추적했다. `split_meta.seed=20260825`인데 `dirichlet.seed_used=20260828`이라
어긋나 보이지만, `make_mock_manifest.py:195`가 `seed + 3`을 넘기고 `attempts=1`이므로
정합하다. 파생 규칙이 코드에 있고 사후 추적이 가능하다.

### 1-5. 명세와 구현의 컬럼 대조: 드리프트 없음

구현이 내보내는 48개 컬럼(manifest 30 + annotations 18) 전부가 A의 스펙 문서에 백틱
표기로 등장한다. **스펙에만 있고 구현에 없는 컬럼도, 구현에만 있고 스펙에 없는 컬럼도
없다.** D의 가정 인터페이스 G1·G2가 요구한 컬럼(`image_id`, `width_px`, `height_px`,
`split`, `client`, `material`, `modality`, `group_id`, bbox 4종)은 전부 존재한다.

---

## 2. Important

```
[Important] 계약 #2(A) ↔ 계약 #4·13_spec_D §4-11. 스케일 부재 시의 판정 경로에서
A는 3값(absolute/conditional/clause_only), D는 2값(scale_available)을 가정하고 있고,
두 mock 모두 clause_only + assumptions null이라 D의 가정값 경로가 발동하지 않는다.
```

**근거.**
- A 스펙 §(라인 391~393): `verdict_mode`는 세 값이다. `conditional`은 "미보유이나
  **가정값을 명시적으로 채택**", `clause_only`는 "미보유 + **가정값 미채택** → 합부 미산출,
  판정 정합성 대신 `N/A`".
- `data/mock/*/data_capabilities.yaml`: 두 스냅샷 모두 `verdict_mode: clause_only`,
  `assumptions: {thickness_mm: null, px_per_mm: null, quality_level: null, rationale: null}`.
- `13_spec_D_평가RAG.md` §4-11: D는 `evaluation.scale_available`(auto/true/false)로 분기하고
  `false`일 때 `assumed_px_per_mm: 10.0`, `assumed_thickness_mm: 12.0`,
  `assumed_quality_level: "C"`를 5칸 공통으로 적용한다고 썼다.
- 즉 **A의 산출물 기준으로는 `clause_only`(판정 정합성 N/A)인데, D의 스펙 기준으로는
  가정값을 넣어 조건부로 산출**한다. 두 문서가 같은 상황에 다른 기본값을 정해 놓았다.

**실패 시나리오.** AI허브 데이터에 두께·픽셀 스케일이 없는 것으로 확정된다(현재 최대
리스크로 등재된 시나리오). W7에 D가 채점기를 돌린다. 채점기는 `scale_available=false`를
읽고 `px_per_mm=10.0` 가정으로 판정 정합성을 산출해 결과표에 0.9x를 채운다. 한편 A의
`data_capabilities.yaml`은 `clause_only`, 즉 "합부 미산출"을 선언하고 있다. 논문 심사에서
"두께 정보가 없는데 합부 정합성 수치가 어떻게 나왔는가"를 물으면, **우리 저장소 안의 두
계약 파일이 서로 다른 답을 한다.** 어느 쪽이 정본인지 문서로 정할 수 없어 수치의 근거를
방어하지 못한다. 반대로 D가 A를 따라 `clause_only`로 가면 §7 해석 기준선의 "판정 정합성
95% 이상 기대" 행이 통째로 비고, 최소 성립선(분리형 3칸 + **조항 정확도**)은 유지되지만
헤드라인 표의 한 열이 사라진다.

Critical로 올리지 않은 이유: 실험을 다시 돌릴 필요가 없고(채점기 재실행으로 흡수된다),
레드라인 위반도 아니며, 논문 주장의 최소 성립선은 두 경로 모두에서 유지된다.

**처방.**
1. **`data_capabilities.yaml`을 정본으로 하고 D가 그것을 읽도록 D 스펙을 고친다.**
   D의 `scale_available` 2값을 폐기하고 A의 `verdict_mode` 3값을 그대로 소비한다.
   계약이 하나여야 한다는 원칙(허용치 CSV 단일 소스와 같은 논리)에 맞다.
2. `conditional`로 승격할지 여부는 확정이 필요하다. 승격하려면 `assumptions` 블록을
   채워야 하며, 그 값이 곧 논문 각주가 된다.
3. 채점기는 `clause_only`를 기본 경로로 가정하고 만든다. 판정 정합성이 `N/A`인 상태에서도
   나머지 12개 지표가 전부 나오는지를 mock으로 먼저 확인한다.

> 채점 명세 수정이 처방의 절반이므로, 이 항목은 지적이라기보다 접점 결정 사안이다.

---

## 3. Minor

```
[Minor] 10_spec_A §스냅샷. snapshot_digest 산출 규칙이 어디에도 적혀 있지 않아
제3자가 검증하려면 역산해야 한다.
```

**근거.** `SNAPSHOT.sha256` 마지막 줄 `# snapshot_digest <64hex>`의 계산식이 스펙 문서와
파일 주석 어디에도 없다. 리뷰어가 후보 4종(hex 연결·이름 정렬 연결·라인 연결·바이트 연결)을
시도해 `sha256("{hash}  {name}\n" 연결)`임을 역산했다.

**실패 시나리오.** 논문 제출 전 `make reproduce-check`를 제3자(공동저자·심사자)가 새
환경에서 돌린다. 개별 파일 해시는 대조할 수 있지만 `snapshot_digest`는 규칙을 몰라
검증할 수 없다. 이 값은 **논문에 싣는 값**이므로(`docs/개발규약.md` 불변조건 1-6), 논문에 실린 해시를 독자가
재현할 수 없는 상태가 된다. 재실험이 필요하지도, 주장이 무너지지도 않으므로 Minor.

**처방.** `SNAPSHOT.sha256` 파일 주석 한 줄 또는 A 스펙 §스냅샷에 계산식을 명시한다.
파일 순서 의존성(나열 순서대로 연결하는지, 이름 정렬 후 연결하는지)도 함께 적는다.
지금 구현은 나열 순서 의존이라 파일이 하나 늘면 순서 규칙이 곧 정본이 된다.

```
[Minor] data/split. 층화 분할이 "모든 strata가 eval에 최소 1건"을 보장하지 않는다.
mock_aihub_v1의 AL|crack(5장·3묶음)이 eval 0건으로 떨어졌다.
```

**근거.** strata별 eval 비율 실측: `AL|crack` 0.000(5장 중 0), `AL|lack_of_fusion` 0.200
(5장 중 1), 나머지 8개 strata는 0.195~0.208. 묶음 단위 층화 분할에서 묶음 수가 폴드 수보다
적으면 구조적으로 발생한다.

**실패 시나리오.** 실데이터에서는 AL 균열이 332장이므로 이 조합은 eval에 ~66장이 들어가
발생하지 않는다. **즉 mock 크기의 산물이지 알고리즘 결함이 아니다.** 다만 가드가 없으므로,
장래에 C4(현장 데이터 3,000장, 저널 단계)를 추가하거나 α 민감도 분석으로 분할을 다시
뽑을 때 희소 조합이 eval에서 사라질 수 있다. 그 경우 D의 클래스별 P/R/F1에서 해당 클래스가
0/0이 되고 Macro-F1의 그 항이 정의되지 않는다. RQ3가 알루미늄 클라이언트 편익을 묻는
연구이므로 **하필 AL 계열 클래스가 빠지면 RQ3 해석에 직접 타격**이다.

**처방.** 분할 직후 어서션 한 줄. "eval에 각 `strata_key`가 최소 1건, 각 결함 클래스가 최소
K건(K는 configs)"을 검사하고 미달 시 경고와 함께 목록을 남긴다. 실패가 아니라 경고면
충분하다. 희소 조합이 실제로 존재하는 데이터도 있을 수 있다. 채점 쪽에서도 `metrics/`가
표본 0인 클래스를 `N/A`로 반환하고 Macro 평균에서 **제외한 뒤 제외 사실을 보고**하도록
방어를 넣는다.

---

## 4. 통과 확인: 지적하지 않은 것들

검토했고 문제가 없었던 항목이다. 다음 검토자가 중복 조사하지 않도록 남긴다.

- **원본 불변**: `test_raw_dir_untouched`가 ingest 전후 `data/raw/` 해시·mtime을 검사한다.
- **누수 검출 테스트의 존재**: `test_IV7_group_split_leak_detected`,
  `test_IV8_group_client_leak_detected`, `test_IV6_eval_with_client`,
  `test_IV9_eval_subset_outside_eval`. 불변식을 **위반 데이터를 만들어 탐지되는지**로
  검사한다. 정상 데이터만 통과시키는 테스트보다 강하다.
- **라벨 하드코딩 금지의 집행**: `test_no_hardcoded_label_strings`가 저장소 전체를
  grep한다. 선언이 아니라 실행으로 막았다.
- **두 어댑터 수렴**: `test_both_adapters_emit_same_columns`가 있고, 실측으로도 컬럼
  집합이 동일했다. 8/25에 어느 쪽으로 확정되든 D의 채점기 코드는 안 바뀐다.
- **결측 표현**: `test_empty_string_not_nan`, `test_mm_null_when_no_scale`(0이 아니라 null).
  D의 지표 계산에서 0과 결측을 구분해야 하는 지점(§4-3 판정 정합성 분모 제외)이 많아
  중요한데, 소스에서 이미 구분돼 있다.
- **Dirichlet 농도 벡터 표기**: "α=0.5" 축약 대신 농도 벡터 `(1/3, 1/6)`를 그대로 넘긴다.
  `α·p`인지 `α·n·p`인지에 따라 이질성 강도가 달라져 재현이 불가능해지는 문제를 회피한
  것으로, 논문에 실을 때도 이 표기가 방어된다.
- **수용 밴드 재추첨**: `partition_with_acceptance`가 `seed, seed+1, …`로 재추첨하고
  **채택 시드와 시도 횟수를 기록**한다. "예산 캡·빈패킹·클래스 0장 재추첨은 쓰지 않는다"는
  선택도 근거가 적혀 있다(실현 분포를 명목 α에서 멀어지게 만든다).

---

## 5. 검증 스크립트

검토에 쓴 스크립트 2종을 저장소에 남긴다. 검토 재현용이며 실험 경로가 아니다.

| 스크립트 | 검사 |
|---|---|
| `scripts/review/verify_snapshot_hashes.py` | §1-2. 선언 해시 무시하고 바이트에서 재계산, snapshot_digest 규칙 역산 |
| `scripts/review/verify_split_invariants.py` | §1-3. 묶음 누수·선분리 순서·층화·참조 무결성·bbox 유효성 |

---

**Critical 0 / Important 1 / Minor 2.** 후속 2건: Important 1건은 확정 필요, Minor 2건은 원저자 재량.
