# 채점 절차 — 다섯 칸 공통 채점기

본실험 채점이 밟는 순서와, 각 단계가 무엇을 보증하는지를 적는다. 채점 전 조치 3건(차단 검사의
종료 코드 반영, 층화 채점 실행의 절차 명시, `prereg_recomputed_v1.json` 선배치)을 반영한 판이다.

**채점 코드는 한 벌이다.** 다섯 칸 출력을 계약 #4 레코드로 바꾼 뒤 `evaluation.score`
하나로 채점한다(개발규약 3-7). 칸마다 채점 코드를 따로 두지 않는다.

## 0. 전제

- `git merge main --ff-only` 로 최신 코드. 저장본 `score_cells_v1.json` 의 `params` 에
  `profile`·`model_cfg`·`predict_chunk`·`imgsz_source` 네 키가 있어야 최종 코드 산출이다
  (13번 D-8 봉인).
- GPU 를 쓰지 않는다. `predict` 만 CPU 추론이고 나머지는 순수 채점이다.
- 채점 디렉터리(`--out`)는 실험마다 따로 둔다 — 파일럿은 `outputs/pilot_d`, 본실험은
  실험마다 정한 경로. 저장본을 덮어쓰지 않는다.
- **프로파일을 명시한다.** 본실험은 `--profile main`(YOLO11s · 640). 빠뜨리면 파일럿
  기본값(YOLO11n · 416)이 잡히고, 가중치 형상 불일치로 시작조차 하지 않는다.

## 1. 순서

| # | 명령 | 보증하는 것 |
|---|---|---|
| 1 | `uv run python scripts/probe/score_cells.py predict --profile main --pilot <C 산출> --out <채점 dir>` | 검출 3칸 하한(0.01) 추론 → 계약 #4 레코드. 내부 256장 청킹. `--at-conf` 는 운용 임계 레코드 |
| 2 | `uv run python scripts/probe/score_cells.py score --profile main --pilot <C 산출> --out <채점 dir>` | **본채점.** 전역 지표(운용점 예시) + **곡선 전 구간** + **임계 독립 헤드라인 지표** + id 구간 층화 블록 + 규격 지름길 각주 + 게이트 전수 + prereg 상수 선배치. 종료 코드가 판정이다(§2) |
| 3 | `uv run python scripts/probe/score_cells.py sweep --profile main ...` | conf 스윕 — 단일 임계 결론 금지. 하한 1회 추론 + 사후 필터의 동치는 `--verify-parity` 로 실측 |
| 4 | `uv run python scripts/probe/stratified_compare.py --k 64 --ladder --pilot <C 산출> --out <채점 dir>` | 층화 **상세** 산출물(구간별 행 포함, `stratified_compare_v1.json`). 판정 6 의 대조표가 이 파일이다 |
| 5 | `uv run python scripts/probe/score_cells.py gate --gate <값>` | (선택) 게이트 상수만 갈아 끼워 재판정. 채점은 건드리지 않는다 |

2번이 4번을 대신하지는 않는다 — 2번은 표(전 K 사다리, 상세 없음)를 산출물에 싣고
`stratified_scoring` 게이트로 **이행을 담보**하며, 4번은 구간별 상세와 판별력 시험 상세를
남긴다. **둘 다 돌린다.** 2번 없이 4번만 돌리면 게이트 기록이 없고, 4번 없이 2번만 돌리면
구간별 근거가 없다.

## 2. 종료 코드 — `score`

| 코드 | 뜻 | 처리 |
|---|---|---|
| `0` | 차단 게이트 실패 없음 · 저장 지표 대조 일치 | 결과로 쓴다 |
| `1` | 저장 지표(65·66번) 대조 불일치 | 정의를 바꾼 것이 아니면 회귀다. `REDEFINED_KEYS` 밖의 키가 달라졌는지 본다 |
| `2` | **차단 게이트 실패** | 산출물(`score_cells_v1.json`)은 증거로 남지만 **이 채점을 결과로 쓰지 마라.** `gates_evaluated.blocking_failures` 를 본다 |

차단은 종료 코드다(13번 D-7). 이전 판은 출력만 하고 0 을 돌려줘 "차단 ○" 이 기록
이상이 아니었다. 산출물 안에도 `exit_code`·`exit_reason` 이 남는다.

## 3. 게이트 — 매 채점마다 전수 호출 (`evaluation/gates.py`)

| 게이트 | 본다 | 차단 |
|---|---|---|
| `no_cloud_logging` | `MLFLOW_`·`WANDB_`·`COMET_`·`NEPTUNE_`·`CLEARML_` 환경변수, `*TRACKING_URI` 의 원격 스킴 | ○ |
| `prereg_constants_reproduced` | 채점 디렉터리의 `prereg_recomputed_v1.json` 이 등록 상수를 재현하는가. **없으면 채점기가 동결본에서 만들어 선배치한다**(13번 D-8 파생) | ○ |
| `scoring_population` | 다섯 칸이 평가셋 전량으로 채점됐는가(정상 이미지 포함) | ○ |
| `stratified_scoring` | 층화 블록이 산출물 안에 있고, 채점된 칸 전부에 행이 있으며, 지름길 규칙 행의 lift 가 정확히 0 인가(13번 D-1) | ○ |
| `coord_space_contract` | 다섯 칸이 같은 좌표 규약(`ABS_ORIG`)을 선언하는가 | × (기록) — 파일럿 통합형이 NORM_1000 시절 것 |
| `recovery_denominator` | 회복률 분모 ≥ 3·시드 sd | 시드 sd 가 있으면 ○, 시드 1세트면 기록(헤드라인 금지) |
| `content_free_gate` | 전역 Macro-F1 이 content-free 천장 통과선(0.9199)을 넘는가 | `gate_status` 가 `적용`일 때만 ○ (지금 `판정_대기`) |
| `required_tags` | 필수 로깅 태그 11종 | 채점 단계 × (run 태그가 없다) |
| `sweep_curve_recorded` | 등록 격자 전 점이 채점된 칸마다 있고, 임계 독립 헤드라인 지표의 회복률이 산출됐는가(판정 1) | ○ |
| `macro_ap_baseline_paired` | macro-AP 를 실으면 무내용 대조선이 **나란히** 실렸는가(판정 22번 §6-2-1). **통과선 게이트가 아니다** — 칸이 대조선을 넘으라고 요구하지 않는다 | ○ |
| `p9_source_separation` | P9 가 출처별(N-crop/N-tile) 오탐률·차·CI·TOST 를 병기하고, 전역 표의 규격 지름길 각주가 자동 생성됐는가(판정 2) | ○ |

`skipped` 는 `passed` 와 구분해 센다. 등록됐는데 안 불린 게이트가 있으면
`tests/test_gate_registry.py` 가 깨진다.

## 3-1. 헤드라인은 곡선과 임계 독립 지표다 (판정 1 · 22번)

**단일 임계 한 점은 확증 기준이 아니다.** 시드 1 에서 연합↔로컬평균의 대소가 Macro-F1·
결함 놓침 두 축 모두에서 임계에 따라 뒤집혔다(17번 §3-5). 그래서 채점기는 세 가지를 함께 낸다.

| 블록 | 무엇 | 역할 |
|---|---|---|
| `threshold_independent` | 하한(`conf_floor`)에서 낸 `map_50`·`map_50_95`·`macro_ap` 와 그 회복률 | **`map_50` 만 확정 대표**(아래 §3-3). 운용점을 고르지 않는다 |
| `curve` | 등록 격자 전 점의 `macro_f1`·`miss_rate`·`class_jaccard`·`n_boxes` + 임계별 회복률 + 대소 뒤집힘 | **칸 비교.** 데이터를 전부 보인다 |
| `metrics` | 운용점 예시(`conf`)의 전 지표 | 기록. **확증 기준 아님** |

**격자는 사전등록 대상이고 단일 임계는 아니다.** 격자는 결과와 무관하게 정할 수 있어
사후 선택이 아니다. 정본은 `configs/base.yaml`(A 소관)의 `conf_sweep_grid` 이고 채점기는
읽기만 한다 — 미등록이면 `evaluation.params.CONF_SWEEP` 폴백이며 산출물의
`conf_sweep_source` 가 그 사실을 밝힌다.

**"임계 독립"의 범위.** 운용 conf 임계에는 독립이다(PR 곡선 전 구간 적분). 그러나 export
하한·NMS IoU·매칭 IoU·`max_det`·모집단에는 여전히 의존한다 — 산출물의
`threshold_independent.still_depends_on` 이 그 목록이다.

## 3-2. 무내용 대조 — 지표마다 "안 보고도 나오는 값"을 함께 낸다 (22번 §5)

높은 값이 곧 학습의 증거가 아니다. 이 평가셋에서는 **화소를 한 번도 열지 않는 규칙**
(`image_id` 512분위 최빈 코드, `evaluation/content_free.py`)이 분류 축에서 다섯 칸을
전부 이긴다(macro-AP 0.9582 대 최고 칸 0.8229 — 17번 §12-2). 그래서 축마다 대조선을 안다.

| 축 | 무내용 대조선 | 상태 |
|---|---|---|
| 위치 `map_50` | 0.0044 (구간별 중앙 박스) | **선다** — 최고 칸의 1/112 |
| 분류 `macro_ap` | **0.9582** | **넘지 못한다** — 재판정 대상 |
| 판별 `Δ`·`Δ_AUC` | **정의상 정확히 0** | 지름길이 통과할 수 없는 유일한 축 |

**점수 구성은 산출 전에 못박는다**(17번 §11, 커밋 `3f8d5cb`). 대조선 규칙은 `train+val`
정답만으로 적합하고 평가셋을 열지 않는다 — 열면 대조선이 오라클이 되어 게이트가 무의미해진다.
구간 배정은 A 의 절단점(`evaluation/strata.bins_for`)을 그대로 쓰고 D 가 분위를 다시 만들지 않는다.

**판별력**(`evaluation/discrimination.py`)은 출처(`N-crop`)를 상수로 묶은 구간에서 잰다.
`Δ` 는 임계 한 점의 발화율 차라 **임계에 의존한다**(격자 안에서 칸 순위가 뒤집힌다 —
17번 §12-5). 임계 없이 같은 질문을 묻는 형태가 `gini`(= `2·AUROC − 1`)이고, 지름길은
상수 점수라 여기서도 정확히 0 이다. **승격 여부는 별도 판정 사항이다.**

## 3-3. 어느 수가 대표인가 — 판정 22번 §6-2 (main dd430ae)

무내용 대조 결과를 받아 대표 지표를 다시 정했다. **산출물이 이 규칙을 스스로 싣는다**
(`headline_policy`) — 표만 읽고 인용하는 사람에게 규칙이 보여야 하기 때문이다.

| 축 | 지위 | 규칙 |
|---|---|---|
| 위치 `map_50` | **유일한 확정 대표** | 회복률을 이 축으로만 말한다 |
| 분류 `macro_ap` | **보조** (대표에서 내려감) | 무내용 대조선 **병기 필수**, 단독 인용 금지. 분류 축에는 대표 숫자를 두지 않는다 |
| 판별 `Δ` | **선별용** | "다섯 칸 전부 신뢰구간 하한이 0 위" 한 문장으로만. 순위·비율로 읽지 않는다 |
| 판별 `Δ_AUC` | **승격 후보** | 시드 3세트 집계 시점에 판정. 그때까지 값만 쌓는다 |

**`HEADLINE_POLICY`(`scripts/probe/score_cells.py`)를 고치는 것은 채점 기준을 고치는 것이다** —
판정 없이 바꾸지 마라. `macro_ap_baseline_paired` 게이트가 병기와 이 지위를 함께 본다.

**대조선과 `Δ_AUC` 는 채점 한 번에 함께 나온다.** 별도 스크립트를 기억해서 돌리는 구조면
시드 2·3 에서 조용히 빠진다. 대조선 파일이 없으면 채점기가 만들고(`prereg_recomputed_v1.json`
선배치와 같은 규칙), `Δ_AUC` 는 하한 레코드에서 매번 다시 낸다.

## 3-4. 채점기가 자기 코드 지문을 남긴다 (27번 §1-0-1)

여러 시드를 한 표에 모으려면 **같은 채점기 코드로 나왔는지**를 확인할 수단이 있어야 한다.
채점 파라미터가 같아도 코드가 다르면 같은 기준이 아니고, 보고서의 "같은 코드로 돌렸다" 는
작성자의 진술이지 검증이 아니다. 그래서 산출물이 `scorer_code` 를 싣는다
(`evaluation/provenance.py`).

| 항목 | 값 |
|---|---|
| 대상 | `evaluation/**/*.py` + `scripts/probe/score_cells.py` (디렉터리 규칙) |
| 방식 | 경로 + 파일 바이트 sha256 을 정렬 순서로 이어 붙여 다시 sha256 → `combined` |
| 판정 | 집계기(`aggregate_seeds.py`)가 시드 간 `combined` 을 대조한다. **다르면 멈춘다** |
| 예외 | `--allow-code-drift` 는 불일치를 **기록하고** 진행한다. 조용히 넘어가는 경로는 없다 |
| 없을 때 | 이 규칙 이전 산출물은 필드가 없다. 멈추지 않고 "독립 확인 불가" 로 보고한다 |

**`sys.modules` 를 쓰지 않는 이유:** 채점기가 함수 안에서 지연 임포트를 하므로 실행 경로마다
모듈 목록이 달라진다. 같은 코드에서 다른 해시가 나오면 지표의 전제가 무너진다.

**소급하지 않는다.** 이미 나온 산출물에 지금 계산한 값을 끼워 넣으면 그것은 당시 코드의
지문이 아니다. git 커밋 해시도 함께 남기지만 **판정 근거가 아니다** — 더러운 트리에서는
코드를 대표하지 않는다.

## 3-5. 회복률의 부트스트랩 CI 와 `recovery_reportable` 의 세 값 (36번·09-16 판정)

대표 숫자(3세트 평균 회복률 `R̄`)를 실어도 되는지는 **세 줄**로 판정한다 — 트립와이어 ①,
회복률 CI ②③④, 대표 채택(별도 판정). 집계 산출물 `aggregate_v{1,2}.json` 의
`recovery.map_50.denominator_gate` 가 그 세 줄을 담는다.

**`recovery_reportable` 은 세 값이다. 소비자(E 의 표 생성 등)는 셋을 구분해야 한다.**

| 값 | 뜻 | 언제 |
|---|---|---|
| `null` | **미판정.** CI 가 없거나, 있어도 이 집계의 입력과 결속되지 않는다 | `recovery_ci_v1.json` 이 없을 때 · 아래 결속 검사 중 하나라도 어긋날 때(사유는 `recovery_ci_rejected.reasons`) |
| `true` | 게재 가능 — ①과 ②③④ 전부 충족 | CI 가 결속되고 집계기 재판정으로 통과한 뒤 |
| `false` | 게재 불가 — ①~④ 중 미달 | CI 가 결속됐고 하나라도 미달 |

`null` 을 "게재 불가" 로 읽으면 안 되고, `tripwire_only_reportable: true` 를 게재 가능으로
읽어도 안 된다(C 34번 M17). 두 필드는 `recovery_reportable_policy` 가 산출물 안에서 설명한다.
`true` 도 대표 채택이 아니다 — 집계본 최상위 `public_status` 가 "CI 산출 완료, 코드·출처 최종 검수
및 대표 채택 대기" 로 그 상태를 적는다.

**결속 검사(검토 §18, 37번 §8).** 집계기는 CI 산출물을 이름만 보고 쓰지 않는다.
아래 셋을 **모두** 돌리고, 하나라도 어긋나면 CI 를 쓰지 않는다(`null`).

1. 입력 결속 — CI 의 `binding`(시드 목록, 시드별 산출물 원시 바이트 sha256, 채점기 지문과 규칙 표기)이
   집계기가 지금 읽은 파일로 만든 블록과 같은가. 생성기와 집계기는 같은 함수(`read_artifact`·`make_binding`)를 쓴다.
2. T1 항등 — 시드 × 칸 전부의 기록이 있고, 모두 통과했고, 차가 0 이며, 기준값이 지금 파일의 값과 같은가.
3. 점추정 — R̄, 시드별 R·D, 칸별 map_50 이 지금 파일로 다시 계산한 값과 `POINT_TOLERANCE`(1e-12) 안에서 같은가.

기록된 `ci_pass` 는 쓰지 않는다. 집계기가 구간·반폭·미정의 수에서 ②③④를 다시 판정한다
(`evaluation/recovery_ci.py` `rejudge_ci_rules`). 재표집 횟수·난수 시드·**신뢰수준(0.05)·구간 방식
(백분위)** 도 등록 조건이라, 기록이 `evaluation/prereg.py` 의 값과 다르면 쓰지 않는다(C 42번 I-1).
원시 레코드 해시(CI 가 캐시를 만들며 잰 값)도 채점 산출물의 `input_records` 와 맞댄다(m-2).

공개 상태(`public_status`)는 다섯으로 가른다 — 통과 "CI 산출 완료, 코드·출처 최종 검수 및 대표 채택 대기",
규칙 미달, 결속 실패(`ci_unbound`), 기록 검사 실패(`ci_invalid` — 항등·점추정·등록 조건·자기모순), 미산출.

**판 번호와 보존(C 42번 I-2).** 집계본·CI 이름은 입력 산출물 파일명의 판(`_v<n>.json`)에서 뽑는다 —
`score_cells_v3.json` → `aggregate_v3.json`·`recovery_ci_v3.json`. **대상 파일이 이미 있으면 계산 전에
멈추고**, 쓸 때도 배타 생성이라 덮지 않는다. 채점기도 v2 부터 같다. 기존 `seed3set/recovery_ci_v1.json`
은 이 규칙 이전에 **v2 산출물로** 만든 CI 다 — 이름은 그대로 보존하고, 기본 경로로는 읽히지 않는다.

**CI 산출:** `scripts/probe/recovery_bootstrap.py --artifact score_cells_v2.json`. pycocotools
매칭을 (칸, 시드)마다 한 번만 하고 재표집마다 집계만 다시 한다(`evaluation/recovery_ci.py`).
중복도 전부 1 의 집계가 채점기 `map_50` 과 **비트 단위로** 같아야 하고(항등 검사), 다르면 CI 를
내지 않고 멈춘다. 규칙 ②③④와 그 등록 경위는 `evaluation/prereg.py` `RECOVERY_CI_*` 에 있다 —
값을 보기 전 제안, 3세트 채점 뒤 등록, 공식 사전등록 아님.

**v2 산출 경로:** `score_cells.py score --artifact-version v2` 는 `score_cells_v2.json` 에 쓴다.
세 시드를 한 코드 상태로 다시 채점할 때 v1 을 덮어쓰지 않기 위한 경로다. v2 산출물은
`scorer_code`(시작·끝 지문, `stable`)와 `input_records`(레코드 sha256)를 싣는다.

## 4. 산출물 (`<채점 dir>/`)

| 파일 | 무엇 |
|---|---|
| `{cell}_s{seed}.jsonl` | 계약 #4 레코드 (검출은 `predict`, 통합형은 `score` 가 어댑터로 생성) |
| `score_cells_v1.json` | 본채점. `scorer_code`(채점기 소스 지문) · `metrics` · `stratified` · `gates_evaluated` · `exit_code` · `coord_health` · `recovery` · `regression` · P9 · `discrimination` · `discrimination_threshold_free` · `content_free_baseline` · `decomposition` · `headline_policy` |
| `prereg_recomputed_v1.json` | 사전등록 상수 동결본 재산출(자동 선배치). `snapshot_digest` 로 출처 고정 |
| `stratified_compare_v1.json` | 층화 상세 (K 사다리 · 구간별 행 · 지름길 규칙) |
| `verify_filter_parity_v1.json` | 하한+필터 ≡ 직접 추론 동치의 표본 재확인 (곡선의 전제) |
| `sweep/` · `sweep_detection_conf_v1.json` | conf 스윕 |
| `content_free_baselines_v1.json` | 무내용 대조선 3종 + `__shortcut__` AP + 위치 축 + 등록 상수 자기 검사 |
| `discrimination_sweep_v1.json` | 판별력 Δ 의 임계 곡선(격자 전 점) + 임계 독립 `Δ_AUC` + 칸 대비(같은 재표집 짝지음) |
| `seed3set/recovery_ci_v1.json` | 회복률 부트스트랩 CI — 항등 검사·`R̄`/`R_s`/`D_s`/칸별 mAP CI·규칙 ②③④ 판정·지문·입력 해시·입력 결속 `binding`(37번 §8 이후 산출분만 — 기존 v2 CI 에는 없다) |
| `coco_evalimgs_{tag}_s{seed}.npz` | pycocotools 매칭 캐시 — **중간물**, 봉인하지 않는다 |
| `score_cells_v2.json` | 한 코드 상태 재채점(v1 보존). `scorer_code.stable`·`input_records` 포함 |
| `score_cells_v3.json` | 머지된 main 커밋에서 줄끝 정규화 지문으로 재채점(v1·v2 보존). 이미 있으면 채점하지 않는다 |
| `seed3set/aggregate_v<n>.json` · `seed3set/recovery_ci_v<n>.json` | n = 입력 산출물의 판. 이미 있으면 멈춘다(`recovery_ci_v1.json` 만 예외적으로 v2 입력 — 이름 규칙 이전 산출) |

## 5. 첫 산출물 감사 (시드 1)

13번 §4 가 지목한 항목을 전부 수행한다. 핵심 지표 3개(Macro-F1 · 놓침 · 층화 Macro-F1)는
채점기와 독립인 경로로 소수 6자리 재계산·대조한다. `coord_health` · 경계 이탈 · 스키마
실패율의 붕괴 신호 유무를 명시한다. 시드 1 수치는 결론이 아니다 — 시드 3세트 집계 전까지
경향만.
