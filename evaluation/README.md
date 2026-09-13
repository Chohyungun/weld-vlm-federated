# 채점 절차 — 다섯 칸 공통 채점기 (트랙 D)

본실험 채점이 밟는 순서와, 각 단계가 무엇을 보증하는지를 적는다. 근거는
`docs/dev_log/2026-08-22-데이터확정/83_체크리스트_D.md` 와
`docs/dev_log/2026-09-03-본실험/13_2차폐쇄검증.md` §3 말미(채점 전 조치 3건).

**채점 코드는 한 벌이다.** 다섯 칸 출력을 계약 #4 레코드로 바꾼 뒤 `evaluation.score`
하나로 채점한다(개발규약 3-7). 칸마다 채점 코드를 따로 두지 않는다.

## 0. 전제

- `git merge main --ff-only` 로 최신 코드. 저장본 `score_cells_v1.json` 의 `params` 에
  `profile`·`model_cfg`·`predict_chunk`·`imgsz_source` 네 키가 있어야 최종 코드 산출이다
  (13번 D-8 봉인).
- GPU 를 쓰지 않는다. `predict` 만 CPU 추론이고 나머지는 순수 채점이다.
- 채점 디렉터리(`--out`)는 실험마다 따로 둔다 — 파일럿은 `outputs/pilot_d`, 본실험은
  총괄 지시서의 경로. 저장본을 덮어쓰지 않는다.
- **프로파일을 명시한다.** 본실험은 `--profile main`(YOLO11s · 640). 빠뜨리면 파일럿
  기본값(YOLO11n · 416)이 잡히고, 가중치 형상 불일치로 시작조차 하지 않는다.

## 1. 순서

| # | 명령 | 보증하는 것 |
|---|---|---|
| 1 | `uv run python scripts/probe/score_cells.py predict --profile main --pilot <C 산출> --out <채점 dir>` | 검출 3칸 하한(0.01) 추론 → 계약 #4 레코드. 내부 256장 청킹. `--at-conf` 는 운용 임계 레코드 |
| 2 | `uv run python scripts/probe/score_cells.py score --profile main --pilot <C 산출> --out <채점 dir>` | **본채점.** 전역 지표(운용점 예시) + **곡선 전 구간** + **임계 독립 헤드라인 지표** + id 구간 층화 블록 + 규격 지름길 각주 + 게이트 전수 + prereg 상수 선배치. 종료 코드가 판정이다(§2) |
| 3 | `uv run python scripts/probe/score_cells.py sweep --profile main ...` | conf 스윕 — 단일 임계 결론 금지. 하한 1회 추론 + 사후 필터의 동치는 `--verify-parity` 로 실측 |
| 4 | `uv run python scripts/probe/stratified_compare.py --k 64 --ladder --pilot <C 산출> --out <채점 dir>` | 층화 **상세** 산출물(구간별 행 포함, `stratified_compare_v1.json`). 총괄 판정 6 의 대조표가 이 파일이다 |
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
| `sweep_curve_recorded` | 등록 격자 전 점이 채점된 칸마다 있고, 임계 독립 헤드라인 지표의 회복률이 산출됐는가(총괄 판정 1) | ○ |
| `macro_ap_baseline_paired` | macro-AP 를 실으면 무내용 대조선이 **나란히** 실렸는가(총괄 판정 22번 §6-2-1). **통과선 게이트가 아니다** — 칸이 대조선을 넘으라고 요구하지 않는다 | ○ |
| `p9_source_separation` | P9 가 출처별(N-crop/N-tile) 오탐률·차·CI·TOST 를 병기하고, 전역 표의 규격 지름길 각주가 자동 생성됐는가(총괄 판정 2) | ○ |

`skipped` 는 `passed` 와 구분해 센다. 등록됐는데 안 불린 게이트가 있으면
`tests/test_gate_registry.py` 가 깨진다.

## 3-1. 헤드라인은 곡선과 임계 독립 지표다 (총괄 판정 1 · 22번)

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
| 분류 `macro_ap` | **0.9582** | **넘지 못한다** — 총괄 재판정 대상 |
| 판별 `Δ`·`Δ_AUC` | **정의상 정확히 0** | 지름길이 통과할 수 없는 유일한 축 |

**점수 구성은 산출 전에 못박는다**(17번 §11, 커밋 `3f8d5cb`). 대조선 규칙은 `train+val`
정답만으로 적합하고 평가셋을 열지 않는다 — 열면 대조선이 오라클이 되어 게이트가 무의미해진다.
구간 배정은 A 의 절단점(`evaluation/strata.bins_for`)을 그대로 쓰고 D 가 분위를 다시 만들지 않는다.

**판별력**(`evaluation/discrimination.py`)은 출처(`N-crop`)를 상수로 묶은 구간에서 잰다.
`Δ` 는 임계 한 점의 발화율 차라 **임계에 의존한다**(격자 안에서 칸 순위가 뒤집힌다 —
17번 §12-5). 임계 없이 같은 질문을 묻는 형태가 `gini`(= `2·AUROC − 1`)이고, 지름길은
상수 점수라 여기서도 정확히 0 이다. **승격 여부는 총괄 판정 사항이다.**

## 3-3. 어느 수가 대표인가 — 총괄 판정 22번 §6-2 (main dd430ae)

무내용 대조 결과를 받아 총괄이 대표 지표를 다시 정했다. **산출물이 이 규칙을 스스로 싣는다**
(`headline_policy`) — 표만 읽고 인용하는 사람에게 규칙이 보여야 하기 때문이다.

| 축 | 지위 | 규칙 |
|---|---|---|
| 위치 `map_50` | **유일한 확정 대표** | 회복률을 이 축으로만 말한다 |
| 분류 `macro_ap` | **보조** (대표에서 내려감) | 무내용 대조선 **병기 필수**, 단독 인용 금지. 분류 축에는 대표 숫자를 두지 않는다 |
| 판별 `Δ` | **선별용** | "다섯 칸 전부 신뢰구간 하한이 0 위" 한 문장으로만. 순위·비율로 읽지 않는다 |
| 판별 `Δ_AUC` | **승격 후보** | 시드 3세트 집계 시점에 판정. 그때까지 값만 쌓는다 |

**`HEADLINE_POLICY`(`scripts/probe/score_cells.py`)를 고치는 것은 채점 기준을 고치는 것이다** —
총괄 판정 없이 바꾸지 마라. `macro_ap_baseline_paired` 게이트가 병기와 이 지위를 함께 본다.

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

## 5. 첫 산출물 감사 (시드 1)

13번 §4 가 지목한 항목을 전부 수행한다. 핵심 지표 3개(Macro-F1 · 놓침 · 층화 Macro-F1)는
채점기와 독립인 경로로 소수 6자리 재계산·대조한다. `coord_health` · 경계 이탈 · 스키마
실패율의 붕괴 신호 유무를 명시한다. 시드 1 수치는 결론이 아니다 — 시드 3세트 집계 전까지
경향만.
