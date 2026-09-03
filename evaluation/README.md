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
| 2 | `uv run python scripts/probe/score_cells.py score --profile main --pilot <C 산출> --out <채점 dir>` | **본채점.** 전역 지표 + **id 구간 층화 블록(같은 산출물 안)** + 게이트 전수 + prereg 상수 선배치. 종료 코드가 판정이다(§2) |
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

`skipped` 는 `passed` 와 구분해 센다. 등록됐는데 안 불린 게이트가 있으면
`tests/test_gate_registry.py` 가 깨진다.

## 4. 산출물 (`<채점 dir>/`)

| 파일 | 무엇 |
|---|---|
| `{cell}_s{seed}.jsonl` | 계약 #4 레코드 (검출은 `predict`, 통합형은 `score` 가 어댑터로 생성) |
| `score_cells_v1.json` | 본채점. `metrics` · `stratified` · `gates_evaluated` · `exit_code` · `coord_health` · `recovery` · `regression` · P9 |
| `prereg_recomputed_v1.json` | 사전등록 상수 동결본 재산출(자동 선배치). `snapshot_digest` 로 출처 고정 |
| `stratified_compare_v1.json` | 층화 상세 (K 사다리 · 구간별 행 · 지름길 규칙) |
| `sweep/` · `sweep_detection_conf_v1.json` | conf 스윕 |

## 5. 첫 산출물 감사 (시드 1)

13번 §4 가 지목한 항목을 전부 수행한다. 핵심 지표 3개(Macro-F1 · 놓침 · 층화 Macro-F1)는
채점기와 독립인 경로로 소수 6자리 재계산·대조한다. `coord_health` · 경계 이탈 · 스키마
실패율의 붕괴 신호 유무를 명시한다. 시드 1 수치는 결론이 아니다 — 시드 3세트 집계 전까지
경향만.
