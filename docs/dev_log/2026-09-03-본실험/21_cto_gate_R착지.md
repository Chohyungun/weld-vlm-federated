# 21. 총괄 게이트 — R(④ 서버측 라운드 경계 재개) 착지 판정 (2026-09-07 총괄)

대상: wt/C-bdr `db125c9`·`d4fab47`(30 파일, +4,562/−116; `detection/budget_audit.py` 포함). 학습 경로 4 파일·`configs/` diff 0.
근거: 총괄 판독(strategy 오프셋 3곳 · round_wiring 마감 강화 · server_app 사전검증·원자 저장 · main_det 재개 경로), D 19번 독립 검수
(착지 가능, Critical 0 · Important 2 · Minor 9, 시험 97 passed, 실물 실패 원장 두 벌에 판정기 읽기 전용 실행 → k=30·k=22 정확), C 18번 착지 절차.

## 0. 판정

**조건부 통과 — 코드는 착지 자격이 있고, 채점 등가는 창의 T-eq 만이 준다(15번 G12).**

| # | 조건 | 주체·시점 |
|---|---|---|
| 1 | D Important 1: ④ 마감 정책의 원자 행 검사에 `val_loader_workers == 0`(v2 이상 라운드, 클라이언트 3) 단언 + 시험 | C, 착지 전 wt/C-bdr 커밋 |
| 2 | D Important 2: T-eq 변형 2(시뮬레이션 하드킬, `finally` 미실행 — 실제 두 사망의 형태)를 **필수**로. 변형 1·2 모두 통과해야 R 채택 | C, 18번 §2-7 정정 + 창 |
| 3 | 창의 T7 재확인(B+R 최종 FAB, ④ 1라운드): `global_r001.npz` sha256 `5c0378a1…` 일치 **그리고** audit.json `failures` 가 "총 epoch ≠ N" 한 건뿐. 행 수·지표 버전·서버 지표 집합·G14(cudnn_deterministic 1 / benchmark 0 / batch 32)·val_loader_workers 실패가 하나라도 있으면 착지 차단 | C 실행, 총괄 판정 |
| 4 | T-eq 불일치 → R 비채택: wt/C 에 머지하지 않고 시드 3 은 B 트리(`cedb194`)로. 텐서별 최대 편차를 10번에 기록 | 총괄 |
| 5 | 시드 2 ④(B 트리, 마감 검사 없음) 라운드 1 마감 직후 원자 로그 `val_loader_workers` 0 ×3 · 트레이서 워커 8 수동 확인(≈ 09-08 11시) | C |
| 6 | E 계약 통지 답(20번) 수령 — 착지 차단 조건은 아니나 G6 통지 흔적 | E |

## 1. 총괄 판독 요지

- 오프셋은 `configure_train`(super 호출 전)·`aggregate_train`·`aggregate_evaluate` 진입부에 있고 기본 0 이면 동일. 반쪽 배선은 회계 `record` ValueError 로 잡힌다.
- 마감 강화(`finalize_accounting(policy, resume_state)`)는 ④·재개 run 에만 켜진다. 라운드별 지표 버전(v1 8 / v2 9 / v3 13)을 상수에서 유도해 세고, 신규 라운드는 현재 버전을 강제한다. 재개면 접두 npz sha256·절단 후 원장 sha256·세그먼트 커버를 재대조한다. audit.json 은 추가 키만. 마감 산출물도 tmp→replace.
- 서버 사전검증은 런처 판정을 다시 하고(k·잔해·지표 버전·`resume_log.jsonl` 마지막 이벤트 대조) 실패 시 audit.json 만 쓴다. `global_r{k}.npz` 인덱스 적재·`assert_compatible`·유한성·l2 상대 1e-9. ⑦ 은 k>0 거부.
- 런처는 명시 재개 전용: 잔해·부분 원장이 있으면 fresh 기동을 거부하고 재개 명령을 출력, `--reason` 필수, k 이중 대조, 구코드 접두 거부, 무진전 거부, `_quarantine/<stamp>/` 격리, 재개 로그 별도, progress.jsonl 세그먼트 행(rc 0/1/2/3). `NEED_COMMIT_GB = 50`, 6 h 대기 후 SystemExit(종전 동작 유지).
- 정상 경로(시드 3 ④ 원 run)에 새로 얹히는 것: 라운드별 accounting.csv 원자 덤프 · `global_index.jsonl` · npz tmp→replace · 마감 strict 검사. 마지막 것이 완주 11 h 뒤에 처음 실행되므로 조건 3 이 오탐을 미리 잡는다.

## 2. 창 절차 (18번 승인, 순서 고정)

launcher 정지(시드 2 CHAIN_DONE 전, 체인 프로세스는 유지) → 창 진입 확인 → **총괄이 wt/C-bdr → wt/C 머지** → CPU 회귀 전량 → T-smoke-resume(Ray) → T7 재확인(조건 3) → T-eq 변형 1·2 → 시드 3 체인 + export 기동. 실패 시그니처면 멈추고 보고. 창 길이 ≈ 1.5~2 h, 시작 ≈ 09-08 23시.

## 3. 별건(착지 무관)

- venv SSD 이관, flwr 판올림, T-hist 스크립트(`compare_global_npz.py`, 시드 1 완주 뒤 CPU) — 15번 §4·18번 §5.
- 시드 1 창 실행본은 `logs/recover/`(미추적)이고 시드 3 부터 `scripts/ops/` 본을 쓴다.
