# 21. 총괄 게이트 — R(④ 서버측 라운드 경계 재개) 착지 판정 (2026-09-07 총괄)

대상: wt/C-bdr `db125c9`·`d4fab47`(30 파일, +4,562/−116; `detection/budget_audit.py` 포함). 학습 경로는 4 파일이고 `configs/` 는 diff 0 이다.
근거: 총괄 판독(strategy 오프셋 3곳 · round_wiring 마감 강화 · server_app 사전검증·원자 저장 · main_det 재개 경로), D 19번 독립 검수
(착지 가능, Critical 0 · Important 2 · Minor 9, 시험 97 passed, 실물 실패 원장 두 벌에 판정기를 읽기 전용으로 돌림 → k=30·k=22 정확), C 18번 착지 절차.

## 0. 판정

**조건부 통과 — 코드는 착지할 자격이 있다. 다만 채점이 등가인지는 창에서 T-eq 를 돌려야만 확인된다(15번 G12).**

| # | 조건 | 주체·시점 |
|---|---|---|
| 1 | D Important 1: ④ 마감 정책의 원자 행 검사에 `val_loader_workers == 0`(v2 이상 라운드, 클라이언트 3) 단언을 넣고 시험을 붙인다 | C, 착지 전 wt/C-bdr 커밋 |
| 2 | D Important 2: T-eq 변형 2(시뮬레이션 하드킬, `finally` 미실행 — 실제로 죽은 두 번이 이 형태였다)를 **필수**로. 변형 1·2 를 모두 통과해야 R 을 채택한다 | C, 18번 §2-7 정정 + 창 |
| 3 | 창의 T7 재확인(B+R 최종 FAB, ④ 1라운드): `global_r001.npz` sha256 `5c0378a1…` 일치, **그리고** audit.json `failures` 에 든 항목이 **모두 "총 epoch ≠ N" 종류**여야 한다(1라운드만 돌린 시험이라 당연하다 — 감사가 클라이언트마다 한 줄을 내므로 3건이 정상, `budget_audit.py:240-242`). 행 수·지표 버전·서버 지표 집합·G14(cudnn_deterministic 1 / benchmark 0 / batch 32)·val_loader_workers 가 하나라도 실패하면 착지시키지 않는다 | C 실행, 총괄 판정 |
| 4 | T-eq 가 어긋나면 R 을 쓰지 않는다: wt/C 에 머지하지 않고 시드 3 은 B 트리(`cedb194`)로 간다. 텐서별 최대 편차를 10번에 기록 | 총괄 |
| 5 | 시드 2 ④(B 트리, 마감 검사 없음) 라운드 1 마감 직후 원자 로그 `val_loader_workers` 0 ×3 · 트레이서 워커 8 수동 확인(≈ 09-08 11시) | C |
| 6 | E 계약 통지 답(20번) 수령 — 이것 때문에 착지를 막지는 않는다. G6 통지를 남긴 흔적 | E |

## 0-1. 조건 이행 기록

- 조건 1·2: wt/C-bdr `8f9cf67`(09-07 11:13) — 마감 `val_loader_workers == 0` 단언(audit.json `b_evidence`), T-eq 변형 2 → 1 순서로 둘 다 필수(18번 §2-7, 창 2~2.5 h), Minor 9·참고 3 반영, 가짜 Grid 로 실제 `Strategy.start` 루프를 도는 CPU 시험 추가(18번 §5 에 남아 있던 미결 1건 해결).
- 조건 6: E 20번(`bad3ad1`·`b54cac3`) 수령 — 재개할 때 접두를 인용해 오는 곳은 제자리 원자 로그의 measured 행이고, 끝나지 않은 라운드 행은 없다. ⑦ 재개도 없고, 조건 표는 부록에 있다.
- 남은 조건: 3(창 T7 재확인)·4(T-eq)·5(시드 2 B-ON 확인). D 는 시드 2 구간 회귀 전량 재실행 때 `8f9cf67` 의 마감 단언·ops 스크립트 변경분을 함께 본다.

## 1. 총괄 판독 요지

- 오프셋은 `configure_train`(super 호출 전)·`aggregate_train`·`aggregate_evaluate` 진입부에 있고 기본 0 이면 동일. 배선이 반쪽만 되면 회계 `record` 가 ValueError 를 내서 잡아낸다.
- 마감 강화(`finalize_accounting(policy, resume_state)`)는 ④·재개 run 에만 켜진다. 라운드별 지표 버전(v1 8 / v2 9 / v3 13)을 상수에서 유도해 세고, 신규 라운드는 현재 버전을 강제한다. 재개면 접두 npz sha256·절단 후 원장 sha256·세그먼트 커버를 다시 대조한다. audit.json 에는 키를 더하기만 한다. 마감 산출물도 tmp→replace 로 쓴다.
- 서버 사전검증은 런처 판정을 다시 하고(k·잔해·지표 버전·`resume_log.jsonl` 마지막 이벤트 대조) 실패 시 audit.json 만 쓴다. `global_r{k}.npz` 인덱스 적재·`assert_compatible`·유한성·l2 상대 1e-9. ⑦ 은 k>0 이면 거부한다.
- 런처는 재개를 명시했을 때만 돈다: 잔해·부분 원장이 있으면 fresh 기동을 거부하고 재개 명령을 출력하며, `--reason` 필수, k 이중 대조, 구코드 접두 거부, 무진전 거부, `_quarantine/<stamp>/` 격리, 재개 로그 별도, progress.jsonl 세그먼트 행(rc 0/1/2/3). `NEED_COMMIT_GB = 50`, 6 h 대기 후 SystemExit(종전 동작 유지).
- 정상 경로(시드 3 ④ 원 run)에 새로 얹히는 것: 라운드별 accounting.csv 원자 덤프 · `global_index.jsonl` · npz tmp→replace · 마감 strict 검사. 마지막 마감 검사는 11 h 완주가 끝나야 처음 도니까, 조건 3 으로 오탐을 미리 잡는다.

## 2. 창 절차 (18번 승인, 순서 고정)

launcher 정지(시드 2 CHAIN_DONE 전, 체인 프로세스는 유지) → 창 진입 확인 → **총괄이 wt/C-bdr → wt/C 머지** → CPU 회귀 전량 → T-smoke-resume(Ray) → T7 재확인(조건 3) → T-eq 변형 1·2 → 시드 3 체인 + export 기동. 실패 시그니처가 나오면 멈추고 보고한다. 창 길이 ≈ 2~2.5 h(T-eq 2 변형). **시작 시각은 시드 2 실측으로 다시 계산했다: ≈ 09-09 04:15**(C 09-08 08:31 보고 — 시드 2 ② 10 h 27 m 실측 반영). 조건 5(시드 2 ④ B-ON 확인) ≈ 09-08 17:12, 시드 3 완주 ≈ 09-10 15:00. ③ 완주 실측이 나오면 C 가 확정한다.

## 3. 별건(착지 무관)

- venv SSD 이관, flwr 판올림, T-hist 스크립트(`compare_global_npz.py`, 시드 1 완주 뒤 CPU) — 15번 §4·18번 §5.
- 시드 1 창 실행본은 `logs/recover/`(미추적)이고 시드 3 부터 `scripts/ops/` 본을 쓴다.
