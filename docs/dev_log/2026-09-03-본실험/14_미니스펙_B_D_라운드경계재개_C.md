# 14. 미니스펙 — B(미사용 검증 로더 워커) · D(Ray 상주 발자국) · R(④ 서버측 라운드 경계 재개) — 총괄 게이트용

2026-09-07 · 트랙 C · 상태: **게이트 대기 — 구현 착수 금지.** 총괄 판정(main `a4772ea`, 09-05 의사결정로그)
"B·D·서버측 라운드 경계 재개는 시드 2·3 착수 전 미니스펙→게이트" 와 09-06 23:50 보충 지시 7항(안정 궤도·10번 커밋 이후 재기동)에 따른 산출물.
선행 커밋: 10번 `62a23ed`(재기동 #3 안정 궤도)·`694ee16`(게이트 정정). 본 문서와 함께 10번 §7-4·§7-5 의 워커 단가·절감치를 정정한다(§0-4).

## 0. 요약 — 총괄이 먼저 볼 것

### 0-1. 세 주제 결론

| 주제 | 권고 | 이득(실측 기준) | 채택 조건 | 등가 논증 상태 |
|---|---|---|---|---|
| **B** 미사용 검증 DataLoader 워커 제거 | **조건부 채택** | 클라이언트 학습 구간 플래토 **−13.2~14.0 GB**(검증 워커 12 × 1.10~1.17 GB 실측 합 13.72) + 액터 내 가시 증분 ≤ 0.3 GB(추정). 벽시계 ≈ −65 s/라운드(≈ −7 %, 추정) | ④ 경로 오라클(`flwr run` 1라운드, `global_r001.npz` sha256 `5c0378a1…` bit 일치) 통과 + **체인 경계(시드 경계)에서만** 착지 + 후속 launcher 정지 창 확보. 창을 못 만들면 **비채택(A1 유지)** | 소스 수준에서 "학습 로더·워커 시드·전역 RNG·모델/옵티마이저 상태 bit 불변" 을 적대 3렌즈가 무너뜨리지 못함(§B-4). 실측(B 코드로 잰 bit 동일)은 0건 → 오라클이 조건 |
| **D** Ray 상주 발자국 축소 | **비채택** (문서 정정만) | Ray 상주 ≈ 5.5 GB 중 flwr 정식 표면(`init-args-num-cpus=2`)으로 닿는 것은 유휴 워커 **0.6~0.8 GB** 뿐. plasma object store 4.33 GB(Windows 커밋 계상 **확인**)는 저장소 밖 env 로만 닿아 기각 | (경로 A 를 B 부속 후보로 넘길지만 결정) | 집계·통신량·벽시계·시드에 Ray 설정을 읽는 경로 없음(§D-2-④). object store 크기가 다른 두 run(5.19 vs 4.64 GB)의 라운드 1 bit 동일이 실측 |
| **R** ④ 서버측 라운드 경계 재개 | **조건부 채택** (명시 재개 전용) | 메모리 0 GB. 중단 시 회수 = 닫힌 라운드 k × 15.66 분(09-04 형태면 7.83 h) | 명시 트리거(`cell4 --seed N --resume-from K --reason`, 자동 재개 없음) + 4조건 닫힌 라운드 판정기 + 잔해 전수 격리 + 마감 재대조 + **T-eq(실기 등가 시험) 통과**가 "재개 run 을 무중단 run 과 동등하게 채점" 의 전제 | 파이썬 가시 부분(매 라운드 트레이너·옵티마이저·EMA·로더 신규, 시드 = f(base, round, client), npz 왕복 무손실, 오프셋 1곳 주입)은 소스로 성립. **"재사용 액터의 (k+1)번째 라운드 = 새 프로세스 첫 라운드"** 는 C++ 상태(cuDNN/cuBLAS/할당자)라 미증명 → T-eq 로만 닫힘 |

### 0-2. 완결성 비판이 찾은 절 간 상충과 조립자 통합안 (게이트에서 확정)

세 절을 따로 읽으면 맞지만 시드 2·3 에 함께 올리면 깨지는 지점 7건. 조립자(C)의 통합안이며 최종 결정은 게이트.

1. **원자 로그 행 수·기대 집합.** B 가 `val_loader_workers` 지표 행을 더하면 라운드당 27 → 30 행인데 R 은 27·108 을 리터럴로 박았고 시드 1 접두(지표 8종)는 R 의 판정에서 "미완비" 로 거부된다. → 행 수는 `3×len(CLIENT_ROUND_METRICS)+3` 로 상수에서 유도, 접두 판정은 "접두 첫 라운드가 쓴 지표 집합" 기준(버전 표: 시드 1 = 8, 시드 2·3 = 9). T-judge 에 "8-지표 접두 위 9-지표 코드" 픽스처.
2. **회계 감사 불변식의 적용 범위.** R 의 `seed == derive_seed`·`budget_fired_at == (r+1)E−1`·`optimizer_steps == epochs × ceil(n/32)` 는 칸 한정 없이 `audit()` 에 들어가면 스모크(seed=base, steps=5, fired=−1)·⑦ 이 즉시 실패한다. → **④(`cell == "sep_fed"`) 한정** 또는 `fl/resume_ledger.py` 로 이동. 감사 기준 강화 자체가 불변조건 "회계 감사 기준" 변경이라 게이트 항목(G6).
3. **착지 순서·FAB 동일성·GPU 창.** B T7 과 D T5 는 같은 run 이고 R 의 pyproject 키 추가는 FAB 해시를 바꾼다. → **세 변경을 한 커밋·한 FAB 로 착지** → CPU 회귀 전량(B T1~T5·T8, R T-offset~T-keys·T-dup, D T3·T4 는 경로 A 시만) → B T6(GPU ~10 분) → R T-smoke-resume(Ray, 수 분) → **B T7 = D T5**(④ 1라운드 오라클, 최종 FAB 해시 == 시드 2 run FAB) → R T-eq(GPU 수십 분) → 시드 2 기동. 총 창 ≈ **1.5~2 h**(B 단독 ~30 분이 아님).
4. **후속 launcher 정지 절차(소유자 C).** `logs/recover/chain_follow.ps1`(git 미추적)은 시드 1 `CHAIN_DONE` 을 120 s 폴링해 즉시 export + 시드 2 chain 을 띄우며 정지 훅이 없다. 창을 만들려면 CHAIN_DONE(≈ 09-07 13:20) **전에** launcher 프로세스를 내려야 한다. 게이트가 그 전에 끝나지 않으면 (a) GPU 를 비워 두고 기다림, (b) 시드 2 를 현행 코드로 시작하고 세 변경은 시드 3 부터 — 어느 쪽인지 총괄 결정(G1).
5. **`_headroom_wait(need_commit_gb=16)` 문턱.** B(+41/+55)·D(≥55)·R(재개 preflight 60 하드코딩)이 제각각 → 값·위치(체인 공통 vs 재개 전용)를 한 질문(G8)으로. R 본문의 60 은 "게이트 값" 참조로 읽는다. 어느 값이든 launcher 문턱은 스테이지 진입 시점(골)만 보고 플래토를 막지 못한다는 B §1 지적은 유지.
6. **속도 축 플래그 스키마.** 이음매(R)·외부 GPU 점유(10번 §5-1)·`val_loader_workers`(B)·격리 세그먼트(R) 플래그 4종과 요약 평균 제외 규칙이 절마다 다르다 → 한 표로 통일해 D·E 에 통지(G9). 원칙은 지시서대로 **제외가 아니라 표시**, 요약 평균만 플래그 라운드 제외.
7. **`FIXED_OVERRIDES["workers"]=8` 등록(B Q5) ↔ R 의 "round_runner.py 무변경" 전제.** 값 불변 키 추가라도 프로파일 고정 항목이 늘어난다 → 한 답을 두 절이 공유(G10); 승인 시 R 의 "손대지 않음" 행을 "값 불변 키 추가 제외" 로 읽는다.

**즉시 조치 요청(게이트 무관, 총괄 확인).** 실물 `_resume/sep_fed/r005_c1/resume_ep0010.pt`(95 MB, 01:44) 가 관측됐다 — 클라이언트 학습 도중 생기는 epoch 경계 덤프로, 클라이언트 완주 시 지워진다(01:5x 현재 0 파일). 10번 §7-2 의 "91 디렉터리 / 0 파일" 은 두 번 다 첫 epoch 안에서 죽은 시점 운이었다. **R 착지 전 시드 1 ④ 가 죽으면** 같은 명령 재기동은 `ResumeIdentity` 일치로 그 파일을 조용히 소비해 bit 비동일 궤적이 된다(`detection/resume.py:281-287`, RNG 상태 미저장) → 재기동 전 원장 이동 + **`_resume/sep_fed/**/*.pt|*.tmp` 격리 필수**(현 `post_reboot_C.ps1` 0단계는 파일이 있으면 exit 2 로 멈추므로 우회는 안 된다). 그 뒤 선택은 라운드 1 재실행(13.1 h 손실) vs R 게이트 통과까지 GPU 유휴 — 총괄 결정(G5).

### 0-3. 통합 게이트 질문 (세 절의 §8 + 비판의 누락 항목을 합쳐 중복 제거)

| # | 질문 | 출처 |
|---|---|---|
| G1 | **창·착지·launcher.** 세 변경을 한 커밋·한 FAB 로 착지하고 §0-2-3 순서로 총 1.5~2 h GPU 창을 시드 1 완주(≈13:20) 뒤에 두는가. launcher 정지·확인·재기동은 C 가 한다. 게이트가 늦으면 GPU 를 비워 둘지, 시드 2 를 현행 코드로 시작하고 시드 3 부터 적용할지 | B Q1·D T5·R Q3·비판 |
| G2 | **시드 경계 동일 원칙.** B·D 경로 A·R 의 적용 시드 경계를 같게 둘 것인가(혼합 "시드 2 = B on·R off, 시드 3 = B on·R on" 금지). 실험 조건 표에 시드별 커밋·FAB·호스트 구성(OMP_NUM_THREADS ②③ 기본 / ④ 액터 2 포함) 열 승인 | B Q2·R Q3·D Q2·비판 |
| G3 | **B 채택.** 조건부 채택(④ 오라클 T7 bit 일치 + 체인 경계)을 승인하는가. 증빙 기록 방식(`RoundResult.val_loader_workers` + ④ 메트릭 + 원자 로그 지표 행 + ②③ meta.json, accounting.csv 열 제외) 승인 | B Q3·Q4 |
| G4 | **D 비채택** 승인. 경로 A(`init-args-num-cpus=2`, −0.6~0.8 GB)를 B 에 부속시킬지 버릴지(부속 시 시드 1↔2·3 정본 분기 1건 + 의사결정로그 1행 + T3~T6) | D Q1·Q2 |
| G5 | **R 트리거·시드 1 사망 시.** 명시 전용(자동 재개 없음, 상한은 "무진전 거부")으로 확정하는가. 시드 1 이 R 착지 전에 죽으면 접두(FAB 31149ece) 위에 게이트 통과본(다른 FAB)으로 재개를 허용하는가, 아니면 라운드 1 재실행인가 | R Q1·Q2·비판 |
| G6 | **회계 감사 기준 강화** — `seed==derive_seed`·`budget_fired_at`·`optimizer_steps` 기대값·`restored:` 분류를 승인하고 **④ 한정**으로 두는가; ④ 의 `resumed_from_epoch ≠ None` 셀을 failure 로 승격하는가; `optimizer_steps` 검사를 ②③ 회계에 소급(시드 1 실물 재감사)하는가; audit.json 신규 필드(`resume_events`·`segments`·`restored_cells`)의 D·E 파서 계약 통지를 게이트 조건으로 두는가 | R Q4·Q7·Q8·비판 |
| G7 | **원자 로그 스키마.** B 의 `val_loader_workers` 행 추가(27→30, 스모크·⑦ 에 −1 행)를 승인하고 R 의 기대 집합을 버전화(시드 1 = 8 지표 호환)하는가 | 비판 |
| G8 | **`_headroom_wait(need_commit_gb)`** 값(≥ 55~60)과 위치(체인 공통 vs 재개 전용)를 한 번에 — 코드 변경이라 별도 게이트 | B Q6·D Q4·R |
| G9 | **속도 축 플래그 스키마·각주 문안** 통일(이음매·외부 GPU·`val_loader_workers`·격리 세그먼트, 요약 평균 규칙, 격리된 열린 라운드 행의 표본 여부) + B 의 칸 비대칭 각주(④ 가 트레이너 생성 비용을 150회 지불, 시드 1 ④ 의 10~14 %) | B Q7·R §5·비판 |
| G10 | `FIXED_OVERRIDES["workers"] = 8` 명시 등록 허용 여부(값 불변, 문서화 목적; R 의 무변경 전제와 조정) | B Q5·비판 |
| G11 | **10번 정정 문안 승인**(694ee16 기준, 본 커밋에 반영): 워커 단가 학습 1.77~1.85 / 검증 1.10~1.17, B 절감 −13.2~14.0(+α ≤ 0.3), §7-4 B·D 행, plasma 계상 확정 | B Q3·D Q3·비판 |
| G12 | **T-eq 불일치 시** 재개 run 을 "통계적 등가 + 이음매 각주" 로 격하해 채택하는가, 비채택하는가(의사결정로그 기록) | R Q6 |
| G13 | R 의 k == R 경계(50 라운드 닫힘·audit.json 부재)를 "마감 전용 재개"(서버 루프 0 라운드, 회계 복원 후 finalize)로 정의하는가 | 비판 |
| G14 | 클라이언트 계측 추가(MetricRecord 에 amp·cudnn_deterministic·cudnn_benchmark·batch 실효값)를 같은 게이트에 묶는가 | R Q5 |
| G15 | **별건 등록.** (a) Ultralytics OOM 자동 배치 반감(`trainer.py:511-541`)이 회계에 흔적 없음 → 함정 #1 접촉점 등록 + `effective_batch` 가드; (b) Ultralytics `settings.json sync: true` → GA4 익명 이벤트 전송 가능성(레드라인 3 인접, `sync=False` 판단 — 설정 파일은 저장소 밖); (c) `detection/dataset_view.py:87-88` "val 은 훈련 중 접근하지 않는다" 부정확(트랙 A 정정 통지); (d) warm 액터의 클라이언트 간 잔존 9.6~10.2 GB 별도 항목; (e) 복구 스크립트(`chain_follow.ps1`·`post_reboot_C.ps1`)의 저장소 이관과 R 의 `[ROUND i/(R−k)]` 로그 시그니처에 맞춘 감시 파서 수정 담당; (f) 시드 1 export 와 시드 2 ② 병행 유지 여부(export 커밋 미실측) | B Q8·Q9·D Q5·Q6·비판 |

### 0-4. 이 커밋에서 함께 정정한 10번 문구 (G11 대상)

- §7-5: DataLoader 워커 20개 = **학습 8 × 1.77~1.85 GB(합 14.4) + 검증 12 × 1.10~1.17 GB(합 13.7)** (트레이서 `_procs.csv` 00:26:55, ppid 36524; spawn 시각으로 두 집단 분리 — 학습 00:21:20~41, 검증 00:21:56~00:22:19). 검증 워커 제거 시 절감 **−13.2~14.0 GB**(앞선 −21~22·−29~30·§7-4 의 −12~19 는 전부 정정).
- §7-4 B 행·D 행: 실측치와 본 문서 권고로 갱신. §7-1 의 "object store 계상 여부 미확인" → 계상됨(raylet 스폰 초 비사설 커밋 +4.33 GB ≈ 선언 4.64 GB 의 93 %, 실사용 76 MB).

### 0-5. 작성 방식 (다관점 병렬 + 적대 검증 워크플로)

Workflow `wf_4ba20718-649`, **agent 18개 전부 `fable`/`xhigh` 명시**, 읽기 전용(파일 수정·python 실행·GPU 작업 금지, 루트 재귀 grep 금지, 실행 중 run 의 원장은 읽기만). 단계: 설계(주제별 2관점 독립, 6) → 적대 검증(B 3렌즈: RNG·계산 불변 / 공정성·프로토콜 / 필요성·비용편익·구현위험; D 2렌즈: flwr 표면 실현성 / 효과·부작용; R 3렌즈: 등가 / 원장·감사 둔갑 경로 / Flower API·함정 #1; 8) → 종합(주제별 1, 상충 사실은 종합자가 소스 재독으로 판정; 3) → 완결성 비판(1). 결과: 완료 18/18, **오류·거부·건너뜀·빈 결과 0**(⚠ 없음), 도구 호출 881, 토큰 4.69 M, 71 분. 설계안 6개 중 폐기 0·"필수 수정 후 생존" 6; 적대 검증이 기각한 주장은 각 절 §4 표에, 종합자가 직접 판정한 상충(예: B 의 raytune 등록 여부, R 의 results.csv unlink)은 본문에 남겼다. 조립자(C)는 비판이 지적한 인용·수치 오류만 손봤고(부록 B 목록) 설계 내용은 바꾸지 않았다.

**실측 근거 위치:** 10번 §5-1·§7-1·§7-2·§7-5; `logs/trace_chain_s1_20260906_235623{,_events,_procs}.csv`(1 s 트레이서); `logs/sysmon_s1_cell4.csv`(60 s, 외부 GPU 점유 열); `outputs/main_c/seed1/fl/sep_fed_failed_20260904/`(30 라운드 유효 원장); 진행 중 run `outputs/main_c/seed1/fl/sep_fed/`(읽기만); `%TEMP%\ray\session_2026-09-07_00-09-07_*/logs/`; `.venv` 설치본 ultralytics 8.4.120 · torch 2.11.0+cu128 · flwr 1.33.0 · ray 2.55.1 · numpy 2.2.6.

---

## B. 미사용 검증 DataLoader 워커 제거 — ④ 클라이언트 학습 구간 호스트 커밋 플래토 완화

프레임워크 실독본: ultralytics 8.4.120(`uv.lock:3850-3851`), torch 2.11.0+cu128, flwr 1.33.0, ray 2.55.1(`.venv/Lib/site-packages/*.dist-info`), CPython 3.11.13(`.venv/pyvenv.cfg`). 기계 `NUMBER_OF_PROCESSORS=12`.

### 1. 목적·필요성 (A1+A2 이후 기준으로 정직하게)

- **B 는 정확성에 필요하지 않다.** 검증 로더는 한 번도 소비되지 않고(§2), 그 워커 12개는 순수 낭비다. B 의 가치는 **보험**이다.
- **A1 뒤의 여유 23.2 GB 는 최저 기준선에서 잰 값이다.** 09-07 실측 플래토 80.43 GB @00:27:00(`logs/trace_chain_s1_20260906_235623.csv`), 기준선 25.6 GB 는 재부팅 직후 값이며 09-05 평시 기준선은 42 GB 였다(10번 §7-1 L171-174: 에이전트 세션 프로세스 8.4·WSL 4.1·Overlay 2.4·게임 4·…). 그 기준선으로 돌아가면 플래토 ≈ 42 + 54.8 = 96.8 GB, 여유 ≈ 7 GB.
- **문제는 기동 순간 스파이크가 아니라 클라이언트당 ~5분 플래토다.** `logs/sysmon_s1_cell4.csv` 00:11:41~00:20:16 76.2→79.2 GB, 00:23:29~00:26:41 79.6→80.1 GB 로 유지되고 골(44.65 @00:21:20, 44.82 @00:27:45)은 ~1분. 트레이서 1 s 로도 spawn 창(00:21:20~00:22:40, GPU 할당 직후까지) 최대 77.76 @00:22:40 < 정상 상태 최대 80.43 @00:27:00. 즉 라운드 벽시계의 ~80 % 가 플래토이고 A2 는 70시간 상시 준수여야 의미가 있는데 권고로 하향됐다(§7-4 L245).
- **구 체제의 실효 상한은 63.2 가 아니라 ~91 GB 였다**(§7-1 L149-150 "63.2→74.3→91.1", 사망 90.8/91.1 L154). A1 의 신뢰 순증은 ≈ +12.5 GB + 확장 경쟁 제거다. "A1 +40 GB = B 의 2배" 는 과대 비교(§4 기각).
- **시드 2·3 은 무인 ~70 h · 병행 허용 구간이다**(§5 L86; `logs/recover/chain_follow.ps1` L18-21 은 시드 1 export(`scripts/main_det.py:519` torch CPU)를 분리 기동한 직후 L24-26 에서 시드 2 chain 을 띄운다). ④ 는 서버측 라운드 재개가 없어(§7-2 L194) 1회 사망 = 13.1 h + 하루 조율. `_headroom_wait`(`scripts/main_det.py:148-168`)는 스테이지 진입 시점(골)에서 커밋 여유 16 GB 만 보고 `cmd_chain` 이 스테이지마다 한 번 부르므로(L450) 플래토 +54.8 GB 를 막지 못한다.
- **이득은 −13.7 GB 실측**(검증 워커 12 × 1.10~1.17 GB, §2) + 액터 내 가시 증분 ≤ 0.3 GB. 플래토 80.4 → ≈ 66.7 GB, 기준선 42 에서도 여유 ≈ 21 GB. 부수로 클라이언트당 검증 워커 spawn 21~23 s 제거(≈ −65 s/라운드 ≈ −7 %).
- 비용: 함정 #1 접촉점 +1(공개 훅 `get_dataloader`), FAB 재패키징(~4.5분), 시드 1↔2·3 의 호스트 메모리·타이밍 비균일(계산은 동일해야 하며 오라클로 실증), GPU 게이트 창 ~25분(현재 launcher 구조상 0분).

**결론: 조건부 채택** — ④ 경로(`flwr run` 1라운드) 오라클 bit 동일 통과 + 체인 경계 착지 + launcher 정지 창 확보가 조건이며, 창을 만들 수 없으면 비채택(A1 유지)이 체인 중간 착지보다 낫다.

### 2. 현재 동작 (파일:줄 근거)

- **무조건 생성.** `engine/trainer.py:291-296` `self.test_loader = self.get_dataloader(val, batch_size*2, mode="val")` — `args.val` 은 학습 루프 게이트 L594 에서만 읽힌다. OOM 자동 재시도 L535 도 같은 `_build_train_pipeline`(L278-308) 을 다시 탄다. 학습 로더가 먼저다(L281-283).
- **워커 수 12(16 아님).** `models/yolo/detect/train.py:100` `workers = args.workers*2`(=16 요청) → `data/build.py:363` `nw = min(os.cpu_count()//1, workers, batches)` = min(12, 16, 79) = **12**. val 5,001장(`outputs/main_c/views/client{0,1,2},central/images/val` 각 5,001) / 64 = 79 배치. 트레이서 `n_dl_worker` 최대 20, 클라이언트 기동마다 spawn 8 + 12(events.csv C2: 00:21:20~41 학습 8, 00:21:56~00:22:19 검증 12). 10번 §7-1 L175 "검증 16"·§7-5 L290 은 오류.
- **생성 즉시 spawn·프리페치·핀 스레드.** `build.py:75` `self.iterator = super().__iter__()` → torch `dataloader.py:428-433` → `_MultiProcessingDataLoaderIter.__init__` L1156-1204 워커 `w.start()`(L1192), L1206-1223 핀 스레드, L1250 `_reset(first_iter=True)` → L1294-1295 `prefetch_factor(4, build.py:376) × 12 = 48` 배치 투입. `_RepeatSampler`(build.py:107-124) 라 StopIteration 없이 전부 만들어진다.
- **소비자 없음.** `trainer.py` 의 `test_loader` 참조는 L291(대입)·L646(close) 두 곳(grep 전수). validator 는 `detect/train.py:206-210` → `engine/validator.py:121` 저장만; 로더를 도는 L235·249·285·298 은 전부 `__call__` 안이고 `self.validator(` 호출처는 L868(`validate`)·L951(`final_eval`) 뿐 → `detection/fed_trainer.py:279-286`·`288-290` 이 no-op. 실패 run 로그(`flwr_sep_fed_failed_20260904.log:81-85`)에 `val: Slow image access`(159회)·`val: Scanning …val.cache… 5001 images`·`Using 8 dataloader workers`(L431 은 train_loader 만 셈, 93회 = 31 라운드 × 3) 만 있고 검증 진행 줄 없음.
- **실측 단가(`trace_chain_…_procs.csv` 00:26:55, ppid 36524).** 학습 8개(00:21:20~41 spawn) 사설 1.765~1.845 GB 합 14.39; **검증 12개(00:21:56~00:22:19 spawn) 1.101~1.165 GB 합 13.72**. 액터 36524: 00:22:20 priv 9.558 / ws 4.540 → 00:22:22 priv 16.684 / ws 4.555(+7.13 GB, 상주 0) → 00:25:00 17.006 / 5.166 — 이 점프는 `atomic_log.csv peak_vram_gb` 7.83/7.82/7.72 와 일치하는 WDDM 의 GPU 할당 커밋 계상이며, 검증 프리페치의 가시 증분은 priv ≤ 0.3 GB. 시스템 커밋 Δ(00:22:20→40: 65.73→77.76 = +12.03) ≈ 사설 합 Δ(36.37→48.29 = +11.92) 라 숨은 공유메모리 커밋도 없다.
- **임계 경로(events.csv, C2).** 학습 spawn 21 s → 검증 데이터셋 구축 15 s → 검증 spawn 23 s → 학습 시작(00:22:22 GPU 할당). 라운드 2 도 동일(C1 00:31:41~57 / 00:32:20~41, C2 00:39:27~43 / 00:39:54~00:40:15, C3 00:44:34~48 / 00:45:04~). 검증 워커는 클라이언트 종료까지 산다(exit 00:27:35~36, 00:39:21~23).
- 우리 래퍼: `FIXED_OVERRIDES`(`detection/round_runner.py:39-60`) `val/plots/save=False`, `workers` 키 없음 → Ultralytics 기본 8(`cfg/default.yaml:21`); 실패 run `runs/r000_c0/args.yaml:14` `workers: 8`, `:39 val: false`, `:12 cache: false`, `:86 cls_pw: 0.0`.

### 3. 설계 — 채택안

**결합안**: 관점 1 의 기제(직접 `build_dataloader(workers=0)`, 데이터셋 생성 유지, 공유 상태 무변경) + 관점 2 의 런타임 단언·필수 증빙 필드·`flwr run` 오라클·체인 경계 착지.

| 파일 | 심볼 | 변경 | 근거 |
|---|---|---|---|
| `detection/fed_trainer.py` | `FedDetectionTrainer.get_dataloader`(신설, 접촉점 6) | `mode=="val"` 이면 `build_dataset` 은 stock 그대로, `build_dataloader` 를 `workers=0` 으로 직접 호출; `num_workers!=0` 이면 `RuntimeError`; `self.val_loader_workers` 기록. 학습 경로는 `super()` 위임 | stock `detect/train.py:78-105`; 호출자 `trainer.py:281·291`; 재진입 L535 |
| `detection/fed_trainer.py` | `__init__` L183-194 | `self.val_loader_workers: int \| None = None` | None = 미계측, 0 = B on, 실측값 = B off |
| `detection/fed_trainer.py` | 모듈 표 L9-15 · 클래스 docstring L162-168 | 행 추가 + "넷"→여섯(optimizer_step 이 이미 5) | — |
| `detection/round_runner.py` | `RoundResult` L92-125 · 반환부 L327-351 | `val_loader_workers: int \| None = None` + `getattr(trainer, "val_loader_workers", None)` | ②③ meta.json 은 `train_cell.py:57` `asdict` 로 자동 포함 |
| `fl/client_det.py` | metrics L71-100 | `"val-loader-workers": float(v if v is not None else -1)` (L82-84 의 −1 관례) | MetricRecord float |
| `fl/round_wiring.py` | 기록기 L86-98 | `"val_loader_workers": float(m.get("val-loader-workers", -1))` 지표 행 | 원자 로그는 long format(`fl/atomic_log.py:45-46`), `audit_rounds` L176-185 는 (round, client) 만 봄 → 시드 1 원장과 호환. **accounting.csv 열은 추가하지 않는다**(`budget_audit.py:61-97` 고정 열) |
| `tests/test_detection_trainer.py` L55-68 | 접촉점 계약 | 목록에 `get_dataloader` 추가, `DetectionTrainer.get_dataloader` 와 다름 단언 | — |
| `tests/test_val_loader_stub.py`(신설) | T2~T5 | §7 | `_Toy` 패턴 `tests/test_loader_seed_independence.py:39-50` |
| `docs/의사결정로그.md` · 10번 §7-1 L175·§7-4 L247·§7-5 L289-293 | 수치 정정 + 결정 | "검증 12 워커 × 1.10~1.17 GB = −13.7 GB, 플래토 프레임" | §2 |
| (무변경) `FIXED_OVERRIDES` · `scripts/main_det.py` · `fl/server_app.py` · `fl/strategy.py` · `pyproject.toml [tool.flwr]` · `detection/resume.py` | — | `FIXED_OVERRIDES["workers"]=0` 은 학습 로더까지 0 → C 기각 사유와 동일, 금지 | `detect/train.py:100` 이 한 값을 두 로더에 씀 |

```python
# detection/fed_trainer.py — 접촉점 6 (신설)
def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
    if mode != "val":
        return super().get_dataloader(dataset_path, batch_size, rank, mode)   # 학습 경로 무변경
    # 검증 로더는 validate/final_eval 이 no-op 이라 소비되지 않는다(trainer.py:594·640, 본 파일 validate/final_eval).
    # stock 은 workers*2(cpu 캡→12)·batch*2 로 만들고 InfiniteDataLoader.__init__ 이 즉시 spawn+48 배치 프리페치를
    # 띄운다(build.py:75, torch dataloader.py:1156-1295). 데이터셋 생성은 stock 그대로 둔다 — get_img_files →
    # check_file_speeds 가 전역 random 을 1회 소비하므로(base.py:184, data/utils.py:86) 없애면 S1 상태가 갈라진다.
    from ultralytics.data import build_dataloader
    from ultralytics.utils.torch_utils import torch_distributed_zero_first
    with torch_distributed_zero_first(rank):                                 # detect/train.py:91-92 복제
        dataset = self.build_dataset(dataset_path, mode, batch_size)
    loader = build_dataloader(dataset, batch=batch_size, workers=0, shuffle=False,   # detect/train.py:97-105 에서
                              rank=rank, drop_last=False, device=self.device)         # workers 만 0 (compile=False → drop_last=False)
    if getattr(loader, "num_workers", None) != 0:                            # 조용한 회귀 금지
        raise RuntimeError(f"검증 로더 워커 {loader.num_workers} != 0 — 상류 배선이 바뀌었다(B 전제 위반)")
    self.val_loader_workers = int(loader.num_workers)                        # 증빙 → RoundResult
    return loader
```

- 호출 순서(변경 후): `train_round`(round_runner:281) → `.train()` → `_do_train` L421 `_setup_train` → L312 모델 → L363 AMP → **L402 `_build_train_pipeline`** → L281 학습 로더(무변경) → **L291 검증 로더 → 오버라이드** → L297-308 옵티마이저·스케줄러 → L403 validator(저장만) → L404 EMA → L405 class weights(`cls_pw=0` → `detect/train.py:175-176` 즉시 반환) → L412 stopper → L413 resume(no-op) → 우리 (a)~(h) `fed_trainer.py:218-276` → 학습 루프.
- `workers=0` 실로더 안전성: `build.py:363` nw=0 → `:376 prefetch_factor=None` → torch `:429-430 _SingleProcessDataLoaderIter`(L773-781, 프로세스·핀 스레드·프리페치 없음, `_base_seed` draw 는 L706-710 에서 그대로 1회); `close()` `build.py:96-99` 는 `_workers` 없어 no-op, `__del__` L86-94 동일; `trainer.py:646-648` close 루프 통과.

**대안과 기각 이유**
- `self.args.workers` 를 val 호출 동안만 0(관점 2 기제): 동작은 같으나 `self.model.args is self.args`(`detect/train.py:147`) 공유 객체를 잠시 변형 — 명시 호출이 더 보수적. 단언·증빙만 가져온다.
- `_build_train_pipeline` 오버라이드: 옵티마이저·스케줄러까지 복제, OOM 재진입(L535)도 타는 사설 API — 기각.
- `test_loader=None`/데이터셋 생성 생략/지연 프록시: validator·close 루프 덕타이핑 파괴 + `check_file_speeds` 의 S1 소비 1회 소실, 프록시는 L646 `hasattr` 이 깨움 — 기각(폴백으로만 보관).
- 뷰 `val:` 을 ≤64장으로 줄여 `batches<=1 → nw=0`: 트랙 A 산출물·`data` 인자·args.yaml 이 시드마다 갈림 — 기각.
- `_setup_train` 직후 `test_loader.close()`: spawn·프리페치 피크는 그대로 — 기각.
- `FIXED_OVERRIDES["workers"]=0` / 학습 `workers` 8→4(C): 워커 시드 배정(`worker.py:258`)·증강 난수열 변경 — 총괄 기각 유지.
- ④ 전용 플래그: 칸별 코드 경로 분기 = "구조 내 동일" 위반; 10번 §7-4 L247 의 "④ 만 바꾸면" 은 채택하지 않는다.
- '하지 않는다'(A1 유지): 게이트 창을 만들 수 없을 때의 정당한 기본값으로 유지(§8-1).

### 4. 등가 논증과 적대 검증 결과

**목표 명제.** B 전후에 학습 로더 배치열·워커 증강 난수열·주 프로세스 전역 RNG 궤적·모델/옵티마이저 상태가 bit 동일하다. 검증 로더 내부 상태는 출력이 소비되지 않으므로 결과에 닿지 않는다.

**살아남은 주장(적대 3건 전부 holds, 본 검토 실독 확인)**
- `_base_seed` 는 로더 자신의 generator 에서만 뽑힌다: torch `dataloader.py:706-710` `.random_(generator=loader.generator)`, `build.py:364-365` 호출마다 새 `torch.Generator()` + 상수 시드, `:380` 전달, torch `:406`. 학습(S5)·검증(S6) 생성기는 별개 객체.
- 학습 로더가 먼저 생성·반복자 확정(`trainer.py:281` → `build.py:75`) → 학습 워커 base seed(`worker.py:258-265` + `seed_worker` `build.py:229-233`)와 셔플(`RandomSampler(generator)` torch `:393-394`)은 검증 로더 유무와 무관. 배치→워커 배정은 `:1277` 라운드로빈 + `in_order=True`(`:267`) 라 타이밍 무관.
- 검증 로더 생성이 전역 RNG 를 소비하는 지점은 `check_file_speeds` 의 `random.sample` 1회뿐(`base.py:117→184`, `data/utils.py:86`); 캐시 분기(`base.py:320·353`)는 `cache=False` 라 미진입(`base.py:137-145`, `default.yaml:19`); `set_rectangle` 은 argsort(`base.py:377`); val 변환은 `LetterBox`+`Format` 생성자(`dataset.py:318-331`, RNG 는 `augment.py:2456` `__call__` 워커 측). 채택안은 `build_dataset` 을 stock 그대로 호출하므로 이 1회 소비가 보존된다. 게다가 S1 의 하류 소비자가 없다(`multi_scale=0.0` `default.yaml:40` → `detect/train.py:120-122` 미진입, `plots=False`).
- 워커 spawn 은 부모 전역 RNG 를 안 쓴다 — **관점 1 주장 vs 적대 3 "미확인" 을 내가 stdlib 로 판정: 성립.** `multiprocessing/synchronize.py:48` `_rand = tempfile._RandomNameSequence()`, `:122-124` `_make_name`; `tempfile.py:46` `from random import Random as _Random`, `:147` 자체 인스턴스, `:155` `choices`; Windows `Pipe` `connection.py:552-556` → `:79` `tempfile.mktemp`; `popen_spawn_win32.py:54` `_winapi.CreatePipe`. `_shared_seed` 는 IterDataPipe 전용(torch `:645-650`).
- 검증 로더 생성 이후 단계(L297-415)에 로더 의존 RNG 소비 없음; 콜백은 전부 불활성 — clearml/comet/dvc/neptune/wandb/tensorboard 패키지 부재(site-packages 확인), platform 은 `api_key ""` 로 `platform.py:282-285` 반환·`:447-449` ctx 없음(`:495` `validator.metrics` 미도달). **raytune — 적대 1·2 "등록되나 무동작" vs 적대 3 "등록조차 안 됨" 을 내가 판정: 등록된다.** `ray/air/session.py:1` 이 `from ray.train._internal.session import *` 이라 `raytune.py:8` `from ray.air import session` 은 서브모듈 import 로 성공(적대 3 은 `__init__.py` 만 grep), `ray/tune` 존재 → `raytune.py:36-42` 등록; `:31` `get_session()`(`ray/train/_internal/session.py:527·612`) 이 None 이라 무동작. 설계 영향 없음.
- 검증 로더를 읽는 미차단 분기 없음(§2). `_handle_nan_recovery` L1020-1058·`resume_training` L1060-1063·`_close_dataloader_mosaic`·`LoaderReseed`(`fed_trainer.py:148-156`)·`resume.py:254-265·289-295` 전부 `train_loader` 만.
- 결정론의 실제 근거: `torch_utils.py:685` 는 benchmark 를 주석 처리했고, `fl/seeding.py:87-91` `deterministic_torch()` 를 `fl/run_gates.py:76-79` 가 호출, `round_runner.py:254-256` 이 매 라운드 실호출. §7-5 L296-300 의 라운드 1 bit 동일(n=2, 다른 날·게임 병행)이 메모리·타이밍 차이가 수치에 닿지 않음을 보인다(k=0 한정).
- 오라클 기준값 실재: `sep_fed_failed_20260904/accounting.csv` 2-4행 param_l2 192.23213244102195 / 194.41977624944028 / 190.76837433025506, steps 1488/916/402, updates 1302/801/352; `global_r001.npz` sha256 `5c0378a1288619930c13e54746664c48993ec5345d977b7b28bbf546e38a0ce2`(본 검토 sha256sum 재확인); `np.savez` `fl/server_app.py:268` 결정적.

**기각된 주장 (설계에 반영)**
| # | 주장 | 판정 근거 | 반영 |
|---|---|---|---|
| K1 | [관점 1] 검증 워커 단가 1.78~1.86 GB → −21~22 GB, 총 −23~26 GB | procs 00:26:55 검증 12개 1.101~1.165(합 13.72); 1.78~1.86 은 학습 워커. 10번 §7-5 L290 오류 승계 | 이득 −13.7 GB |
| K2 | [관점 1] 핀 프리페치 2.3~3.8 GB / [관점 2] α ≤ 7 GB | 액터 +7.13 GB 는 ws 평탄 + peak_vram 7.8 GB 일치 → GPU 커밋 계상. 가시 증분 priv ≤ 0.3, ws ≤ 0.6 | α ≤ 0.3 GB(추정), 총 ≈ −14 |
| K3 | [관점 1] 벽시계 −1.5분 ≈ −1~2 % | 산술 모순(1.5/15.66 = 9.6 %); 실측 검증 spawn 창 21~23 s/클라이언트 → −63~69 s/라운드 ≈ −7 % | §5 각주 |
| K4 | [두 설계안·10번 §7-5] "기동 피크" 프레임 | spawn 창 최대 76.65 < 플래토 80.43; sysmon 76~80 GB 상시 | "클라이언트 학습 구간 플래토" |
| K5 | [관점 2] A1 +40 GB = B 의 2배 | 구 실효 상한 ~91 GB(§7-1 L149-154) → 신뢰 순증 ≈ +12.5 | §1 |
| K6 | [10번 §7-1 L175·§7-5 L290] 검증 16 워커 / −29~30 GB | cpu_count=12 캡(`build.py:363`), `n_dl_worker` 최대 20 | 문서 정정 |
| K7 | [적대 3] raytune 미등록 | `ray/air/session.py` 존재 | 서술 정정만 |

**조건부 주장**
- (g) 디바이스 할당 실패 없음: WDDM 은 GPU 할당을 호스트 커밋에 계상하므로 커밋 고갈 국면에서 cuDNN 워크스페이스 폴백이 수치를 바꿀 가능성(torch C++ 미독·추측). B 가 만드는 차이가 아니라 줄이는 위험이나, 오라클 불일치 시 먼저 배제한다.
- 오라클의 프로세스 문맥: §7-5 는 액터↔액터만 쟀다. 일반 프로세스 `train_round`(관점 1 T5)는 Ray 환경 차이(PYTHONPATH `raybackend.py:112-121`, OMP 스레드 주입 — 미확인)를 끌어들이므로 **판정 시험은 `flwr run` 1라운드(④ 경로)** 로 하고, 일반 프로세스 A/B 는 보조.
- 라운드 k≥1 등가는 소스 논증에 의존(§7-5 도 k=0 한정) — B 와 독립인 기존 미확인.

### 5. 적용 범위와 비교 가능성

- **범위 ②③④ 전부, 시드 경계에서만.** 세 칸이 같은 `train_round` → `FedDetectionTrainer` 를 통과(`detection/train_cell.py:24·95·148`, `round_runner.py:242·281`; 생성자 호출처는 이 한 곳). ②③ 도 같은 검증 12 워커 구조라 플래토 이득은 동일(벽시계 이득은 ②③ 에서 무시 가능 — 트레이너 생성 3회·1회).
- **착지 시점 기제(두 설계안이 놓친 것).** ④ 는 작업트리가 아니라 FAB 스냅샷에서 코드를 읽는다: `flwr/simulation/app.py:270` `install_from_fab` → `:274` `get_project_dir` → `:343` `app_dir`; `flwr/supercore/object_ref.py:177-182` `sys.path.insert`; `raybackend.py:112-121` PYTHONPATH 전파; 실물 `~/.flwr/apps/weld-fl.weld-fl.0.1.0.{ef15ff95,74c59bf1,3f332c2e,31149ece}`. ②③ 은 chain 프로세스가 작업트리를 lazy import(`round_runner.py:242`). 따라서 (i) 지금 wt/C 편집은 돌고 있는 시드 1 ④ 에 닿지 않지만(게이트 전 구현 금지는 별개), (ii) 시드 2 chain 프로세스가 뜬 뒤 ④ `flwr run` 전에 B 가 트리에 들어오면 플래그 없이도 "②③ 구 / ④ 신" 이 된다. **착지 규칙: 시드 2(또는 3) chain 프로세스 기동 전에 커밋·검증 완료.** `chain_follow.ps1` 은 CHAIN_DONE 직후(L14) 시드 2 를 즉시 띄우므로(L24-26) 게이트 창은 launcher 를 내려서 만든다. 커밋 해시와 FAB 해시를 함께 기록한다(§7-3 이 FAB 추적).
- **결과 축(놓침·위치·근거·통신량).** 시드 1(B off, 검증 12) ↔ 시드 2·3(B on, 0) 은 §4 오라클이 bit 동일을 보일 때만 같은 기준. `args.yaml` 은 `__init__`(`trainer.py:150`) 에서 저장돼 두 상태 모두 `workers: 8` 이므로 산출물 구분은 `val_loader_workers` 증빙뿐 — 이 프로젝트의 "설정이 아니라 실제로 돈 값을 남긴다" 원칙(`fed_trainer.py:331-337`, `budget_audit.py:120-131` P9) 에 따라 필수.
- **속도 축.** 시드 1 이 지정 표본(§5 L84). 시드 2·3 원장은 `val_loader_workers` 로 **표시(제외 아님)**, 시드 1 과 합산·평균 금지. B 와 무관한 **칸 비대칭 각주**가 추가로 필요하다: 미사용 검증 로더 비용(데이터셋 구축 10~23 s + spawn 21~23 s ≈ 31~44 s/트레이너)을 ④ 는 150회, ② 3회, ③ 1회 지불 → 시드 1 ④ 13.1 h 의 ≈ 1.3~1.8 h(10~14 %) 가 래퍼 산물(라운드 2 실측: C3 165 s 중 37 s = 22 %). 논문 속도 표는 ④ wall 을 "트레이너 생성 오버헤드 / 학습" 으로 분해해 각주한다. B 채택 시 시드 2·3 ④ 는 ≈ −1분/라운드.
- 회계 감사(`budget_audit`)·집계·통신량 무영향: 검출 클라이언트가 보내는 가중·스텝·epoch 값이 같다.

### 6. 위험과 대응

| 위험 | 등급 | 대응 |
|---|---|---|
| 등가 실측 미완 — B 코드로 잰 bit 동일이 0건 | 높음 | ④ 경로 오라클(T7) 통과 전 적용 금지. 불일치 시 (1) 트레이서로 커밋 고갈·(g) 배제 → (2) 같은 문맥 B off 재실행으로 원인 분리 → 그래도 불일치면 폐기 |
| 체인 중간 착지(②③ 구 / ④ 신) | 중간 | §5 착지 규칙 + launcher 정지. 시드 2 창이 없으면 시드 3 부터 또는 비채택 |
| 상류 변동으로 오버라이드가 조용히 무력화(`workers*2` 규칙·`build_dataloader` 시그니처·`_build_train_pipeline` 구조) | 중간 | 8.4.120 잠금 + 런타임 `RuntimeError` + 카나리 시험(T5) + 증빙 필드 |
| 라운드 45~49 재spawn 파 — `fed_trainer.py:252-254` + `trainer.py:454-456` 이 학습 워커 8개를 접고 다시 띄움(라운드 45 는 2회). 시드 1 은 미관측(라운드 45 ≈ 12:00) | 낮음 | `build.py:101-104` `reset()` 은 close 후 spawn 이라 최대 20 유지; B 는 기준선을 13.7 낮춤. B 무관 기존 사실로 기록 |
| OOM 자동 재시도(`trainer.py:522-541`) — batch 반감 무흔적 + 재구축 시 옛 로더 `__del__` 전 새 spawn 으로 순간 최대 40 워커(B 16; 옛 20 + 새 20, 추정) | 낮음 | B 오버라이드는 재진입에도 적용. **별건**: `RoundResult.effective_batch` 를 회계에 남겨 `FIXED_OVERRIDES["batch"]` 와 대조(§8-9) |
| 이득 과대평가로 A2 판단 왜곡 | 중간 | 본 문서 수치(−13.7 + ≤0.3)로 10번 정정 후 게이트 |
| `validate` no-op 을 누가 되살리면 워커 0 검증이 매우 느려짐(오답은 아님) | 낮음 | docstring 명시; 그 변경은 어차피 등가 게이트 대상 |
| '하지 않는다' 의 위험 — 기준선 42 복귀 시 여유 ≈ 7 GB, 무인 70 h, ④ 재개 없음 | 중간 | 비채택 시 최소 `_headroom_wait(need_commit_gb)` 상향(§8-6) + sysmon 여유 < 10 GB 경보 + A2 준수 요청 |

### 7. 검증 계획 (게이트 통과 후 구현·실행; 현재는 어떤 시험도 실행 금지)

| # | 시험 | 파일 | 검증 내용 | 규모 |
|---|---|---|---|---|
| T1 | 접촉점 계약 | `tests/test_detection_trainer.py:55-68` | 목록에 `get_dataloader` 추가, `DetectionTrainer.get_dataloader` 와 다름 | CPU 즉시 |
| T2 | `get_dataloader` 계약 | `tests/test_val_loader_stub.py`(신설) | `object.__new__` + args 스텁(workers=8, device cpu) + `build_dataset` → `_Toy(640)` monkeypatch: val → `num_workers==0`, 반복자 `_SingleProcessDataLoaderIter`, `close()` 무해, `val_loader_workers==0`, `build_dataset` 호출 1회; train → 센티널로 위임 확인. CPU 는 `trainer.py:163-164` 가 workers=0 강제라 실트레이너 대신 스파이 | CPU 즉시 |
| T3 | 전역 RNG 불변 | 동상 | python/numpy/torch RNG 스냅샷 + 학습용 `build_dataloader(_Toy, workers=0, shuffle=True)` generator 상태 저장 → `build_dataloader(_Toy(256), batch=8, workers=2, shuffle=False, device="cpu")` 실제 spawn+프리페치 → 전부 동일, 학습 로더 첫 순열 불변 | CPU 수십 초 |
| T4 | `check_file_speeds` 소비 현상 고정 | 동상 | `random.seed(0)` 후 호출 전후 `random.getstate()` 변화 단언(상류 변경 감지) | CPU 즉시 |
| T5 | 상류 카나리 | 동상 | `ultralytics.__version__=="8.4.120"`; `trainer.py` 의 `self.test_loader` 참조 정확히 2곳·`self.validator(` 2곳; `DetectionTrainer.get_dataloader` 소스에 `self.args.workers * 2`; `InfiniteDataLoader.__init__` 에 `self.iterator = super().__iter__()` | CPU 즉시 |
| T6 | ②③ 경로 소형 A/B | `scripts/verify_val_loader_stub.py`(신설, `verify_resume.py:57-97` 골격) | `profile="pilot"`·yolo11n·resume_verify 뷰(val 353 → pilot batch 2 → val batch 4 → 89 배치 → B off 에서 nw=12, 비공허) 2 epoch × {B off(monkeypatch 무력화), B on} → 텐서별 `np.array_equal`, steps·updates·lr_trace 동일 | GPU ~10분 |
| T7 | **④ 경로 오라클(게이트 판정 시험)** | `flwr run` `num-server-rounds=1`, 스크래치 `project`, 시드 1 main 프로파일, `initial.npz` 읽기만(`init_weights.py:46`) | `global_r001.npz` sha256 `5c0378a1…`, C1/C2/C3 param_l2·steps·updates 일치, 서버 global_l2 192.23750521425777; 1 s 트레이서로 클라이언트당 dl_worker spawn == 8, 플래토 ≤ ~67 GB, 액터 priv 실측(α 확정) | GPU 21~27분(FAB+uv sync ~4.5분 §5-1 L100 + SuperLink 콜드 §7-3 + 라운드 1 15.7~21.9분) |
| T8 | 회귀 전량 | `tests/` | 전부 통과 + `test_detection_fed.py:123-127` 에 `"workers" not in FIXED_OVERRIDES or == 8` 단언 | CPU |
| T9 | 시드 2 라운드 1 사후 대조 | 본실험 자연 발생 | injection_digest·param_l2 로 T7 과 재대조 — T7 대체 아님 | — |

순서: T1~T5·T8 → T6 → T7 통과 후에만 시드 2(또는 3) chain 기동. 전부 시드 1 ④ 완주(≈ 09-07 13:20) 뒤·launcher 정지 상태·GPU 빈 상태에서만.

### 8. 게이트 질문 (총괄 결정)

1. **게이트 창.** 시드 1 CHAIN_DONE 전에 `chain_follow.ps1`(미추적, 정지 훅 없음)을 내려 GPU ~30분(T6+T7) 창을 만들 것인가. 없으면 시드 3 부터인가, 비채택인가.
2. **적용 시드 경계.** 2 / 3 / 없음. (체인 중간 착지 금지는 전제)
3. **10번 문서 정정 승인.** §7-1 L175 "검증 16" → 12; §7-4 L247 "−12~19" 및 §7-5 L289-293 "20개 × 1.78~1.86 / −29~30 / 기동 피크" → "검증 12 × 1.10~1.17 = −13.7 GB(+α ≤ 0.3) / 클라이언트 학습 구간 플래토 78~80 GB". §7-4 L247 의 "④ 만" 문구 폐기.
4. **증빙 기록 방식.** `RoundResult` + ④ 메트릭 + 원자 로그 지표 행 + ②③ meta.json(accounting.csv 열 제외, 시드 1 은 None) 승인 여부.
5. **`FIXED_OVERRIDES["workers"] = 8` 명시 등록.** 값 불변(args.yaml 이미 8), 학습 워커 수가 증강 난수열 배정을 정하므로 사실상 공통 고정 항목 — 문서화 목적 등록을 허용할지("무변경 불변조건" 과의 충돌 판단).
6. **`_headroom_wait(need_commit_gb=16)` 상향.** 플래토 증분(B 적용 ≈ +41, 미적용 ≈ +55) 이상으로 올릴지 — 학습 코드가 아니라 launcher 설정, 등가 무관.
7. **속도 축 각주 문안.** (i) 시드 2·3 ④ "B 로 ≈ −1분/라운드", (ii) B 와 무관한 칸 비대칭(④ 150× 지불, 시드 1 ④ 의 10~14 %) 분해 각주.
8. **시드 1 export 와 시드 2 ② 병행 유지 여부**(export 커밋 미실측, 기준선 상승 요인).
9. **별건 3건.** (a) OOM 자동 반감(`trainer.py:527-535`)이 회계에 흔적 없음(`budget_audit.py:353` 은 steps 합만) → `effective_batch` 가드; (b) Ultralytics `settings.json sync: true` → `events.py:87-93` 조건 충족 시 GA4(`:69`)로 익명 사용 이벤트(task·data stem·epochs·batch·n) 전송 — 레드라인 3 인접, `sync=False` 판단(설정 파일은 저장소 밖, 등가 무관); (c) `detection/dataset_view.py:87-88` "val 은 훈련 중 접근하지 않는다" 는 부정확(매 기동 val.cache 스캔 + 48 배치 디코딩) — 트랙 A 정정.

### 9. 미결·미확인

- 액터 내 검증 프리페치·핀 몫의 정확한 크기: 트레이서 가시 증분 priv ≤ 0.3 / ws ≤ 0.6 GB 는 상한이지 실측 분해가 아니다 — T7 트레이서로 확정. 프리페치 48 배치가 aspect-ratio 정렬(`base.py:377`)의 앞쪽 "가장 납작한" 배치라 작다는 것은 추정.
- 일반 프로세스 ↔ Ray 액터 bit 동일: §7-5 는 액터↔액터만. Ray 가 액터에 OMP_NUM_THREADS 등 스레드 환경을 주입하는지 소스 미인용(미확인). 그래서 T7 을 ④ 경로로 둔다.
- (g) cuDNN 워크스페이스 폴백(benchmark=False 경로) — torch C++ 미독, 추측.
- 재사용 프로세스의 (k+1)번째 라운드 등가 — §7-5 미확인 항목 그대로, B 와 독립.
- raytune 콜백 실제 등록 여부는 소스 근거(`ray/air/session.py`·`ray/tune` 존재)이며 실행 미확인; 어느 쪽이든 `get_session()` None → 무동작.
- `events.py` 텔레메트리의 실제 전송 여부(ONLINE·IS_PIP_PACKAGE 실효값) 미확인.
- ②③ 의 시드 2·3 플래토 실측 없음(09-03 sysmon 부재; §7-1 L184-187 "~8 GB 가볍다" 는 추정) — 시드 2 ② 기동 시 트레이서로 확인.
- 09-05 `startup_trace_s1_cell4_20260905.csv` 는 role 열에 `dl_worker` 행이 0건이라 §7-1 의 "+46~48" 에는 워커 단위 근거가 없었다 — 09-07 트레이서가 유일한 워커 단위 실측.
- 라운드 45~49 재spawn 파의 실측 커밋(시드 1 은 아직 미도달).
- 벽시계 이득 −63~69 s/라운드는 spawn 타임스탬프 기반 추정이며 B 적용 실측은 T7 에서.

## D. Ray 상주 발자국 축소 (유휴 워커 · object store · dashboard)

버전 기준: flwr 1.33.0 / ray 2.55.1 / ultralytics 8.4.120 / torch 2.11.0+cu128 (`.venv/Lib/site-packages/*.dist-info` 실측). 모든 줄 번호는 이 설치본과 wt/C 트리(학습 코드는 `bcc2a98` 이후 무변경; 문서 HEAD 는 작성 시 `694ee16`) 기준. 실측은 09-07 재기동 #3 의 1 s 트레이서(`logs/trace_chain_s1_20260906_235623*.csv`)·sysmon·Ray 세션 로그(`%TEMP%\ray\session_2026-09-07_00-09-07_255240_33804\logs\`) 기준.

### 1. 목적·필요성 (A1+A2 이후 기준으로 정직하게)

- A1(페이지파일 고정) 뒤 커밋 한도 103.62 GB, ④ 기동 피크 80.43 GB(트레이서 `.csv` 00:27:00 행: commit 80.43 / sum_priv 50.61 / dl_worker 20 / ray_worker 24) → 여유 23.2 GB. 평시 기준선이 09-05 수준(42 GB)으로 되돌아가는 최악 가정에서는 여유 ≈ 6.8 GB.
- Ray 가 상주하며 차지하는 커밋은 실측 분해로 **≈ 5.5 GB** 다 — plasma object store 비사설 커밋 ≈ 4.33 GB(§2-③) + 유휴 프리스타트 워커 10 × 0.078 = 0.78 GB + 보조 프로세스(dashboard·agent·monitor·log_monitor·runtime_env agent) 0.31 GB + gcs/raylet 0.12 GB (`_procs.csv` 00:27:10 행). 기동 피크 증분 +54.8 GB 의 10 %.
- 그중 flwr 1.33 의 정식 표면(`--federation-config`)으로 닿는 것은 유휴 워커 **0.6~0.8 GB** 뿐이다. object store(4.3 GB)는 proto 에 필드가 없어 환경변수로만 닿고, 그 환경변수는 저장소 밖(SuperLink 데몬 수명)에 산다. dashboard 는 이미 최소 모드라 이득 0.
- 즉 D 전량을 적용해도 최악 여유 6.8 → 12.3 GB, 정식 표면만이면 6.8 → 7.5 GB. 어느 쪽도 "경계선 위에서 주사위를 던지는" 상태(10번 §7-1)를 바꾸지 못한다. 안전 여유를 실제로 만드는 항목은 B(미소비 검증 로더 워커 12개 제거, −13.2~14 GB, §4 정정치)와 A2(−14 GB)다.

**결론: 비채택.** 정식 표면 한 토큰(`init-args-num-cpus=2`, −0.6~0.8 GB)은 단독 게이트를 열 가치가 없으므로 B 미니스펙의 부속 후보로만 넘긴다(§8 Q2). 본 스펙에서 실제로 하는 일은 10번 문서의 수치 정정(§7)이다.

### 2. 현재 동작 (파일:줄 근거)

**① 설정이 Ray 까지 가는 길.** `scripts/main_det.py:323-325` `FEDERATION_CONFIG = "num-supernodes=3 client-resources-num-cpus=2 client-resources-num-gpus=1.0"` → `:341-342` `flwr run . local-sim --run-config … --federation-config FEDERATION_CONFIG --stream` → `flwr/cli/run/run.py:179` `init_channel_from_connection` (→ `flwr/cli/utils.py:341` `ensure_local_superlink`: `local_superlink.py:70-73` 데몬이 이미 떠 있으면 재기동하지 않음) → `run.py:184` `build_fab_from_disk` (④ 진입 시 worktree 디스크에서 FAB 재빌드, `flwr/cli/build.py:182-184`) → `run.py:261-263` `parse_config_args`(`flwr/common/config.py:206` 정규식 → `:218-220` tomli 타입화) → `-`→`_` → `flwr/supercore/utils.py:282-296` `simulation_config_from_json`(proto 필드 밖 키는 `:287-289` `ValueError("Unknown simulation config field(s)")`) → `run.py:194-202` `StartRunRequest`. SuperLink: `control_handlers.py:527-533` `DEFAULT_SIMULATION_CONFIG`(`supercore/constant.py:121-131`, `init_args_num_cpus=None`) 복사 후 `MergeFrom(override)` → `:561-567` `create_run` → `flwr/server/superlink/linkstate/sql_linkstate.py:987-990` JSON 저장. flwr-simulation 자식: `flwr/simulation/app.py:227` PullTaskInput → `:247` → `:82-112` `_run_simulation_settings` — init_args 로 옮기는 키는 `:102-109` 의 4개(num_cpus/num_gpus/logging_level/log_to_driver)뿐 → `run_simulation.py:426-435` logging 기본값 → `:445` json → `vce_api.py:349-353` (`import ray` 는 여기서 **지연 import**) → `raybackend.py:102-122` `init_ray`: `:107-109` init_args 전 키를 필터 없이 복사, `:110` `setdefault("include_dashboard", False)`, `:119-122` `ray.init(runtime_env={PYTHONPATH}, **ray_init_args)`. 액터 풀: `raybackend.py:133-148` → `ray_actor.py:113` `int(CPU 12 / 2) = 6`, `:116-121` GPU 조건 `min(6, int(1/1.0)) = 1` → **액터 1개**, `:424-426` `options(num_cpus=2, num_gpus=1.0)`. 실측: raylet.out:267 `ClientAppActor.__init__ pid=36524 {CPU: 2, GPU: 1}`; 09-04 로그 16.7 MB 전체에서 `ClientAppActor pid=39928` 단일; 로그 8행 `Federation @none/default (3 simulated SuperNodes)`. → 현재 값은 기본값(gpus=0.0, 액터 6개)이 아니라 **우리 값으로 돌고 있다.**

**② proto 표면.** `flwr/proto/federation_config_pb2.pyi:40-48` 필드 정확히 9개(num_supernodes, client_resources_num_cpus, client_resources_num_gpus, backend, verbose, init_args_num_cpus: `int`(:45), init_args_num_gpus: int, init_args_logging_level, init_args_log_to_driver). `object_store_memory`·`include_dashboard` 없음. `flwr/cli/federation/simulation_config.py:95-125` CLI 도 같은 4개 init-args 만 받는다. 단 9필드 제한은 CLI/proto 층뿐이고 `RayBackend` 는 필터가 없다(`raybackend.py:107-109`; 인프로세스 `fl/pilot_sim.py:161-171` SMOKE_BACKEND 는 `include_dashboard`·`configure_logging` 을 이미 넘긴다).

**③ Ray 가 띄우는 것(09-07 세션 실측).**
- 노드 자원: `resource_and_label_spec.py:247-248` num_cpus 미지정 시 `get_num_cpus()`=12 → raylet.out:42 `InitialConfigResources: {CPU: 12, GPU: 1, object_store_memory: 4.64e+09}`.
- 프리스타트 워커: `services.py:1979-1981` `--num_prestart_python_workers={int(num_cpus)}`=12 → raylet.out:197-208 00:09:14 에 12개 기동(`Started worker process` 총 13행 = 12 + 액터 전용 1). 액터는 runtime_env 워커로 **따로** 뜬다(raylet.out:214 pid 36384 → `_events.csv:56,59` setup_worker → 액터 36524). 00:09:17 에 2개가 `Disconnecting worker, graceful=true`(raylet.out:215-216)로 소멸 → 잔존 유휴 10(debug_state.txt:98 `num PYTHON workers: 11`, :105 `num idle workers: 10`), run 내내 유지. 사설 0.078 GB/개(`_procs.csv` 00:21:20·00:27:10) + uv 트램펄린 stub 12 × 0.001.
- object store: `worker.py:1857-1860` → `utils.py:557-560` 가용 물리 메모리 × 0.3(`ray_constants.py:80-83`), 상한 `:76-78` `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES`(기본 200 GB, **import 시 평가**), 하한 `:86` 75 MiB(`services.py:2251-2256` 미달 시 ValueError). 세션별 선언치 09-04 5.19426 / 09-05 3.16666 / 09-07 4.64018 GB(각 raylet.out:4) — run 마다 흔들린다. 실사용 최대 0.0758882(09-04)·0.0758732(09-07) GB = ArrayRecord 36.15 MB(09-04 로그 21행) × 2; pinned 37,944,100 B(debug_state.txt:36-37); spill 0(:39-42). **Windows 커밋 계상 실측:** 트레이서 00:09:11→00:09:12 commit 30.67→35.19(+4.52) vs 추적 사설 합 4.44→4.63(+0.19), 같은 초에 raylet.exe 스폰 → 비사설 커밋 **+4.33 GB** ≈ 선언 4.64 의 93 %. sysmon 00:08:30→00:09:33 도 commit +8.71 vs py_priv +4.06 으로 독립 확인. 어느 프로세스의 사설 바이트에도 안 잡히고 시스템 커밋만 오르는 양상 = 페이지파일 배킹 섹션(C++ 컴파일본이라 소스 미확인, §9).
- dashboard: `raybackend.py:110` 로 꺼져 있으나 `node.py:1348-1357` 이 head 에서 `start_api_server` 를 무조건 호출 → 최소 모드(dashboard.log:20 `Available modules: [UsageStatsHead]`, :25 `http server disabled`, :26 `Usage reporting is disabled`; `worker.py:1855` 사용통계 off). 프로세스 0.073 GB. 끄는 환경변수는 ray 2.55.1 에 없다(`ray_constants.py` 의 DASHBOARD 계열 3종은 다른 용도).

**④ 우리 코드는 Ray 설정을 읽지 않는다.** `fl/`·`detection/` 에 `ray` 참조 0(grep, `pilot_sim.py` 제외). 집계 `fl/strategy.py:164-166` → `fl/aggregate.py:92-113` float64 numpy, ServerApp 스레드(`run_simulation.py:305`). 통신량 `detection/round_runner.py:338`(`payload_nbytes`) → `fl/client_det.py:85`(`fl/client_app.py:97` 은 스모크 클라이언트) → `detection/serialize.py:161-163` `sum(a.nbytes)`; 서버 `fl/round_wiring.py:81,99-100`. 벽시계 `round_wiring.py:77` `timer.lap()`. 시드 `detection/round_runner.py:135`. DataLoader 워커 수 `ultralytics/data/build.py:363` `min(os.cpu_count() // max(nd,1), workers, …)` — OS 값이지 Ray 자원이 아님; 로더 시드 `:364-365` 상수. 액터 OMP 스레드 `ray/_private/utils.py:250,265-266` = 배정 CPU(client-resources 2) 기준. `FIXED_OVERRIDES`(`round_runner.py:39-61`)에 `workers` 키 없음 → 기본 8(09-04 로그 45행 `workers=8 val=False patience=10000`).

### 3. 설계 — 채택안 (대안과 기각 이유 한 줄씩)

**채택안 = 코드 무변경. 본 스펙의 산출물은 10번 문서 정정과 시드 2 ④ 첫 라운드 재측정뿐이다.** 경로별 판정:

| 경로 | 내용 | 판정 · 이유 |
|---|---|---|
| A 정식 표면 | `FEDERATION_CONFIG` 에 `init-args-num-cpus=2` 1토큰 → raylet 프리스타트 12→2, 노드 CPU 12→2, 액터 풀 1 불변 | **보류 → B 부속 후보.** 이득 0.6~0.8 GB 로 게이트·시험·정본 분기(시드 1 3토큰 vs 2·3 4토큰) 비용을 못 넘는다 |
| B env | `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=1e9` → −3.6 GB(4.64→1.0) | **기각.** 값이 SuperLink 데몬 env 에 산다(`flower_superlink.py:293` Popen env 없음 → `subprocess_executor.py:63` 상속). 데몬이 떠 있으면(`local_superlink.py:70-73`, 09-07 은 `post_reboot_C.ps1:128-141` 선기동) 조용히 무효. 정본이 저장소 밖·git 미추적 ps1 두 곳 |
| B′ env | `RAY_OVERRIDE_RESOURCES='{"object_store_memory":…}'` (`ray_constants.py:220`, `resource_and_label_spec.py:167-176,181-200,219-226` env 가 init 인자보다 우선) | **기각.** B 와 같은 결함 + 이 env 가 있으면 경로 A 의 num_cpus 도 조용히 덮인다 |
| C dashboard | 추가 축소 | **이득 0.** 이미 최소 모드(§2-③) |
| `flwr federation simulation-config` 영구 설정 | `--init-args-num-cpus 2` | **기각.** local-sim 은 `NoOpFederationManager` 메모리에만 `MergeFrom`(`noop_federation_manager.py:141-152`, linkstate 쓰기 없음) → 데몬 재기동 시 소실·저장소 밖 |
| 외부 `ray start --head` + RAY_ADDRESS | 전 노브 확보 | **기각.** 클러스터 접속 시 num_cpus/object_store_memory 인자와 양립 불가, 데몬 상주, Windows 클러스터 미지원 경고 |
| `client-resources-num-cpus` 축소 | — | **기각.** OMP_NUM_THREADS(`utils.py:250`)가 바뀌어 전처리 공통 고정에 닿음, 절감 0 |
| `fl/server_app.py` 에서 `os.environ` 주입 | — | **기각.** `import ray` 가 `vce_api.py:349-353` 지연 import 라 ServerApp 스레드(`run_simulation.py:305`)와 경쟁 조건 |
| `init-args-log-to-driver=false` | — | **기각.** 메모리 이득 0, ClientAppActor 의 Ultralytics 행(§7-5 GPU_mem 근거)이 `flwr_sep_fed.log` 에서 사라짐 |

**변경 지점 표 (본 스펙 = 문서만; 경로 A 는 총괄이 B 에 부속시킬 때만 아래 3행 활성):**

| 파일 | 심볼 | 변경 | 근거 |
|---|---|---|---|
| `docs/dev_log/2026-09-03-본실험/10_검출3칸_C.md` | §7-1 "DataLoader 워커 24개(학습 8 + 검증 16)" · "object store … 계상 여부 미확인" | 검증 워커 **12**(os.cpu_count 상한), 기동당 spawn 20; plasma **계상됨**(raylet 스폰 초 비사설 +4.33 GB, 실사용 76 MB) | `build.py:363`; `_events.csv` 기동 10회 dl_worker spawn 20(7+13, 10+10, 8+12 …); 트레이서 00:09:11→12 |
| 같은 문서 | §7-5 "20개 × 1.78~1.86 GB" · "B −29~30 GB" | 12개 × 1.10~1.17 + 8개 × 1.77~1.86(합 28.1); B 절감 **−13.2~14.0 GB** | `_procs.csv` 00:27:10 dl_worker 20행 |
| 같은 문서 | §7-4 D 행 "효과 미확정" | Ray 상주 ≈ 5.5 GB(plasma 4.33 / 유휴 0.78 / aux 0.31 / core 0.12), 정식 표면 −0.6~0.8, 나머지 표면 밖 | §2-③ |
| `docs/의사결정로그.md` | (신규 1행) | "D 비채택 — 사유·수치, 시드 1~3 ④ 는 Ray 기본 구성(CPU 12·prestart 12·store 0.3×가용) 동일" | 현재 FEDERATION_CONFIG·federation-config 항목 0건(grep) |
| *(경로 A 시)* `scripts/main_det.py` | `FEDERATION_CONFIG` | `" init-args-num-cpus=2"` 토큰(정수 표기) + 주석: 풀 1 불변·object-store/dashboard 는 표면 밖 | `:323-325`; `pb2.pyi:45` uint32 |
| *(경로 A 시)* `tests/test_fl_round_wiring.py` | 파서 시험 `:206-223` | `HasField("init_args_num_cpus") and == 2`, `isinstance(…, int)`, `init_args_num_cpus >= client_resources_num_cpus`(풀 0 방지), 키 집합 고정, 이빨 시험(`object_store_memory`/`include_dashboard` → ValueError) | `supercore/utils.py:287-289`; `ray_actor.py:126-139` |
| *(경로 A 시)* `scripts/main_det.py` | `cmd_preflight` / `stage_cell4` | `RAY_OVERRIDE_RESOURCES` 부재 단언(있으면 SystemExit) — ray.init monkeypatch 시험으로는 못 잡는다 | `resource_and_label_spec.py:219-226` |

**의사코드 (경로 A 가 채택될 경우의 유일한 변화 지점):**

```
main_det.FEDERATION_CONFIG += " init-args-num-cpus=2"        # :323-325, 다른 init-args 금지
  → run.py:261-263 → simulation_config_from_json → MergeFrom       # init_args_num_cpus=2
  → app.py:102-103 backend_config["init_args"]["num_cpus"] = 2
  → raybackend.py:107-110 ray_init_args = {num_cpus:2, logging_level, log_to_driver, include_dashboard:False}
  → ray.init(num_cpus=2)  → services.py:1979-1981 --num_prestart_python_workers=2
                          → services.py:1726-1729 startup concurrency 2
                          → object store · dashboard · client_resources · 액터 풀(=1) 무변경
preflight/stage_cell4: assert "RAY_OVERRIDE_RESOURCES" not in os.environ
사후: 세션 raylet.out 'Started worker process' 행 수 ≤ 3, InitialConfigResources CPU == 2
```

### 4. 등가 논증과 적대 검증 결과

**살아남은 주장 (근거 줄).**
- 현재 FEDERATION_CONFIG 는 backend client_resources 까지 도달한다 — §2-① 경로 + raylet.out:267·09-04 로그 단일 pid.
- 정식 표면은 9필드, object store·dashboard 는 표면 밖 — `pb2.pyi:40-48`, `supercore/utils.py:287-289`, `simulation_config.py:95-125`.
- `init-args-num-cpus=2` 는 prestart 수·기동 동시성·노드 CPU 선언만 바꾼다; 액터 풀 1 불변(`ray_actor.py:113,116-121`: 현재 min(6,1)=1, 변경 후 min(1,1)=1); OMP 는 배정 CPU 기준(`utils.py:250`); DataLoader 워커는 `os.cpu_count()` 기준(`build.py:363`); 집계·통신량·벽시계는 Ray 무관(§2-④).
- plasma 가 Windows 커밋에 계상된다 — 트레이서·sysmon 이중 실측(§2-③). 단 매핑 종류는 추론(§9).
- object store 하한 75 MiB 대비 실사용 76 MB 라 하한 근처 설정은 즉시 spill — `ray_constants.py:86`, raylet.out 사용량.
- 경로 B 의 "조용한 무효" — env 승계 사슬 `local_superlink.py:70-73,134-143` → `flower_superlink.py:293` → `subprocess_executor.py:63`.
- object store 크기 차이는 결과에 닿지 않는다 — 09-04(5.19 GB) vs 09-07(4.64 GB) 라운드 1 bit 동일(10번 §7-5 sha256 `5c0378a1…`, param_l2 3개·global_l2 일치). k=0 한정.

**기각된 주장 (설계에 반영함).**
- 설계안 2 "`flwr federation simulation-config` 값은 state.db 에 저장" → **기각.** `noop_federation_manager.py:54-59` 초기화·`:141-152` 메모리 `MergeFrom` 뿐, sqlite 는 run 별 resolved config(`sql_linkstate.py:987-990`)만. 데몬 재기동 시 소실 — 기각 사유는 오히려 강화.
- 두 설계안 공통 "검증 워커 16개, B 절감 −29~30 GB" → **기각.** `detect/train.py:100` 은 `workers*2`=16 을 **요청**하지만 `build.py:363` 이 `os.cpu_count()`=12 로 자른다. `_events.csv` 기동 10회 모두 dl_worker spawn 20 = 학습 8 + 검증 12; `_procs.csv` 00:27:10 사설 분포 12개 1.101~1.165 GB(미소비 검증 워커, import 발자국) + 8개 1.771~1.857 GB(학습 워커). B 절감 = 12 × 1.10~1.17 = **−13.2~14.0 GB**. D(env 경로 −3.6~4.3)의 상대 비중은 커지지만 경로 B 의 결함 판정은 변하지 않는다.
- 설계안 1 시험 7 "GPU 유휴 창 ~13 분" → **기각.** 10번 §5-1 라운드 1 wall 1005 s + 기동 오버헤드 ~4.5 분 ≈ **22 분**.
- 설계안 1 인용 줄 번호 → 정정: `raybackend.py` 102-122/110/119-122/133-148(내가 재확인), `ensure_local_superlink` 호출은 `cli/utils.py:341` 경유, `build_fab_from_disk` 는 `run.py:184`, `sql_linkstate.py` 실경로는 `flwr/server/superlink/linkstate/`. 설계안 2 제목 "약 5.1 GB" → 본문·실측 5.5 로 통일.
- 설계안 1 시험 6 "default_worker.py 프로세스 수 ≤ 3" → **기각.** 트레이서 role ray_worker 는 stub+real 쌍(24 = 12×2)이라 오탐. raylet.out `Started worker process` 행 수 또는 uv cpython 실 워커 수로 단언.

**조건부 주장.**
- 유휴 워커 12→2 로 실제 잔존이 0~2 가 되는지 — 유휴 소프트 한도는 C++ 정책(python 층엔 `services.py:1980` 뿐). 실측(raylet.out)으로만 닫힌다. 이득 표기 "0.6~0.8 GB".
- `2.0`(float) 표기 시 uint32 setattr TypeError — tomli 타입화(`config.py:218-220`)까지 소스 확인, protobuf upb 동작은 미실측. 시험의 `isinstance(int)` 단언으로 대체.
- 시험 확장이 '설계 변경' 인가 — R·E·N·동일 출발·FIXED_OVERRIDES·로더 시드·OMP 어느 것도 init_args 를 읽지 않으므로 계산 등가는 성립하나, `pyproject.toml:92-102`·시험 `:206-209` 가 FEDERATION_CONFIG 를 정본으로 선언해 시드 1(3토큰, `bcc2a98`)과 2·3 사이 **정본 분기**가 생긴다 → 게이트 승인 + 의사결정로그 1행 없이는 불가. 비채택이면 이 분기 자체가 없다(§5).

**적대 검증이 새로 찾은 사실 (설계에 반영).** (a) `~/.flwr/config.toml`(336 B, 09-03 04:35)에 flwr 가 이관한 `[superlink.local-sim] options.num-supernodes=3 / client-resources.num-cpus=2 / num-gpus=1.0` 이 살아 있다 — `--federation-config` 가 있으면 무시(`run.py:245-257`, 09-04 로그 2-3행 경고)되지만 빠지면 `_parse_deprecated_options`(`run.py:268-292`)가 이 값을 쓴다. '정본이 저장소 안' 은 시험 `:220-223` 의 명시 단언에 의존한다. (b) `RAY_OVERRIDE_RESOURCES` 가 세션 env 에 있으면 init 인자를 덮는다(§3 B′). (c) `tests/test_fl_round_wiring.py:590-622` 는 `_run_simulation` 을 직접 불러 FEDERATION_CONFIG 경로를 타지 않는다 → 그 경로를 실제로 태우는 검증은 `main_det.py:431` preflight 스모크뿐. (d) 저장소 안 커밋 여유 게이트는 `main_det.py:148` `need_commit_gb=16.0`(`:450` 체인 진입 시 1회) — 피크 +54.8 의 1/3 이라 실질 보호선은 "기준선 ≤ ~48 GB". `post_reboot_C.ps1:17` 의 55 GB 는 저장소 밖·시드 1 한정. (e) 시드 2 체인은 `chain_follow.ps1:21-24` 가 CHAIN_DONE 직후 새 프로세스로 `main_det.py` 를 import 하고, ④ 의 FAB 는 ④ 진입 시(`run.py:184`) 디스크에서 재빌드된다 → 그 사이 wt/C 편집은 ②③(옛 import)과 ④(새 FAB)를 갈라놓는다.

### 5. 적용 범위와 비교 가능성

- **범위: ④ 만.** Ray 는 `flwr run` 시뮬레이션에만 있고 ②③ 은 `main_det` 프로세스 안에서 `train_round` 를 직접 돈다. D 는 시드당 ~13 h 구간에만 닿고, ②③ 의 같은 경계선(트레이너 + 워커 20)에는 아무 도움이 안 된다 — B 는 세 칸 전부에 닿는다.
- **비채택이면 시드 1·2·3 의 ④ 가 같은 Ray 구성(CPU 12 · prestart 12 · store 0.3×가용)이다.** 정본 분기·의사결정로그 예외 항목·시드 1 라운드 1 재실행(22 분 GPU) 전부 불필요. store 선언치가 run 마다 다른 것(5.19/3.17/4.64)은 이미 시드 1 안에서 결과 무관이 실측됐다(§4).
- 경로 A 를 B 에 부속시키면: 시드 1 ④(실행 중, 무변경) vs 시드 2·3 ④ 사이에 호스트 구성 차이 1건이 생긴다. 계산 등가는 §2-④·§4 논증, 실측 대조는 §7 T5. 의사결정로그 1행 + 실험 조건 표 '호스트 구성' 열 필수.
- **속도 축:** 유휴 워커는 CPU 0, spill 없음(debug_state.txt:39-42) → 라운드 벽시계 영향 0 기대. 시드 1 이 §5 비경합 단독 실측 지정 run 이라 시드 2·3 벽시계는 헤드라인이 아니다. §5-1 대역(923~1005 s)을 벗어나면 sysmon `game_proc`·raylet.out spill 로그로 원인을 갈라 **제외가 아니라 표시**한다.
- **적용 시점 규칙(경로 A 든 B 든 코드 편집이 생기면):** 시드 2 체인 기동(④ 완주 ≈ 09-07 13:20 직후) 전에 커밋되어 디스크에 있어야 한다. 체인 기동 뒤 편집은 ④ FAB 에만 실려 'FAB 해시 ≠ 실행 구성' 이 된다(§4-e). 비채택이면 이 규칙은 B 에만 남는다.

### 6. 위험과 대응

| 위험 | 심각도 | 대응 |
|---|---|---|
| 비채택 뒤 시드 2·3 착수 시 평시 기준선이 42 GB 로 회귀 → 여유 6.8 GB, `_headroom_wait` 16 GB 게이트를 통과한 채 죽음 | high | D 와 무관한 항목으로 분리 보고: (i) B 게이트, (ii) `need_commit_gb` 상향(피크 +54.8 기준 ≥ 55) 여부 §8 Q4, (iii) 착수 전 `post_reboot_C.ps1 -CheckOnly` 실측 |
| `FEDERATION_CONFIG` 가 누락되면 `~/.flwr/config.toml` 의 이관값이 조용히 대체 | medium | 기존 시험 `:220-223` 이 명시 문자열을 고정 — 유지. 이관 파일은 건드리지 않는다(삭제도 변경) |
| 세션 env 에 `RAY_OVERRIDE_RESOURCES` 가 있으면 Ray 자원이 우리 의도와 다르게 잡힘 | low(현재 없음 — raylet.out:42 CPU 12 = 기본값) | 경로 A 채택 시 preflight 부재 단언 |
| 경로 B 를 되살릴 경우: 데몬 위상에 따라 적용/무효가 갈리고 사후 검증 없이는 "축소됐다" 가 거짓이 될 수 있음 | high | 되살리지 않는다. 되살리려면 데몬 기동 절차 저장소 내 이관 + raylet.out `Allowing the Plasma store to use up to X` 파싱 후 X > cap 이면 SystemExit 가 선행 조건 |
| object store cap 을 하한(75 MiB) 근처로 두면 즉시 spill → 속도 축 오염 | medium | 경로 B 비채택으로 해당 없음. 참고값: cap ≥ 1 GB |
| "D 로 충분히 안전해졌다" 는 착시가 B·A2 논의를 늦춤 | medium | 게이트 결정문에 "D ≤ 5.5 GB, B −13.2~14 GB, A2 −14 GB" 를 나란히 명기 |
| warm 액터가 클라이언트 사이에도 9.57~10.2 GB 를 들고 있다(`_procs.csv` 00:21:20 pid 36524) — Ray 상주(5.5)의 두 배 | medium | D 범위 밖. 별도 항목 개설 여부 §8 Q5 |

### 7. 검증 계획

비채택 스펙이므로 실행 규모는 문서 정정 + 무GPU 확인 + 시드 2 ④ 첫 라운드의 수동 관측이다. 경로 A 항목은 총괄이 B 에 부속시킬 때만 활성.

| # | 시험 | 검증 내용 | 규모 |
|---|---|---|---|
| T0 | 10번 문서 정정 대조 | §7-1·§7-4·§7-5 의 수치가 §3 표대로 바뀌었는지(검증 워커 12, B −13.2~14, plasma 계상, Ray 상주 5.5) — 원 자료 파일:행 병기 | 문서, 0 분 |
| T1 | 기존 파서 시험 유지 (`tests/test_fl_round_wiring.py:206-223`) | 비채택이면 무변경. 시드 2·3 착수 전 pytest 1회로 `--federation-config` 명시 단언이 살아 있음을 확인 | CPU, 초 단위 |
| T2 | 시드 2 ④ 기동 관측 (실행 중 run 무간섭, 읽기만) | 새 세션 raylet.out:4 선언치·:42 CPU 12·`Started worker process` 13행 — 시드 1 과 같은 Ray 구성임을 기록(비교 가능성 근거). 1 s 트레이서를 체인 전 구간으로 다시 돌려 raylet 스폰 초 Δcommit−Δpriv ≈ 선언치 재확인 | 관측만 |
| T3 *(경로 A 시)* | 파서 고정 확장 + 이빨 | `init_args_num_cpus == 2`·`isinstance(int)`·`>= client_resources_num_cpus`·키 집합 고정; `simulation_config_from_json({"object_store_memory":1})`/`({"include_dashboard":False})` → ValueError('Unknown simulation config field') | CPU |
| T4 *(경로 A 시)* | 풀 크기 불변식 | `ray.nodes` monkeypatch: (CPU 12,GPU 1)→1, (CPU 2,GPU 1)→1, (CPU 1)→ValueError; `_run_simulation_settings(SimulationConfig(3,2,1.0,init_args_num_cpus=2))` 의 backend_config 골든; `ray.init` monkeypatch 로 kwargs num_cpus==2·include_dashboard False | CPU |
| T5 *(경로 A 시, GPU 유휴 창 22 분)* | k=0 실측 대조 | 시드 1 ④ `num-server-rounds=1` 을 새 구성·별도 project 경로로 재실행 → `global_r001.npz` sha256 == `5c0378a1288619930c13e54746664c48993ec5345d977b7b28bbf546e38a0ce2`, param_l2 3개·global_l2 §7-5 값과 bit 일치. 산출물 DO_NOT_CITE, 실행 중 원장 무접촉 | GPU 22 분, 시드 1 완주 후 |
| T6 *(경로 A 시)* | 사후 검증 | 세션 raylet.out `Started worker process` ≤ 3, InitialConfigResources CPU == 2, `_procs.csv` 유휴 실 워커 ≤ 2; preflight 스모크(`main_det.py:431`) 가 같은 FEDERATION_CONFIG 로 완주 | preflight 1회 |

### 8. 게이트 질문 (총괄이 결정할 것만)

1. D 단독 **비채택** 을 승인하는가 — 근거: 정식 표면 이득 0.6~0.8 GB, 전량 ≤ 5.5 GB, object store 는 저장소 밖 env 로만 닿음.
2. 경로 A(`init-args-num-cpus=2`, −0.6~0.8 GB)를 B 미니스펙의 부속 항목으로 실을지, 아니면 아예 버릴지. 실으면 시드 1 ↔ 2·3 정본 분기 1건 + 의사결정로그 1행 + T3~T6 비용이 붙는다.
3. 10번 §7-1·§7-4·§7-5 의 수치 정정(검증 워커 16→12, B −29~30→−13.2~14, plasma 계상 확정)을 승인하는가 — B 스펙의 절감 추정치가 이 정정치에 의존한다.
4. `scripts/main_det.py:148` `need_commit_gb=16.0` 을 실측 피크(+54.8 GB) 기준으로 올릴지 — D 밖이지만 시드 2·3 자동 체인의 실질 보호선이라 여기서 묻는다(코드 변경 = 별도 게이트).
5. warm 액터의 클라이언트 간 잔존 9.6~10.2 GB 를 별도 항목으로 개설할지(D 범위 밖, Ray 상주의 2배).
6. SuperLink 선기동 절차(`logs/recover/post_reboot_C.ps1`, git 미추적)를 저장소 안으로 이관할지 — 경로 B 류를 장래에라도 살리려면 선행 조건이고, 살리지 않으면 불필요.

### 9. 미결·미확인

- plasma 매핑의 정체(페이지파일 배킹 섹션 vs 파일 매핑): `dlmalloc.cc`/`store_runner.cc` 는 컴파일본이라 **소스 미확인**. 판정은 수치 일치(비사설 +4.33 vs 선언 4.64)와 Temp\ray 에 대형 파일 부재라는 방증. 반증은 cap 을 바꾼 세션의 같은 측정으로만 가능.
- raylet 유휴 워커 소프트 한도 정책(12 프리스타트 + 액터 1 → 유휴 10 유지, 2개 kill)은 C++ — **미확인**. `num_cpus=2` 에서 잔존 0~2 인지는 실측 필요(비채택이면 불필요).
- `2.0` float 표기 시 uint32 setattr 의 TypeError — protobuf upb 구현 **미실측**.
- 현 SuperLink 데몬(09-06 23:56 선기동)에 `flwr federation simulation-config` 를 쓴 적이 있는지 — **미확인**(실행 중 조회 금지). 메모리 한정이고 raylet.out:42 CPU 12 로 기본 구성이 소거 확인되므로 결론에 영향 없음.
- flwr 1.34+ 에서 SimulationConfig 에 object_store_memory 류 필드가 추가됐는지 — **미확인**. 추가됐어도 본실험 중 flwr 판올림은 별도 게이트.
- 시드 2·3 착수 시점의 평시 커밋 기준선(25.6 vs 42 GB) — 예측 불가. D 의 필요성을 가르는 유일한 변수이며 착수 직전 `-CheckOnly` 실측으로 판단.
- "재사용 프로세스의 (k+1)번째 라운드 ↔ 새 프로세스의 첫 라운드" 등가는 §7-5 의 미결 그대로 — D 와 무관.
- 검증 로더가 생성 즉시 prefetch(`build.py:376` prefetch_factor=4)로 워커에 배치를 밀어 넣어 warm 액터 잔존(9.6~10.2 GB)에 기여하는지 — **추측**, B 스펙에서 다룰 것.

## R. ④ 분리·연합 서버측 라운드 경계 재개 — 닫힌 접두 k 판정 → 잔해 격리 → `global_r{k}.npz` 재주입 + 라운드 오프셋 (명시 재개 전용)

프레임워크 실측(.venv dist-info): flwr 1.33.0 · ultralytics 8.4.120 · torch 2.11.0+cu128 · ray 2.55.1 · numpy 2.2.6. 본 절은 읽기 전용으로 작성했고 python 은 실행하지 않았다. 진행 중인 시드 1 ④ run 은 FAB `31149ece`(09-07 00:08 설치, `~/.flwr/apps`) 위에서 돌고 있어 작업트리 편집이 닿지 않으며(FAB 는 `.gitignore` 를 적용해 `outputs/` 를 싣지 않는다 — flwr/cli/build.py:254-255, .gitignore:11), 이 문서가 게이트를 지나기 전 구현은 없다.

### 1. 목적·필요성 (A1+A2 이후 기준으로 정직하게)

- 사인은 호스트 커밋 고갈이었고(10번 §7-1), A1(페이지파일 고정, 한도 103.62 GB)만으로 재기동 #3 이 안정 궤도에 들었다 — 실측 피크 80.43 GB, 여유 23.2 GB(§7-5). 즉 **이 스펙은 사인을 고치는 것이 아니다.** 메모리 절감 0 GB.
- 남는 것은 "죽으면 처음부터" 의 대가다. 오늘 코드에서 ④ 재개는 존재하지 않는다(§2). 09-04 형태(라운드 30 에서 사망)면 유효 30 라운드 = 7.83 h 를 버리고 13.1 h 를 다시 돈다(§5-1). 중단 기저율은 GPU 11~18 h 당 1건으로 기록돼 있고(10번 §7-2:204) ④ 는 시드당 13.1 h 라, 시드 2·3 에서 하루를 다시 잃을 확률이 낮지 않다. 회수량 = 닫힌 라운드 k × 15.66 분(§5-1 평균).
- 부수 이득: 09-05 처럼 사람이 원장을 손으로 옮겨 오염을 막는 절차(§7-2 3행)를 코드 검증으로 대체하고, 재개 사실·지점·사유가 산출물(audit.json)에 남는다.
- 대가: 재개 run 은 프로세스 이음매를 갖는다. "이음매 라운드 = 무중단 라운드" 는 **소스로 증명되지 않았고 실측도 없다**(§4). 이 스펙은 그것을 T-eq 로 실측할 조건으로 못박는다.

**결론: 조건부 채택** — 명시 트리거·4조건 판정기·잔해 전수 격리·마감 재대조를 갖춘 아래 설계로 구현하되, 게이트 시험 T-eq(실기 등가) 통과가 "재개 run 을 무중단 run 과 동등하게 채점" 의 전제이며, 통과 전까지 재개 run 은 이음매 표시 run 이다.

### 2. 현재 동작 (파일:줄 근거)

- **서버는 항상 라운드 1 부터 돈다.** `fl/server_app.py:116` 이 `initial.npz` 를 읽고(:193-242 `_load_initial`), :153-159 `strategy.start(initial_arrays, num_rounds=50)` 만 부른다. flwr 의 `Strategy.start` 시그니처에 시작 라운드 인자가 없고(.venv flwr/serverapp/strategy/strategy.py:135-144), 루프는 `for current_round in range(1, num_rounds + 1)`(:201). `global_r*.npz`·`latest.npz` 를 읽는 코드는 없다(10번 §7-2 1행).
- **라운드 번호는 서버 한 곳에서 파생된다.** `FedAvg.configure_train` 이 전달받은 ConfigRecord 에 `config["server-round"] = server_round` 를 제자리 주입한다(fedavg.py:180). 소비처: 클라이언트 `round_idx = int(cfg[SERVER_ROUND_KEY]) - 1`(fl/client_app.py:120) → `derive_seed = base + 10007·r + 101·c`(detection/round_runner.py:135, :228), runs 이름 `r{r:03d}_c{c}`(:239), 재개 디렉터리(fl/client_det.py:66-67), `start_epoch = round_idx × E`(detection/fed_trainer.py:233), `smoke-fail-at`(client_app.py:81); 서버 `round_idx = server_round - 1`(fl/round_wiring.py:78) → 회계·원자 로그·`global_r{sr:03d}.npz`(fl/server_app.py:260).
- **라운드 종료 쓰기 순서**(round_wiring.py:76-115): 클라이언트마다 `accounting.record`(인메모리) 와 원자 8행이 **교대**로(:79-102) → 서버 3행(:103-113) → `on_save` → `np.savez` 로 `global_r{sr}.npz`·`latest.npz`(server_app.py:259-268). 라운드당 원자 행 = 3×8 + 3 = 27. `np.savez` 는 최종 경로에 직접 zip 을 쓴다 — tmp→replace 없음(numpy/lib/_npyio_impl.py:789-800). 실패 라운드는 집계 전 `RoundFailure` 라(fl/strategy.py:113-130) 원자 행 0건, `on_round_end` 는 집계 뒤에만(:169-170).
- **회계는 인메모리, 마감만 디스크.** `finalize_accounting` 이 `finally` 에서 accounting.csv/json·audit.json 을 쓴다(server_app.py:160-166, round_wiring.py:123-159, `to_csv` 는 "w" budget_audit.py:370). 서버 하드킬이면 회계 파일이 없다 — 진행 중 run 디렉터리 실물에 `accounting.csv` 없음(ls: atomic_log.csv·global_r001~004·latest.npz·initial.npz·runs).
- **원자 로그는 append, 감사는 존재 집합.** 헤더만 같으면 기존 파일에 잇고(fl/atomic_log.py:108-120, :127 `open("a")`+fsync :130-131), `audit_rounds` 는 `{(round, client_id)}` 존재만 본다(:176-182) → 같은 원장에 재기동 run 이 붙으면 두 벌이어도 통과(§7-2 3행 "있었다 → 수동 이동").
- **런처 멱등은 스테이지 수준.** `stage_cell4` 는 "audit.json + global_r050 둘 다 있으면 ok/raise, 아니면 같은 디렉터리에서 처음부터"(scripts/main_det.py:386-410). 부분 원장을 감지하지 않고, `_flwr_run` 은 로그를 `open("w")` 로 덮어쓰며(:355), `flwr run --stream` 은 run 이 실패해도 rc=0 이다(flwr/cli/log.py:111-113 `raise AllLogsRetrieved` → :76-77 `pass`; run.py:238-239) — 유일한 실패 신호는 :411 이 무조건 읽는 audit.json. `_ledger("cell4")` 는 성공 시에만 쓴다(:415; 실물 progress.jsonl 에 cell2·cell3 만 있고 09-04 ④ 항목 없음). `_headroom_wait` 의 커밋 문턱은 16 GB(:148) — 실측 기동 델타 +54.8 GB(§7-5)의 1/3.
- **클라이언트 epoch 경계 체크포인트.** `ResumeIdentity` = run_id·round_idx·client_idx·seed·total_epochs·local_epochs·model·data — 가중치 다이제스트 없음(detection/resume.py:84-91); run_id = run-stamp `"main_s{n}"`(main_det.py:302, client_app.py:139)라 재기동 간 신원이 일치한다. `injection_digest` 는 `apply_resume` **앞**에서 계산되고(fed_trainer.py:218-229 → :238-241) 서버로 전송되지 않는다(client_det.py:71-109 에 없음; RoundResult 에만 round_runner.py:339). 정상 완주 시 `clear_resume` 는 파일만 지우며 OSError 를 삼킨다(resume.py:347-363, :361-362). 실물 `_resume/sep_fed`: 91 디렉터리 / 0 파일.
- **09-04 실물.** atomic_log.csv 811줄 = 헤더 + 810 = 30×27; accounting.csv 90행(라운드 0~29 전 필드, value_source=measured); audit.json ok=false(빈 셀 60·총 epoch 60≠100·결측 80); global_r001~030 각 37,960,500 B; latest.npz == global_r030; runs/r030_c0·c1 은 args.yaml 만, r030_c2 는 results.csv 61·62 두 행(완주분이나 원장에 없음); 단일 액터 `ClientAppActor pid=39928` 이 라운드 1~31 전부(로그 :45, :88831, :89295).

### 3. 설계 — 채택안 (대안과 기각 이유 한 줄씩)

두 설계안은 모두 "조건부 생존" 이라 폐기된 안은 없다. 채택안은 **관점 2 의 판정기·격리·해시 연쇄·마감 재대조** 위에 **관점 1 의 명시 트리거·출처 표식(value_source·audit.json)** 을 얹은 병합이며, 적대 검증의 필수 수정을 전부 반영했다.

- 기각 — 관점 2 의 "멱등 재기동 = 자동 재개": 후속 launcher 가 시드 2·3 `chain` 을 무인 실행하고(logs/recover/chain_follow.ps1:14-26) `_headroom_wait` 16 GB 로는 09-05 형태 재사망을 못 막으므로 사인 미해소 재기동을 사람 확인 없이 1회 허용하게 된다 — "우회 금지·멈추고 보고"(main_det.py:363, post_reboot_C.ps1:44) 와 충돌.
- 기각 — 관점 1 의 k 판정 (a) "존재 집합": 중복·여분·다른 시드 행을 닫힘으로 세고 절단이 그것을 보존한다.
- 기각 — 관점 1 의 k=0 경로 "오늘 경로 그대로": `_resume/r000_c*/*.pt`·runs·stale audit 을 격리하지 않아 라운드 0 이 체크포인트를 이어 간 run 이 ok 로 닫힌다(감사는 resumed_cells 에 적을 뿐 budget_audit.py:357-360).
- 기각 — 관점 2 의 `accounting.csv` `segment` 열: 시드 1 실물 25열(budget_audit.py:61-97)과 갈라진다. 세그먼트는 audit.json·resume_log.jsonl 에만.
- 기각 — 관점 2 의 닫힘 조건 (iv) results.csv 행 수: 트레이너 생성 시 unlink 되는 부속물(§4). 경고로 강등.
- 기각 — `Strategy.start` 루프 복제, `latest.npz` 기준 재개, 원자 로그 재구성 회계, run-stamp 변경, 부분 라운드 완주 클라이언트 결과 재활용, 잔해 삭제: 두 설계안의 기각 사유 그대로(버전 드리프트 / 번호 없음·찢어짐 / (4)(5) 검사 공허 / RQ3 조인 파괴 / 채점 트리 오염 / 포렌식 상실).

**변경 지점 표**

| 파일 | 심볼 | 변경 | 근거 |
|---|---|---|---|
| `fl/strategy.py` | `WeldFedAvg.__init__(round_offset=0)`, `configure_train`, `aggregate_train`, `aggregate_evaluate` | `super().configure_train(server_round + offset, …)`; 집계 두 메서드는 진입부에서 `server_round += offset`. 기본 0 이면 오늘과 동일 | fedavg.py:180; strategy.py:201-233; fl/strategy.py:107-202 |
| `fl/server_app.py` | `main` | `k = int(cfg.get("resume-from-round", 0))`; k>0 이면 **try 밖에서** 사전검증(판정기 재실행·`_load_resume_point`·회계 접두 복원) → 실패 시 `audit.json {ok:false, failures:["resume-precheck: …"]}` 만 쓰고 accounting.csv 는 건드리지 않고 종료; `cell != "sep_fed"` 이면 k>0 거부(⑦ 재개 미정의); `WeldFedAvg(round_offset=k)`; `strategy.start(num_rounds=num_rounds − k)`; RESUME 배너(절대 라운드 대응표); `train_cfg["num-rounds"]`·`AccountingMatrix(num_rounds)`·`finalize(num_rounds)` 는 R 유지 | server_app.py:92-166, :140-152 |
| `fl/server_app.py` | `_load_resume_point` [신규], `_save_round`/`_write_arrays`, `on_save` | `[z[f"arr_{i}"] for i in range(len(keys))]` 인덱스 적재 → `assert_compatible` → 유한성 → l2 ↔ 원장 global_l2 상대 1e-9; npz 저장을 tmp→`os.replace` 로 원자화; `on_save` 순서 = accounting.csv 원자 덤프 → npz → `global_index.jsonl` append({server_round, file, sha256, nbytes, global_l2}). **원자화 대상 파일 목록(비판 L7 반영): `global_r{k}.npz`·`latest.npz`·`accounting.csv`(현재 `to_csv` 는 `open("w")` budget_audit.py:370)·`audit.json`(round_wiring.py:149)·`global_index.jsonl`(append, 찢어진 마지막 행은 판정기가 잔해 처리) — T-npz/T-truncate 에 각각 픽스처** | _npyio_impl.py:778, :789-800; flwr/app/message/arrayrecord.py:275, :317; serialize.py:103-143, :146-158; server_app.py:245-268 |
| `fl/round_wiring.py` | `CLIENT_ROUND_METRICS`/`SERVER_ROUND_METRICS` 상수, `finalize_accounting(resume_events=…)` | 지표명을 상수로 승격(판정기와 공유); 마감에 행 수 == 27×R, (round, client, metric) 중복 0, 라운드별 다중집합 일치, 접두 `global_r{1..k}` sha256 재대조, 세그먼트가 1..R 을 빈틈·겹침 없이 덮는지, `kept_rows_sha256` 재대조; audit.json 에 `resume_events`·`segments` | round_wiring.py:86-98, :107-111, :123-159 |
| `fl/resume_ledger.py` [신규] | `judge_closed_prefix`, `truncate_to`, `restore_prefix`, `ResumeEvent`, `load_resume_events` | 아래 "닫힌 라운드" 정의; 잔해 전수 목록화·격리 이동(삭제 아님); 절단은 tmp→fsync→replace, 전후 sha256·행 수 기록; `resume_log.jsonl` append; 회계 접두 복원(measured 우선·없으면 reconstructed + notes) | atomic_log.py:108-131, :176-182; round_wiring.py:76-115 |
| `fl/atomic_log.py` | `rows_by_round`, `sha256`, `audit_rounds` | 중복·다중집합 검사 추가. FIELDS·열 순서 불변 | atomic_log.py:37-50, :170-185 |
| `detection/budget_audit.py` | `AccountingMatrix.from_csv` [신규], `AuditReport.restored_cells`, `audit()` | CSV 전 필드 복원(momentum·arg_momentum·participated 포함 — 원형 scripts/gate_reduced_pilot.py:476-492 는 셋이 빠져 있다); `value_source="restored:accounting.csv"` 는 `reconstructed` 가 아니라 `restored` 로 분류(현재는 measured 아니면 전부 "인용하지 마라" notes :323-330); 검사 추가: `seed == derive_seed(base, r, c)`, `budget_fired_at == (r+1)·E − 1`, `optimizer_steps == epochs_ran × ceil(num_examples/32)`(FIXED batch, round_runner.py:51) | budget_audit.py:61-97, :203-207, :213-363 |
| `scripts/main_det.py` | `stage_cell4(seed, *, resume_from=None, reason=None)`, `_flwr_run_config(resume_from)`, `_flwr_run`, `_ledger`, argparse | 부분 원장·잔해 감지 시 **거부 + k·재개 명령 출력**; `cell4 --seed N --resume-from K --reason "…"` 만 재개 진입(K == 판정 k 이중 대조, 사유 필수); 재개 로그는 새 파일 `flwr_sep_fed_resume_r{K:03d}.log`(원 로그 무손상); audit.json 부재 → "서버 사전검증 실패/하드킬" 로 SystemExit(트레이스백 아님); 세그먼트 종료(성공·실패)마다 progress.jsonl 에 segment·start_round·end_round·wall_s·rc; 재개 preflight 커밋 여유 문턱 ≥ 60 GB(실측 +54.8 + 마진) | main_det.py:148, :290-311, :355-358, :386-415, :445-455, :565-581 |
| `pyproject.toml` | `[tool.flwr.app.config] resume-from-round = 0` | 선언 없는 키는 `fuse_dicts(check_keys=True)` ValueError → INVALID_RUN_CONFIG. FAB 해시가 바뀐다 | pyproject.toml:86-89; flwr/common/config.py:84-106; control_handlers.py:512-516, :587-593 |
| `tests/…` | §7 참조 | 회귀 고정 | tests/test_fl_round_wiring.py:203, :445-457 |
| **손대지 않음** | `detection/fed_trainer.py`, `round_runner.py`, `resume.py`, `serialize.py`, `init_weights.py`, `fl/client_app.py`, `fl/client_det.py`, `configs/` | 학습 경로·5칸 공통 고정·클라이언트 계약 무변경 — 등가 논증의 전제. 함정 #1 접촉점(fed_trainer.py:162-292) 그대로 | round_runner.py:37-60 |

**닫힌 라운드 j (server_round 1..50, round_idx = j−1) 의 정의** — 아래 4개 동시 성립:
(i) atomic_log.csv 에 round_idx=j−1 행이 **정확히 27개**, (client_id, metric_name) 다중집합이 기대 집합과 같고 12 필드 전부 파싱 가능(중복 0); (ii) 그 행들의 run_id·seed·cell·split_hash 가 기대값과 같고 세 클라이언트 `epochs_ran == E`; (iii) `global_r{j:03d}.npz` 가 열리고(찢어진 zip 은 BadZipFile) 배열 수 == len(keys), `assert_compatible` 통과, 전 원소 유한, `params_l2_norm` 이 서버 행 `global_l2` 와 **상대 1e-9 이내**(float64 BLAS 누적 순서 보호, serialize.py:153-158); (iv) accounting.csv 가 있으면 라운드 j−1 셀 3개 전부 존재(부분 셀은 절단 대상 — round_wiring.py:79-102 교대 쓰기 때문), 없으면 원자 로그 재구성 + notes. `global_index.jsonl` 이 있으면 sha256 대조, 없으면(시드 1 현재 run) notes.
**k\* = 1..k 가 전부 닫힌 최대 연속 접두.** 거부(LedgerInconsistent, 자동 dedup 없음): (round, client, metric) 중복; npz 는 정상인데 (i) 미완비(쓰기 순서상 불가능); 다른 run_id/seed/split_hash 행 존재. results.csv 는 판정에 쓰지 않고 경고만.

**의사코드**

```
# scripts/main_det.py
def stage_cell4(seed_no, *, resume_from=None, reason=None):
    out = sd/"fl"/"sep_fed"
    if audit_ok(out) and (out/"global_r050.npz").exists(): return            # 완료 마커
    st = judge_closed_prefix(out, R, E, clients=(0,1,2), keys, ref, expect=dict(run_id=…, seed=…, cell="sep_fed", split_hash=digest))
    debris = st.debris | resume_files(sd/"_resume"/"sep_fed") | stale({audit.json, accounting.*, runs/*, flwr_sep_fed.log})
    if resume_from is None:
        if st.k > 0 or debris: raise SystemExit(f"④ 부분 원장 k={st.k}, 잔해 {len(debris)} — 자동 재개 금지. 사인 진단 후 "
                                                f"`cell4 --seed {seed_no} --resume-from {st.k} --reason '…'`")
        return _fresh_start(seed_no)                                           # 오늘 경로 (k=0, 잔해 0 일 때만)
    if not reason or resume_from != st.k: raise SystemExit("k 불일치 또는 사유 없음 — 멈추고 보고")
    events = load_resume_events(out/"resume_log.jsonl")
    if events and st.k <= events[-1].k: raise SystemExit("무진전 재개 — 같은 라운드에서 재사망, 멈추고 보고")
    ev = truncate_to(out, st, quarantine=out/"_quarantine"/stamp)              # 절단·격리·sha256·행 수; PermissionError 는 그대로 SystemExit
    append_jsonl(out/"resume_log.jsonl", ev | {"reason": reason, "segment": len(events)+1, "git_head": …, "prev_fab": …})
    _headroom_wait(need_commit_gb=60.0)
    _flwr_run(_flwr_run_config(…, resume_from=st.k), log_path=sd/f"flwr_sep_fed_resume_r{st.k:03d}.log")
    if not (out/"audit.json").exists(): raise SystemExit("④ audit.json 부재 — 서버 사전검증 실패 또는 하드킬: " + tail(log))
    …기존 audit ok 검사…; _ledger(seed_no, "cell4", segment=…, start_round=st.k+1, end_round=R, wall_s=…, rc=0)

# fl/server_app.py  main()
k = int(cfg.get("resume-from-round", 0)); initial_arrays, keys, ref = _load_initial(cell, cfg)   # 항상 (키·ref·initial 검사)
if k:
    if cell != "sep_fed": raise RoundFailure("⑦ 재개는 정의되지 않았다")
    st = judge_closed_prefix(out_dir, …)                                       # 절단 없이 판정만; 런처 값과 이중 대조
    if st.k != k or st.debris: write_audit(out_dir, ok=False, failures=[f"resume-precheck: k={st.k}≠{k} 또는 잔해"]); return
    initial_arrays, seam = _load_resume_point(out_dir, k, keys, ref, ledger_l2=server_row(k-1, "global_l2"))
    restore_prefix(accounting, out_dir, k)                                     # value_source="restored:accounting.csv" / "reconstructed:atomic_log"
    print(f"[RESUME] k={k}: 절대 라운드 {k+1}..{R} = flwr 내부 1..{R-k}")
strategy = WeldFedAvg(…, on_round_end=on_round_end, round_offset=k)
try:    strategy.start(grid, initial_arrays, num_rounds=num_rounds - k, train_config=train_cfg)   # train_cfg["num-rounds"] = R
finally: finalize_accounting(…, num_rounds=num_rounds, resume_events=load_resume_events(out_dir))

# fl/strategy.py
def configure_train(self, r, arrays, config, grid): return super().configure_train(r + self.round_offset, arrays, config, grid)
def aggregate_train(self, r, replies):  r += self.round_offset; …기존 본문(절대 라운드로 RoundFailure 문구·on_round_end)…
```

### 4. 등가 논증과 적대 검증 결과 — 살아남은 주장 / 기각된 주장 / 조건부 주장

주장의 형태: "라운드 k 글로벌에서 재개한 run 의 라운드 j>k 는 무중단 run 의 라운드 j 와 같은 입력·난수·스케줄 위치에서 돈다."

**살아남은 주장(소스로 성립)**
- 라운드 상태는 매 호출 새로 만들어진다: `train_round` 가 호출마다 `FedDetectionTrainer` 신규(round_runner.py:281-289); 생성자에서 `init_seeds(seed+1+RANK)`(trainer.py:139, RANK=-1 utils/__init__.py:56 → 실효 시드 = derive_seed); 로더·옵티마이저·스케줄러 신규(trainer.py:278-308), GradScaler 신규(:374-378), `last_opt_step` 은 `_do_train` 지역변수(:425); 래퍼가 `start_epoch = r×E`(fed_trainer.py:233), `scheduler.last_epoch = start_epoch−1`(:248), mosaic 재적용(:252-254), EMA off(:257-258), stopper 스텁(:261).
- 시드는 (base, round, client) 만의 순수 함수(round_runner.py:135) — 실물 accounting.csv r0: 20260828/20260929/20261030, r29 c2: 20551233 = base + 10007×29 + 101×2. 메인 프로세스 RNG 는 학습 루프에서 소비되지 않는다(로그 :45 `multi_scale=0.0 cls_pw=0.0 dropout=0.0`; `set_class_weights` 즉시 return detect/train.py:175-176; 모델 초기값은 주입으로 덮임 fed_trainer.py:219-226).
- 데이터 순서·증강 난수는 로더 생성기 상수의 함수: `generator.manual_seed(6148914691236517205 + RANK)`(data/build.py:364-365), 반복자는 생성 즉시(build.py:75) `_base_seed` 를 그 생성기에서 뽑고(torch dataloader.py:706-710), 순열도 같은 생성기(dataloader.py:394 → sampler.py:160-167), 워커는 `base_seed + worker_id`(worker.py:258-265) + `seed_worker`(build.py:229-233). 라운드마다 새 로더라 프로세스 이력과 무관(tests/test_loader_seed_independence.py:61-76 고정). 전제: 워커 수 `nw = min(cpu//nd, workers, batches)`(build.py:363) 동일 — 로그 "Using 8 dataloader workers" 93건 전 라운드.
- 서버 입력 bit 동일: 무중단 run 도 `ArrayRecord(agg.ndarrays)`(fl/strategy.py:181 → strategy.py:228-230)를 다음 라운드에 보낸다. 집계 출력은 float32 캐스팅·정수 버퍼 최댓값(fl/aggregate.py:107, :113; `global_norm` 은 저장 배열과 같은 객체 :116); `np.savez` 는 `arr_%d` 삽입 순서(_npyio_impl.py:778), ArrayRecord 는 `record[str(i)]`·삽입 순서 반환(arrayrecord.py:274-275, :314-317), `assert_compatible` 이 dtype 정확 일치(serialize.py:130-143). 0차원 int64 왕복은 tests/test_fl_strategy.py:174-175 가 고정.
- 오프셋 배선: 루프 번호 소비처는 로그 `[ROUND i/n]`(:203)·`Result` 키(:233)뿐이고 클라이언트·회계·npz 이름은 전부 `server-round` 파생(§2). 오프셋이 aggregate 쪽에 빠지면 복원 셀과 충돌해 `record` 가 ValueError(budget_audit.py:203-207, 내장 트립와이어).
- 실측: 09-04 run 과 09-07 run 의 `global_r001` sha256 일치(§7-5), 적대 검증이 r002·r003 도 일치 확인(0c436ad8…, 3b874b53…; 다른 날·다른 FAB ef15ff95↔31149ece·게임 병행 조건). 단 두 run 모두 "새 프로세스에서 라운드 1 시작" 이라 아래 기각 항목의 증거는 아니다.

**기각된 주장(설계에 반영 완료)**
- "재개 run = 무중단 run bit 동일" 을 논증 완료로 쓴 것: cuDNN 엔진 선택·플랜 캐시·cuBLAS 워크스페이스(env 를 라운드 끝 `unset_deterministic` 이 지우고 :649, :698-703 다음 트레이너가 다시 건다 :686-691)는 C++ 라 .venv 로 확인 불가. **T-eq 로 닫을 가설**로 격하.
- T-hist(09-04 vs 진행 run 의 r001..030 대조)가 이음매 등가의 증거라는 주장: 양쪽 다 단일 액터(pid 39928 / vce_api.py:371-382 가 ClientApp 을 1회 로드해 캐시, raybackend.py:173-176; 풀 크기 `min(cpus/2, gpus/1.0)=1` ray_actor.py:113-121)의 k번째 라운드라 "재사용 경로의 재현성" 만 보여 준다. 결론 문구 한정, 완주 후 실행(HDD 경합).
- "results.csv 가 append 라 격리 없이는 행이 중복" (관점 1) 및 "results.csv 행 수 == E 가 닫힘 조건" (관점 2): 직접 확인 — trainer.py:199-201 `if self.csv.exists() and not self.args.resume: self.csv.unlink()`(로그 :45 `resume=False`). append(:926) 는 한 프로세스 안의 epoch 누적. 따라서 격리 사유는 "포렌식 보존"(r030_c2 완주분), (iv) 는 경고로 강등. 적대 검증 3 이 이 항목을 holds 로 둔 것은 :199-201 을 보지 않은 오판이다.
- "`params_l2_norm` 정확 일치": float64 `np.dot`(serialize.py:153-158) 이라 BLAS 스레드 수에 따라 마지막 비트가 흔들릴 수 있음 → 상대 1e-9.
- 관점 1 의 k 판정 (a)·k=0 경로, 관점 2 의 자동 재개·`segment` 열: §3 기각 목록.
- 관점 1 T-eq 구성(`num-server-rounds=1` 완주 run 을 `--resume-from 1` 로 잇기): `epochs = N` 전역(round_runner.py:233)·`start_epoch`(fed_trainer.py:233)·LR 위치(:248)가 N 의 함수라 R=1 완주 run 은 스케줄이 다르고, R=3 로 1라운드만 돌리면 audit 가 총 epoch≠N 으로 실패(main_det.py:395). → 동일 R·E·N + **외부 중단**(액터 kill / flwr-simulation kill)으로 교체.
- 인용 정정: 관점 1 `ray_client_proxy.py:60-63` 은 레거시 `start_simulation` 경로(결론은 유지); 관점 1 의 현재 run FAB 는 ef15ff95 가 아니라 31149ece; 관점 2 `fl/aggregate.py:380·386`·적대 검증 3 의 `:305·311` 은 둘 다 틀렸다(파일 186줄; fp32 :107, 정수 :113, global_norm :116); 관점 2 `client_det.py:239-241` → :66-67; `strategy.py:222-227` → 대입은 :228-230; "check_amp 결과 사후 대조 불가" → 회계에는 없으나 flwr 로그에 트레이너마다 남는다(실패 로그 :77-78 `AMP: checks passed`, 93건 = 31 라운드 × 3).
- 관점 2 시험 항목 "서버 sent 집합에 resume-from-round": `context.run_config`(server_app.py:94) 키이지 클라이언트 `train_cfg`(:140-152) 키가 아니다 — 삭제.

**조건부 주장(설계에 조건을 박음)**
- 프로세스 이력이 파이썬 가시 수치에 닿는 경로 1개: trainer.py:511-541 — 라운드 첫 epoch(④ 는 매 라운드)에서 OOM·CUBLAS_STATUS_ALLOC_FAILED·CUDNN_STATUS_INTERNAL_ERROR·'unable to find an engine' 이 나면 **배치를 반으로 줄이고 경고 한 줄만 남기고 계속 돈다**(최대 3회). RoundFailure 도 회계 실패도 아니다. 실측 30 라운드 미발화(optimizer_steps 1488/916/402 상수, 로그 'Reducing to batch' 0건). → 회계 검사 `optimizer_steps == epochs_ran × ceil(num_examples/32)` 와 flwr 로그 시그니처 사후 검사를 필수로(재개와 무관하게 함정 #1 의 새 접촉점 — 감독 보고).
- `check_amp` 는 라운드마다 실기 추론으로 판정하며 AssertionError 면 amp=False 로 조용히 진행(checks.py:1039-1044), ConnectionError 면 amp=True 유지(:1032-1033). 회계에 실효값이 없다 → 로그 보존(재개 로그 새 파일) + 클라이언트 계측 추가는 게이트 질문.
- 기존 epoch 경계 재개는 bit 비동일이 소스·자체 시험으로 확정(resume.py:281-287; tests/test_detection_resume.py:231-264 `assert after != ep1`; prefetch_factor=4 build.py:376 라 저장 시점에 다음 순열이 이미 뽑혀 있음). ②③ 에도 해당 — `resumed_from_epoch` 열이 이음매 각주 대상임을 D·E 에 통지. ④ 에서는 부분 라운드 체크포인트를 **격리**(≤1 epoch 재계산이 대가).
- 회계 라운드 완전성: `_cell_from_metrics` 가 두 번째 클라이언트에서 예외를 내면 8행 + 인메모리 부분 셀이 남고 finally 가 그것을 accounting.csv 에 쓴다(round_wiring.py:79-102) → 복원은 3셀 전부 있는 라운드만 measured.
- 오프셋 반쪽 배선(configure 누락·aggregate 적용)은 회계 충돌 없이 잘못된 스케줄 run 을 유효로 닫는다 → `seed == derive_seed`·`budget_fired_at == (r+1)E−1`(실물 1,3,…,59) 불변식 검사가 이를 잡고 "다른 시드 접두 위 재개" 도 막는다.
- 동일 출발: `initial.npz` 캐시는 해시 검증 없이 적재되고(init_weights.py:45-52) `injection_digest` 는 미전송(§2) — ④ 에는 오늘도 대역 내 동일 출발 검증이 없다. 서버가 initial.npz sha256(수동 대조 실측 6c7ec4b0…, 세 사본 일치)·재개 출발 npz sha256 을 audit.json 에 기록하는 것까지가 클라이언트 0줄의 한계.

### 5. 적용 범위와 비교 가능성 (시드 1 vs 2·3, 칸 ②③④, 속도 축 표기)

- **④ 만.** ②③ 은 클라이언트 측 epoch 경계 재개(main_det.py:283 `resume_root=sd/"_resume"/"sep_central"`)가 이미 있고 손대지 않는다. 서버 배선은 ⑦ 과 공통이라 오프셋은 자동 적용되지만 ⑦ 의 어댑터 npz·회계 복원 규칙은 범위 밖 — 서버가 `cell != "sep_fed"` 에서 k>0 을 거부한다(조용한 경로 차단). `train_cfg["num-rounds"]` 는 R 유지(fl/client_vlm.py:111 소비).
- **시드 1:** 진행 중 run(FAB 31149ece, 01:18 기준 global_r004 까지)은 단일 세그먼트 완주가 기본 경로이고 이 스펙이 닿지 않는다. 죽으면 게이트 통과본으로 재개할 수 있는가는 총괄 결정(§8-2) — 접두는 구코드·뒤는 신코드(학습 파일 무변경, FAB 바이트 상이), `global_index.jsonl` 없음(notes). **시드 2·3:** 착수부터 게이트 통과본(`resume-from-round=0`, 라운드별 accounting 덤프·index 포함) — 무중단이면 단일 세그먼트, 중단 시 명시 재개. 세 시드의 클라이언트 학습 경로는 동일 파일·동일 FIXED_OVERRIDES 라 채점 비교는 성립하고, 차이는 서버 기록·재개 기제뿐이다.
- **속도 축:** 시드 1 이 비경합 단독 실측 런(10번 §5)이며 무중단이면 영향 0. 재개 세그먼트의 첫 라운드는 `RoundTimer` 가 서버 main 진입 시 생성되므로(server_app.py:128) 기동 비용이 wall 에 실린다 — 이음매 라운드 = {1} ∪ {k_i+1}. 규칙(제외가 아니라 표시): (i) 50 라운드 전량을 이음매·외부 GPU 점유(sysmon `game_proc`) 플래그와 함께 표로, (ii) 요약 평균은 두 플래그 제외 라운드로, (iii) 이음매 라운드는 별도 행, (iv) ④ 총 벽시계 = progress.jsonl 세그먼트 wall 합(기동 오버헤드 포함). 파서·감시 스크립트는 flwr `[ROUND i/(R−k)]` 대신 원장 round 열·서버 절대 라운드 배너를 본다(post_reboot_C.ps1:5 의 `ROUND 2/50` 시그니처 교체).
- **회복률·통신량:** 모델 궤적이 T-eq 로 등가 확인되면 채점 결과 차이 0 이어야 한다. 통신량은 라운드별 bytes_up/down 불변. 부분 라운드에서 흘린 학습(≤ 1 라운드, 3 클라이언트 ≤ 15.7 분)은 원장에 없고 재개 run 이 같은 라운드를 처음부터 돌므로 R×E=N 은 정확히 50×3 셀 × 2 epoch 로 닫힌다. T-eq 불일치 시 재개 run 은 "통계적 등가 + 이음매 각주" 로 격하할지 비채택할지 게이트가 정한다(§8-6).

### 6. 위험과 대응

| 위험 | 등급 | 대응 |
|---|---|---|
| 재사용 액터의 (k+1)번째 라운드 ≠ 새 프로세스 첫 라운드 (cuDNN/cuBLAS/할당자/check_amp) | 높음 | T-eq(동일 R·E·N·외부 중단·소규모 뷰·pilot 프로파일) 를 채택 조건으로; 불일치 시 텐서별 최대 편차 보고 후 게이트 결정 |
| OOM 자동 배치 축소(trainer.py:511-541)가 batch 32 를 조용히 깸 — 재개 여부와 무관 | 높음 | 회계 `optimizer_steps` 기대값 검사 + 로그 'Reducing to batch' 사후 검사; 함정 #1 접촉점으로 감독 보고 |
| 잔존 epoch 체크포인트가 신원 일치로 조용히 소비(다이제스트는 서버 값 가리킴) | 높음 | `_resume/sep_fed/**/*.pt|*.tmp` 라운드 무관 전수 격리 후에만 진행; ④ 셀 `resumed_from_epoch≠None` 승격은 §8-4 |
| 다른 설정 접두(다른 시드·split·initial) 위 재개 → 무효 run 둔갑 | 치명 | 판정 (ii) 열 대조 + initial.npz sha256 + seed/budget_fired_at 불변식 + 런처·서버 이중 판정 + 접두 sha256 마감 재대조 |
| 찢어진 npz(np.savez 비원자)·찢어진 CSV 마지막 행 | 중간 | (iii) BadZipFile/유한성/l2 → 열린 라운드로 절단; 12 필드 파싱 검사; 앞으로는 tmp→replace |
| 원장 두 벌(append + 집합 감사) | 중간 | 판정 시점 중복 거부 + 마감 행 수 27×R·다중집합 검사 |
| 서버 사전검증 실패·하드킬 시 audit.json 부재 → 런처 트레이스백(`flwr run` rc=0) | 중간 | 런처가 부재를 분류해 SystemExit; 서버는 precheck 실패 시 audit.json 만 쓰고 accounting.csv 미접촉 |
| 사인 미해소 맹목 재개(09-05 형태) | 중간 | 명시 `--resume-from K --reason`; 무진전(같은 k) 거부; 재개 preflight 커밋 여유 ≥ 60 GB·SuperLink 선기동(local_superlink.py:150-171 15 s 함정) |
| Windows 에서 열린 파일에 `os.replace` PermissionError(§7-3 `tail -f` 사고와 같은 부류) | 낮음 | 절단 실패는 원본 무손상 채 SystemExit; 재개 로그는 새 파일이라 원 로그를 만지지 않음 |
| audit.json 신규 필드·`restored:` 값에 D·E 파서 취약 | 낮음 | 열 추가 없음(FIELDS·25열 불변); scripts/citation_ban.py:100·139·145 는 열 존재만 검사; D 에 계약 통지 |

### 7. 검증 계획 (시험 파일·검증 내용·스모크·실행 규모)

| 시험 | 파일 | 검증 내용 |
|---|---|---|
| T-offset | tests/test_fl_strategy.py | `round_offset=30`, 상대 1 → `config["server-round"]==31`, on_round_end 31, RoundFailure 문구 "라운드 31"; 기본 0 이면 기존 시험 불변; 오프셋 누락 시 복원 셀과 `record` ValueError(이빨) |
| T-npz | tests/test_fl_server_app.py [신규] | 저장→적재 `np.array_equal`·dtype·shape 전부(0차원 int64 포함), `.files` 를 뒤섞은 픽스처에서도 arr_i 순서; l2 상대 1e-9 초과·키 수·dtype 불일치 거부; `_write_arrays` 원자화 후 `global_index.jsonl` 행 |
| T-judge | tests/test_fl_resume_ledger.py [신규] | 09-04 원장 축소 복제 → k=30·잔해 목록; npz 만 있고 원자 미완비 → 거부; 앞 절반 잘린 npz → 열린 라운드; 텐서 하나 바꾼 npz → k−1; 27행 한 번 더 append → 거부; 다른 run-stamp/split_hash 행 → 거부; 클라이언트 2/3 라운드 → 열림; 찢어진 마지막 CSV 행 → 잔해; results.csv 부재는 경고만 |
| T-truncate | 동상 | 절단 중 예외 주입 시 원본 무손상; 전후 sha256·행 수 == 27×k 기록; `_resume/**/resume_ep*.pt` 격리 이동; latest.npz sha 일치 시 보존/불일치 시 격리; 열린 파일에 대한 PermissionError 가 SystemExit 로 올라감 |
| T-acct | tests/test_detection_fed.py(:138-220 의 `AccountingMatrix` 시험 옆) 또는 신설 tests/test_budget_audit_restore.py — `tests/test_train_budget.py` 는 configs R·E·N 고정 시험이라 부적합(비판 L9) | `to_csv→from_csv→to_csv` 바이트 동일(momentum·arg_momentum·participated·None 포함); restored 셀은 `restored_cells` 로만 분류되고 "인용하지 마라" notes 없음; 복원 90 + 신규 60 = 150 셀 ok; 셀 하나 빼면 빈 셀 실패; seed/budget_fired_at/optimizer_steps 기대값 위반 → failure(이빨) |
| T-guard | tests/test_main_det_cell4.py [신규] | 부분 원장·잔해 + `--resume-from` 없음 → 거부 문구(k 포함); K≠k·사유 없음 → 거부; 무진전 → 거부; audit ok + global_r050 → 스킵; 재개 로그가 새 파일이고 원 로그 sha256 불변; audit.json 부재 → 분류된 SystemExit; progress.jsonl 세그먼트 행(성공·실패) |
| T-dup | tests/test_atomic_log.py | 같은 (round, client, metric) 두 번 쓴 로그에서 `audit_rounds` 실패(현재 코드는 통과 — 빨간 뒤 초록) |
| T-keys | tests/test_fl_round_wiring.py | `resume-from-round` 선언(:203 목록 확장); `_flwr_run_config` 출력이 `parse_config_args` 로 파싱돼 int; fl/ 에 'server-round' 리터럴 0 |
| T-smoke-resume (resource_heavy) | tests/test_fl_round_wiring.py | 스모크 칸 R=4·E=1, `smoke-fail-at="2,1"` → audit ok=false·k=2 → `resume-from-round=2` 재실행 → audit ok, 원자 108행(중복 0), 12셀(0..1 restored, 2..3 measured), resume_events[0].from_round==2, segments [[1,2],[3,4]], global_r002 sha 불변, DO_NOT_CITE 유지. 스모크 클라이언트가 서버 입력을 무시하므로(client_app.py:75-79) 재주입 이빨은 `arrays_out = arrays_in·(1+0.01c)+c`(스모크 전용 변경 — "학습 클라이언트 경로 0줄" 로 문구 정정) 뒤 무중단 R=4 의 global_r004 sha 일치 |
| T-eq (GPU, 게이트 시점) | scripts/verify_resume_eq.py [신규] | 소규모 뷰·pilot 프로파일·R=3·E=2·N=6·같은 initial.npz: run A 무중단 참조 → run B 는 라운드 3 진입 후 **외부 중단**(변형 1 ClientAppActor pid taskkill / 변형 2 flwr-simulation kill = finally 미실행) → 판정 k=2·격리 → `--resume-from 2` → global_r003 sha256·원장 param_l2/global_l2/bn_divergence·optimizer_updates 전부 일치. 학습 경로에 kill 스위치를 두지 않는다. 실행 순서 c0→c1→c2 를 로그로 대조. 이것이 §4 기각 항목 1 을 닫는 유일한 시험 |
| T-hist (CPU, 완주 후) | scripts/compare_global_npz.py [신규] | 09-04 run r001..030 ↔ 시드 1 run 같은 파일 sha256·l2 — 결론은 "재사용 경로 재현성" 으로 한정, 진행 중 run 완주 뒤 실행(E: HDD 경합 회피) |

실행 규모: 단위 시험은 GPU 무관·수 초. T-smoke-resume 는 실 flwr 런타임(Ray 기동, 수 분). T-eq 는 GPU 수십 분(추정) — 시드 1 ④ 완주(≈09-07 13:20) 뒤, 후속 launcher 의 시드 2 자동 기동(chain_follow.ps1:14-26) 전에 창을 잡아야 한다.

### 8. 게이트 질문 (총괄이 결정할 것만)

1. 트리거를 **명시 전용**(`cell4 --resume-from K --reason`, 자동 재개 없음, 상한은 "무진전 거부" 만)으로 확정하는가. 두 경로를 다 구현하지 않는다.
2. 시드 1 현재 run 이 죽을 경우, 접두(구코드 FAB 31149ece) 위에 게이트 통과본(다른 FAB)으로 재개를 허용하는가 — 학습 파일 무변경이라 등가 논증은 서지만 FAB 바이트가 다르다.
3. T-eq 의 GPU 창: 시드 1 ④ 완주 후 후속 launcher(시드 2 자동 기동)를 잠시 세우는가, 아니면 시드 2 ④ 완주 뒤로 미루고 그때까지 재개를 열지 않는가.
4. ④ 한정으로 `resumed_from_epoch ≠ None` 셀을 notes 가 아니라 **failure** 로 승격(감사 기준 강화, ②③ 은 유지)하는가.
5. 클라이언트 계측 추가(MetricRecord 에 amp·cudnn_deterministic·cudnn_benchmark·batch 실효값 — 학습 산술 불변이나 fl/client_det.py·round_runner.py 변경)를 같은 게이트에 묶는가.
6. T-eq 불일치 시 처리: "통계적 등가 + 이음매 각주" 로 격하해 채택하는가, 비채택하는가. 결정은 의사결정로그에.
7. OOM 자동 배치 축소(trainer.py:511-541)를 지정 함정 #1 의 접촉점으로 등록하고, `optimizer_steps` 기대값 검사를 ②③ 회계에도 소급 적용(시드 1 실물 재감사)하는가.
8. audit.json 신규 필드(`resume_events`·`segments`·`restored_cells`)와 `value_source="restored:accounting.csv"` 값에 대해 D·E 파서 계약 통지를 게이트 조건으로 두는가.

### 9. 미결·미확인

- cuDNN 플랜 캐시·엔진 선택, cuBLAS 워크스페이스 크기의 프로세스 1회 캐시 여부, 캐싱 할당자 상태가 커널 선택에 닿는지 — C++ 라 .venv 에서 읽을 수 없다. T-eq 만이 답한다.
- `verify_warm_vs_fresh` 실측이 없다: 현재까지의 bit 동일 실측(r001~r003)은 모두 "새 프로세스에서 라운드 1 시작" 형태다. r004 이후 대조는 완주 후.
- 한 라운드 안의 클라이언트 실행 순서(c0→c1→c2)를 우리 코드가 고정하는지 — flwr 백엔드 스케줄링 미확인(로그상 순차). T-eq 에서 로그 대조.
- SuperLink `state.db`(52.9 MB + WAL)에 하드킬된 run 이 'running' 으로 잔존하는지와 새 run 간섭 여부 — 시뮬레이션 메시지 상태는 flwr-simulation 프로세스 인메모리(run_simulation.py:270-272)이고 SuperExec 는 pending 만 집는다(run_superexec.py:234-235)까지 확인, 하트비트 실패 전환 코드는 못 찾음(추측: 간섭 없음).
- Windows 에서 다른 프로세스가 연 파일에 `os.replace` 가 PermissionError 를 내는 경우 — §7-3 `tail -f` 사고에서 유추, 절단 경로에서는 미실측.
- T-eq 용 소규모 뷰: `profile`·`views-root` 가 이미 run-config 키다(scripts/main_det.py:306-307, fl/server_app.py:173-181, fl/client_app.py:129). `scripts/verify_resume.py:41-55` 처럼 파일럿 스냅샷(`data/processed/aihub71761_rt_v1_pilot3000`)에서 `build_yolo_view` 로 작은 뷰를 만들어 `views-root` 로 넘기면 `fraction` 없이 성립한다 — 미확인이 아니라 설계 미기재였다(비판 L10). T-eq 의 뷰 구성(스냅샷·`views-root`·`profile="pilot"`·`num-examples`)을 T-keys 에 고정한다.
- Ray object store 5.19 GB 의 커밋 계상 여부(§7-1) — 이 스펙과 무관하나 재개 preflight 문턱 60 GB 의 마진 근거로 남는다.
- 적대 검증이 보고한 sha256 실측(r002·r003, initial.npz 세 사본, latest==r030)은 본 절에서 재실행하지 않았다 — 게이트 전 CPU 스크립트로 재확인 권고(진행 중 run 완주 후).

---

## 부록 A. 완결성 비판 (원문 — 조립자 요약 없이 표로만 정리)

비판자 요약: 완결성 검토 결과(읽기 전용, HEAD 694ee16 실측). 세 절의 소스 인용은 대부분 정확했고(ultralytics 8.4.120·torch 2.11.0+cu128·flwr 1.33.0·ray 2.55.1·numpy 2.2.6 실독; 트레이서·sysmon·실패 원장·raylet.out 인용값 전부 재확인), R §9 가 "재실행 안 함" 으로 남긴 sha256 4건(r002 0c436ad8…·r003 3b874b53…·initial 세 사본 6c7ec4b0…·latest==r030 58bf85f9…)은 본 검토에서 일치 확인했다. 그러나 세 절을 함께 시드 2·3 에 올리면 깨지는 지점이 있다. (1) B 가 원자 로그에 `val_loader_workers` 행을 추가하면 라운드당 행 수가 27→30 이 되는데 R 의 닫힌 라운드 정의·마감 검사·T-smoke-resume 가 27/108 을 리터럴로 박았고, 시드 1 접두(클라이언트 지표 8개)는 R Q2 경로에서 "미완비" 로 거부된다. (2) R 이 `budget_audit.audit()` 에 넣는 세 불변식(seed==derive_seed·budget_fired_at·optimizer_steps=epochs×ceil(n/32))은 칸 한정이 없어 스모크(seed=base-seed, steps=5, budget=-1: fl/client_app.py:95-99)·⑦ 까지 같은 `finalize_accounting` 을 타므로(fl/server_app.py:163, fl/pilot_sim.py:88-116) preflight 와 회귀 스모크가 즉시 깨진다 — 동시에 이것은 "회계 감사 기준" 변경이라 게이트 항목이어야 한다. (3) 게이트 창: B Q1 "~30분", D T5 "22분", R T-eq "수십 분" 이 각각이고 launcher(`logs/recover/chain_follow.ps1`, 미추적·정지 훅 없음, CHAIN_DONE 120 s 폴링 직후 시드 2 기동) 정지 절차·소유자가 없다. B T7 과 D T5 는 같은 run 이며 R 의 pyproject 변경이 FAB 해시를 바꾸므로 오라클은 세 변경이 다 착지한 최종 트리에서 1회 돌아야 한다. (4) 지금 실물 `_resume/sep_fed/r005_c1/resume_ep0010.pt`(95 MB, 01:44) 가 있다 — 10번 §7-2 의 "파일 0" 은 시점 운이었고, R 착지 전 시드 1 이 죽어 같은 명령을 재기동하면 신원 일치로 조용히 소비된다(detection/resume.py:281-287 bit 비동일). 이 중간 절차가 어느 절에도 없다. (5) D 의 HEAD 표기(2cf5a0e)와 B·D 의 10번 정정 대상 문구는 694ee16 에서 이미 일부 반영돼 있어(§7-1 L175 "검증 16 요청 → 실효 12", §7-5 L296-301 "관측 20 = 8 + 12") 남은 정정은 단가 1.8→1.10~1.17·"21~22"→13.2~14.0·§7-4 L247 뿐이다. (6) 소수 인용 오류: R T-acct 파일(`tests/test_train_budget.py` 는 configs R·E·N 고정 시험, 회계 시험은 `tests/test_detection_fed.py:138-220`), D `fl/client_app.py:97`(스모크 경로; 검출은 `fl/client_det.py:85`), D `build.py:372`(실제 376), R `arrayrecord.py` 경로(`flwr/app/message/arrayrecord.py:275·317`). 불변조건과 직접 충돌하는 문장은 없었으나 R 의 감사 기준 강화, B Q5 의 FIXED_OVERRIDES 키 추가, `_headroom_wait` 문턱(B +41/+55 · D ≥55 · R 60 하드코딩)이 게이트 승인 없이 설계 본문에 들어가 있다.

### A-1. 누락·오류 (20건)

| 심각도 | 위치 | 누락·오류 | 제안 |
|---|---|---|---|
| critical | R §3 변경 지점 표 `detection/budget_audit.py` 검사 추가 / §7 T-acct | 새 불변식 `seed == derive_seed(base, r, c)`·`budget_fired_at == (r+1)·E−1`·`optimizer_steps == epochs_ran × ceil(num_examples/32)` 가 칸 한정 없이 `AccountingMatrix.audit()` 에 들어간다. `audit()` 은 ④ 뿐 아니라 스모크·⑦ 도 같은 `finalize_accounting` 으로 탄다(fl/server_app.py:163-166, fl/pilot_sim.py:88-116). 스모크 클라이언트는 `seed = base-seed`(fl/client_app.py:98), `optimizer-steps = 5`, `budget-fired-at = -1`(L95-99) 을 보내므로 preflight(scripts/main_det.py:431-441)와 tests/test_fl_round_wiring.py 의 resource_heavy 스모크 4건, ⑦(VLM steps 의미 다름)이 즉시 실패한다. 실물 근거로는 ④ 만 맞다(accounting.csv r0 c0: steps 1488 = 2×ceil(23807/32), fired 1; r1: 3). | 불변식을 `cell == "sep_fed"`(또는 profile/batch 를 인자로 받는) 경로로 한정하거나 ④ 판정기(`fl/resume_ledger.py`)로 옮긴다. T-acct 에 스모크 셀·⑦ 셀 픽스처를 넣어 통과를 고정하고, 감사 기준 강화 자체를 게이트 질문으로 올린다(불변조건 '회계 감사 기준'). |
| high | R §3 닫힌 라운드 정의 (i) '정확히 27개' · `finalize_accounting` '행 수 == 27×R' · T-smoke-resume '원자 108행' ↔ B §3 `fl/round_wiring.py:86-98` 지표 행 추가 | B 가 `val_loader_workers` 지표 행을 추가하면 라운드당 원자 행이 3×9+3 = 30 이 되고, 스모크·⑦ 도 같은 기록기라 -1 행이 붙는다. R 은 27·108 을 리터럴로 쓰고 '기대 집합' 을 현재 코드 상수로 정의하므로, (a) B+R 동시 착지 시 R 의 판정·마감이 전부 실패하고, (b) 시드 1 원장(클라이언트 지표 8개: 실물 atomic_log.csv 지표명 8종 × 90 = 720 + 서버 3 × 30 = 90)은 R Q2 경로(게이트 통과본으로 시드 1 재개)에서 '(i) 미완비' 로 거부된다. | R 의 기대 집합을 `CLIENT_ROUND_METRICS` 로 승격하되 행 수를 `3×len(CLIENT_ROUND_METRICS)+3` 로 계산하고, 접두 판정은 '접두 첫 라운드가 쓴 지표 집합' 을 기준으로 하거나 버전 표를 둔다(시드 1 = 8, 시드 2·3 = 9). T-judge 에 '8-지표 접두 위 9-지표 코드' 픽스처를 추가한다. |
| high | R §2·§4·§6 · B §6 — 시드 1 이 R 착지 전에 죽는 경우 | 실물 `outputs/main_c/seed1/_resume/sep_fed/r005_c1/resume_ep0010.pt`(95,158,635 B, 09-07 01:44) 가 지금 존재한다 — 라운드 6 C2 의 첫 epoch 경계 덤프다. 10번 §7-2 의 '91 디렉터리 / 0 파일' 은 두 번 다 첫 epoch 안에서 죽은 시점 운이었다. 같은 명령 재기동(현 `stage_cell4`)이면 `ResumeIdentity`(run_id main_s1·round·client·seed…, detection/resume.py:84-91) 가 일치해 조용히 소비되고 `resumed_from_epoch≠None` 인 채 bit 비동일 궤적이 된다(resume.py:281-287, tests/test_detection_resume.py:231-264). R 은 게이트 통과본 재개(Q2)만 묻고, R 착지 전 사망 시의 중간 절차(원장 이동 + `_resume/sep_fed/**` 격리 + 라운드 1 재실행 13.1 h vs R 대기)를 어디에도 적지 않았다. | R §5 또는 §8 에 '착지 전 사망 시 절차' 를 명시하고, 즉시 지시로 '같은 명령 재기동 전 `_resume/sep_fed/**/*.pt\|*.tmp` 격리 필수' 를 10번에 기록한다. |
| high | B §7 T7 · D §7 T5 · R §7 T-eq/T-smoke-resume 실행 순서와 FAB 동일성 | B T7(④ 1라운드 오라클)과 D T5 는 같은 run 인데 별건으로 적혀 있다. R 의 `pyproject.toml [tool.flwr.app.config] resume-from-round=0` 추가는 FAB 해시를 바꾸므로(flwr/cli/build.py:182-184, .gitignore 적용 flwr/cli/utils.py:549-580) B-only 트리에서 돌린 T7 의 FAB 는 시드 2 실제 FAB 와 다르다. B 는 '커밋 해시와 FAB 해시를 함께 기록' 만 말하고 '최종 병합 트리에서 1회' 를 요구하지 않는다. R 의 T-eq 도 B 가 바꾼 `fed_trainer.py`/`round_runner.py`/`client_det.py` 를 포함한 트리에서 돌아야 R §4 의 '클라이언트 경로 무변경' 전제가 시드 2·3 실물과 같아진다. | 순서를 하나로 확정: 세 변경 착지 → CPU 회귀 전량(B T1~T5·T8, R T-offset~T-keys·T-dup, D T3·T4) → B T6(GPU ~10 분) → R T-smoke-resume(Ray) → B T7 = D T5(④ 1라운드, 최종 FAB 해시 기록, sha256 5c0378a1… 대조) → R T-eq → 시드 2 기동. FAB 해시 == 시드 2 run 의 FAB 해시를 착지 조건으로 명시. |
| high | B §5 착지 규칙 · B §8-1 · R §7 실행 규모 · D §5 적용 시점 — launcher 정지 절차 | `logs/recover/chain_follow.ps1`(git 미추적) 은 CHAIN_DONE 을 120 s 폴링(L14)해 즉시 시드 1 export(L18-21)와 시드 2 chain(L24-26)을 띄우며 정지 훅이 없다. 세 절 모두 '시드 2 기동 전' 을 말하지만 누가·어떻게(프로세스 종료? 센티널?) 내리고 누가 내려간 것을 확인하는지, 게이트가 시드 1 완주(≈09-07 13:20) 전에 끝나지 않을 때 GPU 를 비워 둘지 시드 2 를 R 없이 시작할지가 없다. B Q1 의 '~30 분' 은 B T6+T7 만 센 값이고 R T-smoke-resume·T-eq(수십 분)·D T5 를 더하면 1.5~2 h 급이다. | 공통 운영 절차 1개(launcher 정지 명령·확인·재기동, 총 창 추정, 소유자)를 세 절 밖 공통 절로 빼고 게이트 질문을 하나로 합친다. |
| medium | R §3 `_save_round`/`_write_arrays`/`on_save` 원자화 | 'accounting.csv 원자 덤프 → npz(tmp→replace) → global_index.jsonl append' 라 했지만 `AccountingMatrix.to_csv` 는 `open("w")` 직접 쓰기(detection/budget_audit.py:370), `global_index.jsonl` append 와 `audit.json`(fl/round_wiring.py:149) 도 비원자다. 찢어진 마지막 행을 판정기가 '잔해' 로 처리한다고 §6 에 적었지만 어느 파일에 tmp→replace 를 적용하는지 목록이 없다. | 원자화 대상 파일 목록(npz·accounting.csv·audit.json·global_index.jsonl 마지막 행 처리)을 표로 고정하고 T-npz/T-truncate 에 각각 픽스처를 둔다. |
| medium | R §3 의사코드 · §7 T-judge — k == R 경계 | 50 라운드가 전부 닫혔는데 마감 중 하드킬로 `audit.json` 이 없는 경우: 완료 마커(audit ok AND global_r050)는 불성립, 판정기는 k=50, `strategy.start(num_rounds=0)` 이 되고 `_load_resume_point(k=50)`·복원 150 셀·마감만 남는다. flwr `Strategy.start` 는 `range(1, 1)` 로 빈 루프를 돌아 returned Result.arrays 가 initial 그대로다(flwr/serverapp/strategy/strategy.py:201). 이 경로의 의도 동작·시험이 없다. | k==R 을 '마감 전용 재개' 로 정의(서버 루프 0 라운드, 회계 복원 후 finalize)하고 T-judge/T-guard 에 픽스처 추가. |
| medium | R §7 T-acct 파일 지정 | `tests/test_train_budget.py` 는 configs/base.yaml 의 R·E·N 고정 시험 3건뿐이다. `AccountingMatrix`/`AccountingCell` 시험은 `tests/test_detection_fed.py:138-220` 에 있다. | T-acct 를 `tests/test_detection_fed.py`(또는 신설 `tests/test_budget_audit_restore.py`)로 옮긴다. |
| medium | R §9 '소규모 뷰를 어느 API 로 만드는지 미확인 — flwr 경로에는 fraction 을 넘길 구멍이 없다' | `profile` 과 `views-root` 는 이미 run-config 키다(scripts/main_det.py:306-307, fl/server_app.py:173-181, fl/client_app.py:129). `scripts/verify_resume.py:41-55` 처럼 파일럿 스냅샷(data/processed/aihub71761_rt_v1_pilot3000)에서 `build_yolo_view` 로 작은 뷰를 만들어 `views-root` 로 넘기면 `fraction` 없이 T-eq 가 성립한다. 미확인이 아니라 설계 미기재다. | T-eq 의 뷰 구성(스냅샷·`views-root`·`profile="pilot"`·`num-examples`)을 명시하고 `_flwr_run_config` 의 project/views-root 인자만으로 되는지 T-keys 에 고정. |
| medium | B §3 표 `docs/…/10_검출3칸_C.md` 정정 행 · B §4 K6 · B §8 Q3 · D §3 정정 표 · D §7 T0 · D §8 Q3 — 대상 문구가 HEAD 와 어긋남 | D 는 'HEAD 2cf5a0e' 기준이라 적었지만 실제 HEAD 는 694ee16(2cf5a0e 이후 문서 커밋 2건, 학습 코드는 bcc2a98 이후 변경 0 — `git diff --stat bcc2a98..HEAD -- fl detection scripts/main_det.py pyproject.toml configs` 비어 있음). 694ee16 의 10번은 이미 §7-1 L175 '검증 16 요청 → 실효 12', §7-5 L296-301 '관측 20 = 학습 8 + 검증 12' 로 정정돼 있다. 남은 오류는 §7-5 L293-294 '20개 × 1.78~1.86 GB', L300 '12 × 1.8 ≈ 21~22 GB', §7-4 L247 '−12~19' 와 '④ 만' 문구, §7-1 L177 'object store … 계상 여부 미확인' 이다. B §4 가 인용한 '§7-5 L296-300 의 라운드 1 bit 동일' 은 L303-312 다. | B·D 의 정정 표를 694ee16 본문 기준으로 다시 쓰고, 두 절이 다른 절감치(B '−13.7 GB(+α ≤ 0.3)' vs D '−13.2~14.0')를 내지 않게 한 문구로 통일한다. |
| medium | B §4 조건부 주장 · B §9 'Ray 가 액터에 OMP_NUM_THREADS 를 주입하는지 소스 미인용' ↔ D §2-④ · §3 표 | D 가 `ray/_private/utils.py:250, 265-266`(배정 CPU 2 → `OMP_NUM_THREADS=2`) 를 인용해 B 의 미확인을 이미 닫았다. 두 절 모두 ②③ 은 이 주입 없이(main_det 프로세스, OMP 기본) 돈다는 ②③↔④ 호스트 차이를 '호스트 구성' 열에 적지 않았다. 이 차이는 GPU 산술에는 닿지 않지만 액터의 CPU float64 `params_l2_norm`(detection/serialize.py:153-158) 마지막 비트에 닿을 수 있어 R 의 '상대 1e-9' 근거와 같은 항목이다. | B §9 에서 D 인용으로 대체하고, 호스트 구성 표에 'OMP_NUM_THREADS: ②③ 기본 / ④ 액터 2' 행을 추가한다(실험 조건 표·논문 각주 대상). |
| low | B §3 대안 표 · B §8 Q5 `FIXED_OVERRIDES["workers"]=8` 등록 ↔ R §3 '손대지 않음 … round_runner.py:37-60' | B 는 5칸 공통 고정 dict(detection/round_runner.py:39-60) 에 키를 추가하자고 묻고, R 은 같은 범위를 '무변경' 전제로 등가 논증을 세운다. 값은 8 로 불변이지만 `FIXED_PILOT`·`FIXED_PROBE_NONDET` 가 상속하고 `train_round` 가 `extra_overrides` 충돌을 거부하므로 프로파일 고정 항목 자체가 늘어난다. | 두 절이 같은 답을 참조하게 하고, 승인 시 R 의 '손대지 않음' 행에서 round_runner.py 를 제외하거나 '값 불변 키 추가 제외' 로 문구를 고친다. |
| low | B §6 위험표 'OOM 자동 재시도 … 순간 최대 28 워커(B 16)' | 재구축(trainer.py:535 `_build_train_pipeline`) 시 옛 `test_loader` 는 `self.validator.dataloader`(detect/train.py:209, engine/validator.py:121)가, 옛 `train_loader` 는 루프의 `pbar = enumerate(self.train_loader)`(trainer.py:452) 가 참조를 잡아 `__del__` 이 지연된다. 순간 워커 수는 옛 20 + 새 20 = 최대 40(B off) / 8+8 = 16(B on) 이지 28 이 아니다. 추정치 표기가 없다. | '추정' 으로 표기하고 상한을 40/16 으로 정정하거나 산식을 적는다. |
| low | B §1·§2 'spawn 창 최대 76.65 @00:22:35 < 정상 상태 최대 80.43' | 트레이서에서 00:21:20~00:22:40 구간 최대는 77.76 @00:22:40 이다(B §2 자체 Δ 계산 '65.73→77.76' 과 같은 값). 창을 00:22:35 에서 끊어 76.65 로 적었다. 결론(플래토 > spawn 창)은 유지된다. | 'GPU 할당 직후 00:22:40 까지 77.76' 으로 정정. |
| low | D §2-④ '통신량 `fl/client_app.py:97` → `detection/serialize.py:161-163`' · D §9 '`build.py:372` prefetch_factor=4' · R §3 '`arrayrecord.py:274-275, :314-317`' | L97 은 스모크 클라이언트(`smoke_client_round`)다. 검출 경로는 `detection/round_runner.py:338 payload_nbytes` → `fl/client_det.py:85`. `prefetch_factor=4` 는 `ultralytics/data/build.py:376`. ArrayRecord 는 `flwr/app/message/arrayrecord.py:275(record[str(i)])·317(values() 순서)` 이고 R 은 경로를 안 적었다. | 인용 정정. |
| low | D §3 경로 A · §7 T4 — Ray 노드 CPU 2 를 액터가 전부 점유하는 상태 | `ray.init(num_cpus=2)` + 액터 `{CPU: 2}` 면 노드 유휴 CPU 0 이다. flwr 시뮬레이션 경로가 액터 메서드 외 Ray task 를 내지 않는 것은 확인했으나(`raybackend.py:174`, `ray_actor.py:385·451` terminate.remote 뿐) D 는 이를 전제로만 두고 시험(T3~T6)에 카나리가 없다. | 경로 A 채택 시 T3 에 'flwr/simulation·fleet/vce 에 `@ray.remote`/`.remote(` 가 액터 정의·메서드 외 0건' 소스 카나리를 추가. |
| medium | R §5 속도 축 · B §5 속도 축 — 격리된 세그먼트 행의 취급 | R 은 k 초과 행을 절단·격리하고 '④ 총 벽시계 = progress.jsonl 세그먼트 wall 합' 이라 했지만, 격리된 열린 라운드 행(예: 10번 §5-1 이 속도 표본으로 쓴 `sep_fed_failed_20260904` 방식)이 속도 축 표본인지 아닌지 말하지 않는다. B 는 시드 2·3 원장을 `val_loader_workers` 로 표시·비합산, 10번은 외부 GPU 점유 표시, R 은 이음매 표시 — 같은 라운드 표에 플래그 3종이 붙고 요약 평균 제외 규칙이 절마다 다르다. | 플래그 스키마(이음매·외부 GPU·val_loader_workers·격리 세그먼트)와 요약 평균 규칙을 한 표로 통일해 E·D 에 통지. |
| medium | R §3 `stage_cell4` '재개 preflight 커밋 여유 문턱 ≥ 60 GB' · B §8 Q6 · D §8 Q4 | `_headroom_wait(need_commit_gb=16.0)`(scripts/main_det.py:148) 상향을 B 는 '+41(B 적용)/+55' 로, D 는 '≥ 55' 로 게이트에 묻는데 R 은 60 을 설계 본문에 박았다. 셋 다 launcher 문턱이 스테이지 진입 시점(골)만 보고 플래토를 못 막는다는 B §1 의 지적을 반영하지 않았다. | 값·위치(체인 공통 vs 재개 전용)를 게이트 질문 하나로 합치고 R 본문의 60 은 '게이트 값' 으로 바꾼다. |
| low | R §5 '`post_reboot_C.ps1:5` 의 `ROUND 2/50` 시그니처 교체' | `logs/recover/*.ps1` 은 git 미추적이라 교체 소유자·시점이 없고, R 의 `[ROUND i/(R−k)]` 로그 변경은 40 분 감시(post_reboot_C.ps1 9단계)를 오판하게 만든다. | 복구 스크립트의 저장소 이관 여부(D Q6 확장) 또는 교체 담당을 명시. |
| low | B §7 T6 'resume_verify 뷰(val 353 → … 89 배치 → B off 에서 nw=12)' | scripts/verify_resume.py 는 파일럿 스냅샷·FRACTION 0.022·ROOT outputs/probe_c/resume_verify 를 쓰며 val 크기는 소스에서 확인되지 않는다(스냅샷 실물 미독). nw=12 가 성립하려면 val 배치 수 ≥ 12 여야 한다. | T6 착수 시 뷰 생성 로그의 val 장수·배치 수를 기록해 nw=12 성립을 실측으로 남긴다. |

### A-2. 절 간 상충 (8건)

| 사이 | 상충 | 해소안 |
|---|---|---|
| B §3(`fl/round_wiring.py:86-98` 지표 행 추가 · `fl/client_det.py` 메트릭) ↔ R §3 닫힌 라운드 정의 (i)·마감 검사·T-smoke-resume | B 적용 후 라운드당 원자 행 30(클라이언트 9×3 + 서버 3), R 은 27·108 리터럴과 '현재 코드 상수 = 기대 집합'. 시드 1 접두(8 지표)는 R Q2 에서 거부. R 의 '클라이언트 계약 무변경' 전제도 B 가 `client_det.py` 를 바꾸면 문면상 깨진다. | 행 수를 상수에서 유도, 기대 집합을 접두 첫 라운드 기준 또는 버전 표로; R 의 '무변경' 문구를 'B 이외 무변경' 으로 고치고 T-eq 를 B 포함 트리에서 실행. |
| B §7 T7 ↔ D §7 T5 ↔ R §7 T-eq/T-smoke-resume 및 R §3 `pyproject.toml` 변경 | B T7 과 D T5 는 동일한 ④ 1라운드 재실행인데 별건이고, R 의 pyproject 키 추가가 FAB 해시를 바꿔 B-only 트리의 T7 은 시드 2 FAB 와 달라진다. 창 추정도 B '~30 분'·D '22 분'·R '수십 분' 으로 갈린다. | 세 변경 착지 후 최종 트리에서 T7(=T5) 1회, 그 뒤 T-eq; 총 창을 합산해 게이트 Q1 하나로. |
| B §8 Q6 ↔ D §8 Q4 ↔ R §3 `stage_cell4` 60 GB | 같은 `_headroom_wait(need_commit_gb)` 문턱을 B 는 +41/+55, D 는 ≥55(별도 게이트), R 은 60 으로 본문에 하드코딩. | 값·위치를 한 질문으로 합치고 R 본문은 '게이트 값' 참조로. |
| B §4 조건부·§9(OMP 미확인) ↔ D §2-④(`ray/_private/utils.py:250,265-266` 인용) | 같은 사실을 한 절은 미확인, 다른 절은 확인으로 적었고 ②③↔④ 의 OMP 비대칭은 둘 다 호스트 구성 표에 없다. | B 가 D 인용을 받아 미확인 목록에서 지우고 호스트 구성 표에 행 추가. |
| B §5 속도 축(val_loader_workers 표시·시드 간 비합산·트레이너 오버헤드 분해 각주) ↔ R §5 속도 축(이음매 표시·격리 행·세그먼트 wall 합) ↔ 10번 §5-1(외부 GPU 점유 표시) | 플래그 3종과 요약 평균 제외 규칙이 절마다 다르고, R 이 절단·격리한 열린 라운드 행이 속도 표본인지 미정. | 단일 플래그 스키마·평균 규칙·격리 행 취급을 정해 D·E 통지. |
| B §8 Q5(`FIXED_OVERRIDES["workers"]=8`) ↔ R §3 '손대지 않음 round_runner.py:37-60' | R 이 무변경으로 선언한 dict 에 B 가 키 추가를 묻는다. | 게이트 답 하나를 두 절이 공유; 승인 시 R 문구 수정. |
| B Q2(적용 시드 2/3/없음) ↔ R Q3(T-eq 창: 시드 2 전 vs 시드 2 ④ 완주 후) ↔ D Q2(부속) | 세 변경의 시드 경계가 따로 정해지면 '시드 2 ④ = B on·R off, 시드 3 = B on·R on' 같은 혼합이 생기는데 그 비교 가능성 서술이 없다. 시드 1(코드 무변경)과의 비교는 B 는 오라클 bit 동일, D 는 '호스트 구성 차이 1건', R 은 '서버 기록·재개 기제만 차이' 로 각각 서술한다. | 게이트에서 '세 변경 동일 시드 경계' 원칙을 먼저 정하고, 실험 조건 표에 시드별 코드 커밋·FAB·호스트 구성 열을 한 표로 둔다. |
| D 헤더 'HEAD 2cf5a0e' · D T0/Q3 · B K6/Q3 ↔ 실제 HEAD 694ee16 의 10번 본문 | 두 절이 정정하자는 '검증 16' 은 이미 정정돼 있고, 남은 오류(단가·−12~19·'④ 만'·plasma 미확인)와 두 절의 절감치 표기가 다르다. | 694ee16 기준 정정 표 하나로 통일. |

### A-3. 미검증 채 남은 주장 (19건)

- B: 검증 프리페치·핀 스레드의 액터 내 몫 α ≤ 0.3 GB — 트레이서 가시 증분 상한이지 분해 실측 아님(B 자체 표기).
- B: 액터 priv +7.13 GB(00:22:20→22, 9.558→16.684 실측 확인)가 WDDM 의 GPU 할당 호스트 커밋 계상이라는 해석 — 수치 상관(peak_vram 7.7~7.8)뿐, OS 동작 소스 미독.
- B: raytune 콜백이 등록되되 무동작 — `SETTINGS["raytune"]=true`(settings.json 실측)·`ray/air/session.py:1`·`raytune.py:8,31,36-42` 는 확인했으나 런타임 import 성공은 미실행.
- B: `events.py` GA4 텔레메트리 실제 전송 여부(ONLINE·IS_PIP_PACKAGE 실효값) — settings.json `sync: true` 만 확인.
- B: T6 뷰 규모(val 353·89 배치·nw=12) — 스냅샷 실물 미독.
- B: 벽시계 −63~69 s/라운드(≈ −7 %) — spawn 타임스탬프(8개 00:21:20~41, 12개 00:21:56~00:22:19 실측 확인) 기반 추정, B 적용 실측 없음.
- B: (g) cuDNN 워크스페이스 폴백이 커밋 고갈 국면에서 수치를 바꿀 가능성 — torch C++ 미독(B 자체 '추측').
- B: Windows spawn 이 부모 전역 RNG 를 소비하지 않음 — SemLock(`synchronize.py:48,122-124`)·AF_PIPE(`connection.py:79,552-556`)·`popen_spawn_win32.py:54` 는 실독 확인했으나 torch multiprocessing 의 pickling/prep-data 경로 전수는 미확인.
- D: plasma 비사설 커밋 +4.33 GB(트레이서 00:09:11→12 commit 30.67→35.19 / sum_priv 4.44→4.63 실측 확인)가 페이지파일 배킹 섹션이라는 매핑 종류 — C++ 미확인(D 자체 표기).
- D: raylet 유휴 워커 소프트 한도 정책(prestart 12 → 유휴 10 유지, 2개 kill 은 raylet.out:214-216·debug_state 105 로 확인) 하에서 num_cpus=2 시 잔존 0~2 — C++ 미확인.
- D: `2.0` float 표기 시 uint32 setattr TypeError — protobuf upb 미실측.
- D: 현 SuperLink 데몬에 `flwr federation simulation-config` 사용 이력 없음 — 실행 중 조회 금지로 미확인(raylet.out:42 CPU 12 로 기본 구성임은 확인).
- R: 재사용 액터의 (k+1)번째 라운드 ↔ 새 프로세스 첫 라운드 등가(cuDNN/cuBLAS/할당자/check_amp) — T-eq 만이 답(R 자체 표기). r001~r003 bit 동일은 본 검토에서 sha256 재확인(0c436ad8…·3b874b53…).
- R: 한 라운드 안의 클라이언트 실행 순서 c0→c1→c2 를 flwr 백엔드가 고정하는지 — 로그 순서 외 소스 근거 없음.
- R: Windows 에서 열린 파일에 `os.replace` PermissionError — 절단 경로 미실측.
- R: SuperLink `state.db`(실측 54.6 MB + WAL 7.5 MB) 에 하드킬 run 이 'running' 으로 잔존할 때 새 run 간섭 없음 — 하트비트 전환 코드 미발견(R 자체 '추측').
- R: 중단 기저율 'GPU 11~18 h 당 1건'(10번 §7-2:204·detection/resume.py 문서) — 파일럿 기록 인용, 재검증 없음.
- B·D·R 공통: 09-05 의 커밋 63.2→74.3→91.1 / 평시 42 GB 구성(10번 §7-1) — sysmon_until 파일 미독, 10번 인용 그대로.
- R: `flwr run --stream` 이 서버 실패에도 rc=0 — `flwr/cli/log.py:111-113, 76-77`·`run.py:238-239` 소스로 성립하고 09-04 rc=1 이 `stage_cell4` SystemExit 에서 난 것과 정합하나 실행 대조는 없음.

### A-4. 세 절의 게이트 질문에 빠져 있던 것 (11건 — §0-3 에 반영)

- 세 변경(B·D 경로 A·R)의 공통 착지 절차와 총 GPU 창: `logs/recover/chain_follow.ps1`(미추적, 정지 훅 없음) 을 누가·어떻게 내리고 확인하는가, 총 창(B T6 + T7=D T5 + R T-smoke-resume + T-eq ≈ 1.5~2 h 추정)을 시드 1 완주(≈13:20) 뒤 어디에 두는가, 게이트가 늦으면 GPU 를 비워 둘지 시드 2 를 R 없이 시작할지.
- 세 변경을 한 커밋·한 FAB 로 착지시키고 시드 경계를 동일하게 둘 것인가(B Q2·R Q3·D Q2 를 따로 답하면 시드 2·3 사이 코드 구성이 갈릴 수 있음). 실험 조건 표에 시드별 커밋·FAB·호스트 구성 열을 두는 것을 승인하는가.
- R 의 회계 감사 기준 강화(seed==derive_seed·budget_fired_at·optimizer_steps 기대값·`restored:` 분류) 자체를 '회계 감사 기준 변경' 으로 승인하는가, 그리고 그 적용 범위를 ④(sep_fed) 한정으로 둘 것인가(스모크·⑦ 회귀 보호).
- R 착지 전 시드 1 ④ 가 죽을 때의 중간 절차: 원장 이동 + `_resume/sep_fed/**` 격리 후 라운드 1 재실행(13.1 h 손실) vs R 게이트 통과까지 GPU 유휴. 지금 `_resume/sep_fed/r005_c1/resume_ep0010.pt` 가 실재하므로 '같은 명령 재기동 전 격리 필수' 를 즉시 지시로 확정할 것인가.
- 원자 로그 스키마 소유: B 의 `val_loader_workers` 지표 행 추가(라운드당 27→30, 스모크·⑦ 에 −1 행)를 스키마 추가로 승인하는가, R 의 기대 집합을 어떻게 버전화할 것인가(시드 1 접두 8 지표 호환).
- `_headroom_wait(need_commit_gb)` 의 값과 위치를 한 번에 결정: B(+41/+55)·D(≥55)·R(재개 preflight 60 하드코딩) 통일, 체인 공통 vs 재개 전용.
- 속도 축 플래그 스키마 통일(이음매 · 외부 GPU 점유 · val_loader_workers · 트레이너 생성 오버헤드 분해)과 요약 평균 제외 규칙, R 이 격리한 열린 라운드 행을 속도 표본으로 쓸지.
- 10번 문서 정정 문안 하나(694ee16 기준: 단가 1.10~1.17·절감 −13.2~14.0(+α)·§7-4 L247·plasma 계상)로 B Q3 와 D Q3 를 합쳐 승인할 것인가.
- B T7 과 D T5 를 같은 run 으로 합치고 '최종 병합 트리의 FAB 해시 == 시드 2 run FAB 해시' 를 착지 조건으로 둘 것인가.
- R 의 k == R(50 라운드 닫힘·audit.json 부재) 경계 동작을 '마감 전용 재개' 로 정의하는가.
- 복구 스크립트(`chain_follow.ps1`·`post_reboot_C.ps1`)를 저장소로 이관하고 R 의 로그 시그니처 변경(`[ROUND i/(R−k)]`)에 맞춰 감시 파서를 누가 고칠지 — D Q6 은 SuperLink 선기동만 묻는다.

## 부록 B. 조립자 정정 목록 (비판 지적 중 인용·수치 오류만; 설계 내용 무변경)

- B §1 spawn 창 최대 76.65@00:22:35 → 77.76@00:22:40 (비판 L15)
- B §6 OOM 재시도 순간 워커 28 → 40/16 추정 (비판 L14)
- D 헤더 HEAD 2cf5a0e → 694ee16 (비판 L11)
- D §2-④ 통신량 인용 client_app.py:97 → round_runner.py:338/client_det.py:85 (비판 L16)
- D §9 build.py:372 → :376 (비판 L16)
- R §3 원자화 대상 파일 목록 추가 + arrayrecord 경로 (비판 L7·L16)
- R §7 T-acct 파일 지정 정정 (비판 L9)
- R §9 T-eq 뷰 구성 미확인 → views-root 경로로 설계 기재 (비판 L10)
- 10번 §7-4·§7-5 정정은 §0-4 (같은 커밋).

## 부록 C. 워크플로 출처

- run: `wf_4ba20718-649` (스크립트 `minispec-b-d-round-resume`, 2026-09-07 00:44~01:55), 이전 시도 `wf_47b257b9-374` 는 재부팅 직전 총괄 지시로 중단(완료 agent 0).
- agent 라벨: design:{B,D,R}:{관점 1, 관점 2} / attack:B:{RNG·계산 불변, 공정성·프로토콜·비교가능성, 필요성·비용편익·구현위험} / attack:D:{flwr 표면 실현성, 효과·부작용} / attack:R:{등가(bit 동일), 원장·감사 둔갑 경로, Flower API 실현성·함정 #1} / synth:{B,D,R} / critic:B+D+R — 전부 `model: 'fable', effort: 'xhigh'`.
- 원본 산출물(JSON): 세션 tasks 출력 `wd5gpl0if.output`, 에이전트별 `journal.jsonl` (로컬, 미추적).
