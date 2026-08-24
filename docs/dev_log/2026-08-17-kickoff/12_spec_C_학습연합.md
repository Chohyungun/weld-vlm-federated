# 트랙 C 미니스펙 — 학습·연합 (detection / vlm / fl)

작성 2026-08-17 · Phase B · 브랜치 `wt/C` · 함정 #1~#4는 다관점 병렬 검증(관점이 다른 설계안
2건을 병렬로 만든 뒤 적대 검증으로 종합)으로 작성했다. 지정 함정 구간이라 심화 검토를 적용했다.

이 문서는 스펙이다. 구현 코드는 없다. A·D 계약이 아직 얼지 않았으므로 §3의 인터페이스는
**전부 가정**이며, 어긋나는 지점을 드러내 놓는 것이 목적이다. 게이트에서 맞춘다.

---

# 0. 범위와 전제

- 소유: `detection/`(분리형 3칸) · `vlm/`(통합형 2칸 + 배경지식 학습) · `fl/`(Flower)
- 소비: 계약 #2 manifest(A) · 계약 #4 공통 예측 스키마(D) · D3/D4 합성 자산(B)
- 실행: Flower 1.x 시뮬레이션(Ray), GPU 1장. 당장은 로컬 RTX 5060 Ti 16GB(sm_120, cu128 이상).
  환경 부트스트랩(uv·Python 3.11·PyTorch cu128)은 A 소유(함정 #5) — C는 그 위에서 실측한다.
- 불변조건 전부 상속. 이 트랙에서 특히: 조기 종료 금지 / last 채점 / 5칸 공통 고정 8종 /
  greedy 1회 / MLflow 로컬 / manifest 단일 진실.

---

# 1. 5칸 학습 경로

## 1-0. 공통 준비 — 배경지식 학습 (칸 아님, 5칸 공통 출발점)

```
Qwen3.5 4B 기본 체크포인트
  ↓ 지시 학습: D3-(b) QA 2만 건, QLoRA r=16/alpha=32, 3 epoch, 조기 종료 없음
  ↓ 추론 학습: D3-(c) 판정 추론 1만 건, QLoRA r=16/alpha=32, 3 epoch, 조기 종료 없음
  ↓ 어댑터 병합(bf16) → 도메인 체크포인트 동결, sha256 기록
```

- CPT 미수행 확정(의사결정로그). 지시·추론 2단만.
- 병합본이 다섯 칸의 공통 출발점이다. 5칸 run 전부가 이 sha256을 MLflow에 로깅해 동일 출발 증명.
- **통합형과 분리형 판정부가 이 체크포인트를 공유한다 — 확정(2026-08-21, Q5 닫힘).**
  Qwen3.5-4B가 네이티브 멀티모달임을 HF 실물 config로 확인했다(`23_모델확정_Qwen35_C.md`).
  통합형은 비전·언어 경로를 함께, 판정부는 같은 체크포인트의 언어 경로만 쓴다. 배경지식
  학습을 **한 번만 수행해 두 구조가 공유**하므로 백본 비동일 교란 변수가 발생하지 않는다.
  chat template(7,756자)도 같은 체크포인트에서 오므로 스냅샷 해시에 포함한다.
- 판정부는 병합 후 **동결**. 분리형 세 칸은 이 동결본을 그대로 쓴다 — 세 칸의 차이는 검출
  가중치에서만 발생한다.

## 1-1. 분리·로컬

| 항목 | 내용 |
|---|---|
| 학습 | YOLO11s, C1/C2/C3 **각자** train split으로 100 epoch (N=100). 시드 3세트 → 9 run |
| 트레이너 | §4-1의 `WeldDetTrainer` — 로컬 칸도 같은 래퍼를 통과시켜 "구조 내 동일" 유지 |
| 산출 | `last.pt` × 3사 × 3시드. best.pt는 생성돼도 채점 경로에서 참조 금지(§2-5 assert) |
| 평가 | 각 last를 글로벌 평가셋 추론 → 검출 JSON → 판정부(공통 동결) → 공통 예측 JSON → 단일 채점기 |
| 보고 | 3사 개별값 + 평균±sd. RQ3의 C3 개별값이 여기서 나온다 |

## 1-2. 분리·중앙

학습 풀 전체(80%, ~56k)로 YOLO11s 1회 100 epoch × 3시드. 나머지는 1-1과 동일.
성능 상한 참조용 — 결과표에 "현실에서는 반출 불가" 각주.

## 1-3. 분리·연합

| 항목 | 내용 |
|---|---|
| 교환 단위 | `model.model.state_dict()` 전체 → ndarray 리스트 (BN buffer 포함, §4-2 정책) |
| 라운드 | **R=50 × E=2 (N=100 등가)** 기본, 파일럿 후 확정. R·E·N·총 갱신 횟수를 논문에 전부 기재 |
| 집계 | 표본 수 가중 FedAvg (C1:C2:C3 ≈ 32k:16k:8k) |
| optimizer | **라운드마다 재생성 — momentum 리셋은 FedAvg 표준 동작**이며 논문에 명시 |
| 체크포인트 | 최종 + 마일스톤 라운드(25/50/75%) 글로벌만. 클라이언트 로컬 상태 저장 안 함 |
| 채점 | 최종 라운드 글로벌 = 이 칸의 last. 시드 3세트 |
| 복구 | 서버가 최신 글로벌 파라미터를 매 라운드 디스크에 남김 → 라운드 단위 재시작 |

## 1-4. 통합·중앙

| 항목 | 내용 |
|---|---|
| 출발점 | 1-0 도메인 체크포인트(동결 병합본), 4bit(NF4) 로드 |
| 학습 | D4 페어(학습 풀 전체) QLoRA r=16/alpha=32, LR 1e-4, 유효 배치 32, **3 epoch (N=3)** |
| 어댑터 | 언어부 all-linear, **비전 인코더 동결·어댑터 제외**(§4-3 명시 제외 목록) |
| 좌표 | 학습 타깃은 모델 네이티브 좌표 — collator에서 변환(§4-4). 데이터 파일은 원본 픽셀 불변 |
| 산출 | 최종 epoch 어댑터 → 병합(§4-3 정책) → 평가용 체크포인트 |
| 평가 | vLLM, greedy(온도 0), 최대 토큰 고정, **재시도·재프롬프트 없음.** 파싱 실패는 parse_ok=false |
| 시드 | 1시드 + 클라이언트별 분산 보고(비용 사유 명시) |

## 1-5. 통합·연합

| 항목 | 내용 |
|---|---|
| 교환 단위 | `get_peft_model_state_dict()` 결과(LoRA A·B)만. 기본 모델은 사전 배포 동일 체크포인트 |
| 라운드 | **R=6 × E=0.5 epoch (N=3 등가).** E=0.5는 로컬 스텝 수로 환산(§6 Q6) |
| 집계 | 행렬별 가중 평균 avg(A)·avg(B), 가중 n_k = 클라이언트별 D4 학습 페어 수 — **단일 소스 = B의 `counts.json`**(게이트 #6). **avg(B)·avg(A) ≠ avg(B·A) 논문 명시**(§4-3) |
| 병합 | 라운드 중 금지. **최종 라운드 후 1회** 병합 → 평가용 체크포인트 |
| 동시성 | 클라이언트 순차 실행(동시성 1) — 클라이언트 하나가 GPU를 다 쓴다 |
| 시드 | 1시드 |

## 1-6. RIAWELC 백업 시나리오 (8/25 미승인 시)

검출 3칸이 분류 3칸(YOLO11s-cls)으로 바뀐다. FL 구조·래퍼·집계·통신량 계측은 동일하게
성립하고 bbox 계열 경로만 비활성(§3 가정 2의 null 전파). 위치 지표 제외는 논문 명시(D 소관).

---

# 2. 모듈 명세

## 2-1. 디렉터리·모듈

```
detection/
  dataset_view.py   # 계약 #2 소비 → 칸별 Ultralytics 뷰(data.yaml + 파일 리스트) 생성.
                    # **A 제공 로더 함수 경유만 — 자체 조인 금지**(게이트#4 Q1 조건 2).
                    # 원본 폴더 직접 읽기 금지 원칙의 구현 지점. split=='eval' 유입 즉시 실패
  fed_trainer.py    # FedDetectionTrainer(DetectionTrainer 서브클래스) — §4-1 접촉점 6곳
  round_runner.py   # train_round(...) — 검출 3칸 공통 유일 진입점 (무상태, §4-1)
  serialize.py      # 정본 키 순서·ndarray 직렬화·sha256 (§4-1 교환 규약)
  budget_audit.py   # 조기 종료 부재 사후 증명 + epoch/step 회계 (§2-5)
  train_cell.py     # 분리·로컬/중앙 진입점 — train_round를 R=1·E=100으로 호출
  export_preds.py   # 최종 raw state_dict → 글로벌 평가셋 추론 → 검출 JSON. 체크포인트 assert
vlm/
  coords.py         # bbox 좌표 변환·역변환 단일 모듈. §4-4 처방. D 트랙과 공유(§3 가정 9)
  pair_dataset.py   # D4 jsonl → 학습 샘플. collator에서 네이티브 좌표 변환·동적 패딩
  train_sft.py      # 배경지식(지시·추론)·본학습 공용 SFT 진입점 (TRL SFTTrainer + PEFT + bnb)
  merge_adapter.py  # 어댑터 병합 → 평가용 체크포인트. §4-3 병합 정책
  infer_eval.py     # 평가셋 추론(vLLM, greedy 1회) → 공통 예측 JSON. 역변환 호출 지점
fl/
  server_app.py     # Flower ServerApp. 전략 구성, evaluate 콜백(MLflow round=step), 체크포인트
  client_det.py     # ClientApp(검출). WeldDetTrainer.train_round 호출
  client_vlm.py     # ClientApp(VLM). 어댑터 로드→학습→어댑터 반환
  aggregate.py      # 표본수 가중 FedAvg(BN 정책 §4-2 포함) + LoRA 행렬별 평균(§4-3)
  comms.py          # 직렬화 바이트·norm·shape/dtype 시그니처 로깅 (§2-6)
  checkpoint.py     # 글로벌 파라미터 저장(마일스톤 규칙), 라운드 재시작 복구
configs/
  base.yaml + 칸별 override 5개 + gpu 프로파일 2벌(16GB/48GB — A가 뼈대, C가 학습 값 채움)
```

**정식 산출물 2종 (게이트 #6):**
- `consumed_image_ids.txt` — 모든 학습 러너(검출·VLM·FL 클라이언트)가 실제 소비한 image_id
  목록을 run별로 남긴다. D의 누출 해시 검사(학습 소비 ∩ eval split = ∅)의 입력 (결정 F).
- **클라이언트 × 라운드 회계 매트릭스**(`accounting.csv`) — budget_audit이 생성. §2-2 참조.
  **머지 차단 조건** (결정 A).

- 확정 하이퍼파라미터는 `configs/`에만 둔다. 이 문서의 수치는 파일럿 전 초기값이다.
- 시드: random/numpy/torch/cuda 전역 고정 + `cudnn.deterministic=True`. 시드 값은 config에.

## 2-2. Flower 실행 구조

| 항목 | 명세 |
|---|---|
| 스캐폴드 | `flwr new` 구조. **검출용·VLM용 app 2벌** (`fl/server_app.py`가 cell 인자로 분기, ClientApp은 별도 파일) |
| run-config | `pyproject.toml` [tool.flwr.app.config]: `num-server-rounds`(R), `local-epochs`(검출 E) / `local-steps`(VLM), `batch-size`, `seed`, `cell`. 실행 시 `flwr run --run-config`로 override |
| 백엔드 | Ray 시뮬레이션. 검출 `num_gpus=1.0`(순차 — 16GB에서 3동시는 위험, 48GB 프로파일에서 0.33 재검토), VLM `num_gpus=1.0` 고정(동시성 1) |
| 가중 | fit 반환 `num_examples` = 클라이언트 train 표본 수 → FedAvg 가중 |
| momentum | optimizer는 라운드마다 재생성. **momentum 리셋 = FedAvg 표준 동작** — 논문 명시 |
| 라운드 모니터링 | **[게이트 #6 결정 B·E 확정, 정본 구현설계 7-4 정정 완료]** 라운드별 곡선은 **val**(학습 풀의 10%, 묶음 단위·층화, **로깅 전용** — A의 Q3 자산)로 낸다. 서버 evaluate 콜백은 val 기준으로 동작하고 MLflow `round`=step(클라이언트 로깅 없음). **글로벌 평가셋은 전 실험을 통틀어 최종 채점 1회만 접근한다.** 불변식은 "접근 0회"가 아니라 "평가셋이 학습·선택에 영향을 주지 않는다" — 조기 종료 금지·last 채점이라 라운드별 val 곡선에 선택압이 없다 |
| 실패 정책 | **[게이트 #6 결정 A 확정] `accept_failures=False` 강제** + `min_fit_clients = min_available_clients = 3` (검출·VLM 양쪽 전략). 기본값 `True`는 클라이언트 실패 시 나머지로 집계해 실효 학습량 감소가 지표에 흔적 없이 사라진다 — R×E=N을 조용히 깨는 경로. 실패 시 라운드 중단·사람 개입 |
| 회계 매트릭스 | **클라이언트 × 라운드 회계 매트릭스를 정식 산출물로 남긴다(머지 차단 조건).** 매 라운드 각 클라이언트의 참여 여부·로컬 스텝 수·표본 수 기록, 종료 시 R×E=N 실측 검증(R×3 전 셀 epochs_ran==E, VLM은 steps_ran==S_k, **빈 셀 = run 무효**). 논문의 총 갱신 횟수는 이 실측치 |
| 시뮬레이션 명시 | 물리 분산 아님. 통신량은 교환 객체 직렬화 바이트 실측 × 라운드로 산출 — 논문 명시 |

## 2-3. 검출 칸 공통 학습 설정 (초기값, 파일럿 후 configs 확정)

100 epoch(중앙·로컬) 또는 R=50×E=2(연합), batch 32, SGD(momentum 0.937), AMP, cosine LR,
입력 640 letterbox. 증강은 세 칸 동일 고정 — 값은 §6 Q8. patience는 §2-5.

## 2-4. VLM 칸 공통 설정 (초기값)

QLoRA 4bit(NF4) + bf16 연산(Blackwell), r=16/alpha=32, LR 1e-4, 유효 배치 32(grad accum),
max_seq_length 2,048(판정문), 초과 샘플 폐기 후 건수 보고, 패킹 미사용(노출 횟수 등가 단순화),
grad checkpointing 켬, 비전 인코더 동결. 프롬프트·출력 스키마는 파일로 고정하고 스냅샷
해시에 포함 — 두 칸 사이 한 글자도 다르면 안 된다.
**리사이즈 상·하한은 `base.yaml` 공통 고정, 프로파일 override 금지** — 게이트 웨이브 #5에서
5칸 공통 고정 항목에 정식 편입(Q15). 프로파일이 리사이즈 치수를 바꾸면 좌표 공간이 조용히
드리프트한다.
**[2026-08-21 키 이름 정정]** Qwen3.5-4B의 프로세서는 `min_pixels`/`max_pixels`가 아니라
**`size: {shortest_edge, longest_edge}`** 형식이다(실측: 65536 / 16777216). Qwen2.5-VL만
`min/max_pixels`를 쓴다. Q15의 취지는 그대로이되 **고정 대상 키가 `size.shortest_edge`·
`size.longest_edge`** 이며, 프로파일 yaml에는 이 키를 두지 않는다(어길 수단 제거).
D 트랙에 통지 대상 — 계약 #4 §3의 Q15 서술도 같은 정정이 필요하다.

## 2-5. 조기 종료 완전 제거와 사후 검증

**끄는 방법**
- Ultralytics: `patience`를 epoch 수보다 큰 값(예: 10000)으로 명시 설정. `patience=0`이
  내부적으로 비활성 처리되는지는 버전 의존이라 **믿지 않는다** — 명시적 큰 값이 안전 기본값.
  configs에 주석: `# 조기 종료 금지 — R×E=N 학습량 등가가 논문의 공정성 주장 (docs/개발규약.md 모델·실험 정책 1)`
- TRL SFTTrainer: EarlyStoppingCallback을 넣지 않는다(기본 없음). `load_best_model_at_end=False`,
  `metric_for_best_model` 미설정. val loss는 로깅만.
- best 선택 금지: best.pt 생성 자체는 두되(스톡 동작 보존), **채점 경로가 last만 참조**하도록
  `export_preds.py`·`merge_adapter.py`에 체크포인트 파일명/스텝 assert를 박는다.

**사후 확인 (모든 run에 자동 적용, 실패 시 run 무효)**
1. 검출: `results.csv` 행 수 == 설정 epoch 수 (조기 종료가 걸렸다면 행이 모자란다).
2. 검출: 학습 로그에 EarlyStopping 발동 문구 부재를 grep으로 확인.
3. 검출: last.pt 메타(내장 epoch 값) == N−1.
4. FL: **회계 매트릭스 전 셀 검증** — R×3 전 셀에 대해 epochs_ran==E(검출) /
   steps_ran==S_k(VLM), **셀 부재 자체를 실패로 정의**한다(실패 클라이언트가 로그에 아예
   없는 경우를 잡는다 — 게이트 #6 결정 A).
5. VLM: `trainer_state.json`의 `global_step` == 사전 계산한 예정 총 스텝 수(§6 Q6 공식).
6. 채점 직전: 채점 대상 체크포인트 경로가 `last` 규칙과 일치하는지 assert.

이 6개를 `check_no_early_stop` 스크립트 하나로 묶어 MLflow에 pass/fail 로깅한다.

## 2-6. 통신량·norm 계측

- `fl/comms.py`: 라운드마다 업링크(클라이언트→서버 fit 반환)·다운링크(서버→클라이언트 배포)
  각각 `sum(ndarray.nbytes)` 실측 → MLflow (`round`=step).
  총 통신량 = Σ_라운드 Σ_클라이언트 (업 + 다운). 결과표의 "검출 전체 가중치 vs 어댑터" 대비가
  여기서 나온다 (YOLO11s fp32 ≈ 수십 MB/라운드·클라이언트, LoRA r=16 언어부 ≈ 수십 MB —
  실측으로 채움, 직렬화 dtype은 §4-3 확정값).
- **와이어 포맷 fp32 확정**(§4-1·§4-3 — Flower 직렬화가 bf16 불가, fp16 왕복은 집계 정밀도
  드리프트). 통신량 표는 fp32 실측 바이트로 채우고, fp16 배포 절감 가능성은 각주로만.
- norm 안전장치: 집계 **직전** 클라이언트별 전체 L2 norm + 집계 **직후** 글로벌 L2 norm 로깅.
- assert 3종(위반 시 즉시 중단): ① 키 리스트 동일성(순서 포함) ② shape 리스트 동일성
  ③ dtype 동일성. 순서 뒤섞임·dtype 불일치가 집계를 조용히 오염시키는 것을 여기서 잡는다.

---

# 3. A·D에 가정한 인터페이스 (게이트에서 맞출 것)

> **갱신 (게이트 웨이브 #5, 2026-08-17):** 계약 4종 동결. 아래 표의 가정 1·9·10은 게이트
> 판정으로 **확정**됐고(01_설계검토 §5-1·§5-3·게이트#4 Q1), 이후 변경은 총괄 승인 사항이다.

| # | 상대 | 가정 | 어긋나면 |
|---|---|---|---|
| 1 | A | **확정(게이트#4 Q1):** 계약 #2는 `manifest.csv`(이미지 단위) + `annotations.csv`(결함 인스턴스 단위) **2파일, 단일 `SNAPSHOT.sha256`으로 함께 잠금.** 조인은 **A 제공 로더 함수로만** — C의 `dataset_view.py`는 자체 조인 금지, A 로더 경유 | (동결 — 변경은 총괄 승인) |
| 2 | A | RIAWELC 흡수 시 bbox 계열 컬럼은 **null**이며, null 의미는 "위치 라벨 없음"(0개 결함과 구분됨 — 정상은 label_orig로 구분) | null 의미론이 다르면 dataset_view의 필터 규칙 수정 |
| 3 | A | 학습 풀 내에 클라이언트별 **train/val 서브분할이 존재**한다 (val은 곡선 로깅·서버 evaluate 전용, 채점은 eval만) | val이 없으면 C가 train에서 group_id 단위로 재현 가능하게 떼는 유틸 추가 — 단 분할 주체는 A여야 한다고 본다 |
| 4 | A | **확정(게이트 #6 결정 F):** 두께·스케일 가용성은 **스냅샷별 `data_capabilities.yaml`**(A 구현 채택 — mock 2벌에 실물 존재). C는 이 파일을 읽는다. 종전 가정(`configs/data_flags.yaml`) 폐기 | (동결) |
| 5 | A | `configs/label_map.yaml`에 YOLO 클래스 인덱스 ↔ 원본 라벨 ↔ ISO 코드 사상 포함. 검출 학습 클래스는 결함 4종(정상 = 박스 없음) | 인덱스 사상이 없으면 추가 요청 — 라벨 문자열 하드코딩 금지라 필수 |
| 6 | A | 폴리곤→bbox·COCO/YOLO 변환은 A(`data/convert`) 소유. C의 `dataset_view.py`는 **manifest 필터링으로 칸별 뷰(data.yaml + 파일 리스트)만 생성** | A가 뷰까지 만들면 dataset_view는 검증만 수행 |
| 7 | B | D4 페어 jsonl: `{image_id, image_path, client, split, skeleton{defects[{type, bbox_px, size_*}], verdict, clauses[]}, target_text}` + `SNAPSHOT.sha256`. **bbox는 원본 픽셀** — 네이티브 변환은 C collator 수행 | 필드명 조정만. bbox가 원본 픽셀이 아니면 반려 요청(불변조건 8) |
| 7b | B | **확정(게이트 #6 결정 F):** 클라이언트별 D4 학습 페어 수의 단일 소스는 B의 **`counts.json`**(정식 산출물). C의 통합·연합 n_k와 §6 Q6 스텝 환산이 이 파일을 읽는다 | (동결) |
| 8 | B | D3 corpus jsonl은 messages 또는 prompt/response 형식 — TRL 표준 로더로 흡수 가능 | 변환 어댑터 1개 추가 |
| 9 | D | **확정(§5-1 + 게이트 #6 어휘 통일):** 계약 #4는 `bbox_px`(역변환 완료 원본 픽셀) + `coord_space`·`coord_cfg_hash` 메타. **네이티브 좌표 필드 없음.** C 추가 요청 필드 `cell`·`seed`는 D 스키마에 이미 존재 확인, `raw_text`는 D의 `raw_output_ref`(파일:줄 참조) 방식 수용 | D의 §5-1 반영 재보고로 동결 발효 |
| 10 | D | **확정(§5-1): C = 규약 소유자, D = 검증 소유자.** `vlm/coords.py`는 C 소유·**의존성 제로 리프 모듈 준수 의무**·D가 import. D의 검증 장치 전부 유지: 골든 픽스처 독립 재실행 + Q20 실측 봉인(Ultralytics xyxy 원본 좌표 가정, 표본 50장) + D1 정규화 잔재 검사 + 해시 대조 | (동결 — 변경은 총괄 승인) |
| 11 | D | 분리형 검출→판정부 인터페이스: C가 `{image_id, dets[{cls_orig, iso_code, bbox_px, conf, major_axis_px, equiv_diam_px, size_mm?}]}` JSON을 내면 D의 rag/판정부가 소비 | 필드 협상. conf 임계는 §6 Q4 — 5칸 공통 고정 |
| 12 | D | MLflow 규약(D 소유): experiment `weld-fl`, run 명명 `{cell}_{seed}_{yymmdd}`, FL은 round=step. C는 이 규약대로 로깅만 | 규약 확정본을 따라감 |
| 13 | D | latency·VRAM 측정 스크립트는 D 소유. C는 추론 진입점(`export_preds.py`, `infer_eval.py`)을 batch=1 호출 가능하게 제공 | — |

---

# 4. 함정 4건 — 처방과 검증

> 다관점 병렬 검증으로 수행했다. 함정마다 관점이 다른 설계안 2건을 병렬로 만들고, 적대
> 검증 담당이 반박과 웹 출처 재확인을 거쳐 단일 처방으로 종합했다(분업 단위 12개,
> run `wf_e5b3bb68-696`). 아래는 요약이며 **전문(처방·검증·실패 모드·논문
> 문장·출처·기각 사유)은 `12a_함정구간_판정_C.md`** 에 있다.

## 4-1. 함정 #1 — Ultralytics × Flower 커스텀 트레이너

**채택: `FedDetectionTrainer`(DetectionTrainer 서브클래스) + stock resume 경로 재활용.**
warmup·cosine LR·close_mosaic이 전역 epoch의 순수 함수임을 소스로 확인 —
`start_epoch = r×E` + `scheduler.last_epoch = start_epoch−1` 오프셋만으로 중앙 칸과 LR
궤적이 동일해진다(V2 통과 조건: 100개 epoch 값 일치, 허용 오차 1e-12 — 검증 통과 전에는
단정하지 않는다). 공개 API 반복 호출안(관점 A)은 warmup 제거·스케줄 근사라는
레시피 변경을 유발해 기각하되 안전장치(정본 키 sha256, 주입 증빙, 파생 시드)는 흡수.

- **3칸 공통 유일 진입점** `train_round(weights_in, round_idx, E, base_seed, client_idx,
  data_yaml, cfg)` — 무상태. **로컬·중앙 칸은 R=1(E=100) 퇴화 케이스**로 같은 함수를
  통과시켜 루프 동일성이 문자 그대로 성립.
- 내부 접촉점 6곳으로 한정: `_setup_train` 후 strict 주입 + epoch·스케줄러 오프셋 +
  close_mosaic 재적용 + EMA off + stopper 스텁(`possible_stop=False` 필수),
  `validate`/`final_eval`/`save_model` no-op. 콜백은 인스턴스 로컬만(전역 dict 수정 금지).
- 예산 정지: `on_fit_epoch_end`에서 로컬 epoch E 도달 시 `trainer.stop=True` — 조기 종료가
  아니라 예산 도달 정지이며 스텁 stopper 이력과 구분 로깅.
- **EMA: 검출 3칸 공통 비활성. raw fp32 가중치를 교환·집계·채점.** stock last.pt(EMA
  저장본)는 어느 칸에서도 미사용 — "last 채점" = 래퍼·서버가 저장한 최종 raw state_dict
  (선택 절차 전무). 라운드마다 리셋되는 EMA와 중앙 100ep EMA는 평활 지평이 달라 등가가
  깨지므로 비활성이 유일한 대칭 해법.
  **[게이트 웨이브 #5 확인 요청에 대한 답 — 확정]** FedAvg가 평균하는 것은 **raw weight다.
  EMA가 아니다.** EMA 평균안을 기각하는 근거: ① 클라이언트별 EMA는 각자의 로컬 궤적에 대한
  지수 평활 상태라, 이를 평균하면 서로 다른 평활(모멘텀) 상태가 섞여 다음 라운드 학습의
  출발점이 어떤 칸과도 등가가 아니게 된다. ② E=2 epoch 단위로 리셋되는 EMA(50회)와 중앙
  100 epoch 연속 EMA는 평활 지평이 구조적으로 달라, EMA를 쓰는 순간 연합↔중앙 등가 주장이
  깨진다. ③ raw 교환·EMA off는 세 검출 칸에 동일 적용되므로 5칸 공통 고정 유지. 이 사실
  (Ultralytics last.pt/best.pt = EMA 가중치)과 집계 대상 선택은 **논문에 명시한다.**
- 교환 규약: 정본 키 리스트 sha256 고정 + shape/dtype 라운드별 assert, **wire fp32**,
  `num_batches_tracked`는 가중 평균 금지·클라이언트 간 max pass-through.
- 파생 시드 `seed = base_seed + 10007·r + 101·client` — 고정 시드면 50라운드가 같은 셔플을
  반복하는 숨은 비대칭 발생(init_seeds가 트레이너 init마다 호출됨을 소스 확인).
- `ultralytics==8.4.120` 정확 핀 + 계약 테스트. 진짜 방어선은 행동 등가 테스트다.
- **평가 접근 정책 [게이트 #6 결정 B로 통일]:** 라운드별 곡선은 **val**(학습 풀 10%, 로깅
  전용) 기준 서버 evaluate로 산출. **글로벌 평가셋은 전 실험 최종 채점 1회만 접근.**
  train loss·norm 로깅은 별도 유지. (함정 #1 판정의 "접근 0회" 문구는 이 기준으로 대체 —
  부록 12a 정정 고지 참조.)
- **숨은 프레임워크 기본값 사냥 결과 2건** — Ultralytics `final_eval()`의 best.pt 로드
  (no-op 처방), Flower `accept_failures=True`(§2-2 실패 정책으로 차단). 구현 중 발견되는
  같은 종류의 기본값은 즉시 총괄 보고 목록에 올린다.

**검증(요점):** V1 래퍼 R=1×E=100 vs stock 100ep 행동 등가(지표 오차 범위) /
V2 LR 100개 값 완전 일치(이진 판정) / V3 R=50×E=2 연결 시 전역 LR·mosaic 종료 시점 일치 /
조기 종료 부재 사후 회계(§2-5) / 주입 증빙 다이제스트 라운드별 대조. 전문은 부록 12a.

## 4-2. 함정 #2 — BatchNorm 연합 평균

**채택: BN buffer(running_mean/var) 포함 전체 표본수 가중 FedAvg 확정.**
`num_batches_tracked`는 평균 대상에서 제외하고 서버가 element-wise max로 결정론적 덮어쓰기.

- 메인 런은 **관측 전용 진단**만: 누락 분산비 ρ(가중 평균 분산이 놓치는 클라이언트 간 평균
  차이 항), 클라이언트 간 buffer 거리, 재질별(ST/AL) 서브셋 분해, 집계 전후 sanity.
  어떤 신호도 런 중 동작을 바꾸지 않는다. **라운드별 진단·서브셋 분해는 val 기준**(게이트
  #6 결정 B — 12a 함정 #2 열린질문 6의 "글로벌 평가셋 evaluate 확장" 기본값은 이 기준으로
  대체), 글로벌 평가셋 기준 분해는 최종 채점 1회에서만.
- **사전 등록 트리거**(T1 회복률<80% / T2 AL 서브셋 선택적 격차 / T3 진단 곡선 임계) 발동
  시에만 부록 실험 가동: (a) buffer-swap 절제 — 닫힌형 풀링으로 원인 판별,
  (b) **클라이언트 협조 BN 재보정** — 각 클라이언트가 자기 train 분할로 forward-only 모멘트
  수집 → 서버가 표본수 비례 결합. gradient 없음 → 학습량 등가 불훼손.
- **구현설계 3-3의 "글로벌 평가셋의 train 부분으로 BN 재추정"은 평가 자산 격리(docs/개발규약.md
  데이터 규칙 4) 위반으로 명시 기각** — (b)로 교체를 게이트에 제안. 승인 전까지도 평가셋은
  어떤 보정·통계 갱신에도 무접촉.
- 균등 결합 주장은 사실 오류로 격추 — 층화 평가셋은 모집단 비율(강재:알루미늄≈6:1)을
  보존하므로 기본 가중은 표본수 비례(32:16:8). 균등은 부록 변형 행으로 강등.
- **주 결과표는 어떤 경우에도 보정 전 last다.**

**검증(요점):** 진단 로깅의 라운드 곡선 확인, 트리거 수치의 파일럿 후 고정(메인 런 시작 후
변경 금지), 부록 재보정의 클라이언트 경계 준수 격리 증명. 전문은 부록 12a.

**[2026-08-21 추가 — 통합형 FL에도 buffer 검증이 필요해졌다]** 위 처방은 검출부
BatchNorm 대상이고, 통합형(VLM)은 RMSNorm 계열이라 buffer-free로 가정해 왔다. 그런데
Qwen3.5-4B 언어부가 **SSM/linear attention 하이브리드**로 확인돼(§4-3), conv state 등
persistent buffer가 존재할 수 있다. 12a 함정 #2의 **V10(`named_buffers()` 전수 검증)을
통합·연합 착수 전 필수 게이트로 승격**한다. buffer가 발견되면 어댑터만 교환하는 설계에서
그 buffer가 어떻게 처리되는지(로컬 잔존 여부)를 명시해야 한다.

## 4-3. 함정 #3 — LoRA 어댑터 교환·집계

**채택: 행렬별 표본수 가중 평균(FedIT 패턴) + 서버 1회 어댑터 초기화·배포 + 키드 교환
(wire fp32) + 최종 1회 bf16 비양자화 병합본 단일 평가 경로.**

- Flower 직렬화 소스 확인 결과 bf16 와이어 불가(`.numpy()` 직행) → **wire fp32 확정**.
  dtype 체인: 학습 bf16 → 직렬화 fp32 → 집계 fp32 → 재배포 캐스트.
- **`target_modules="all-linear"` 금지** — 명시 target 목록으로 확정(채택 시점
  `named_modules()` 전수 덤프로 비전 모듈 `visual.`·merger 제외 검증). `modules_to_save`
  (embed/lm_head) 금지 — 교환량 급증.
  **[2026-08-21 근거 강화]** Qwen3.5-4B 언어부는 순수 트랜스포머가 아니라 **SSM/linear
  attention 하이브리드**다(`text_config`에 `linear_conv_kernel_dim`·`linear_num_key_heads`·
  `mamba_ssm_dtype`·`mtp_num_hidden_layers`). `all-linear`를 쓰면 SSM 프로젝션·conv 계층까지
  어댑터가 붙어 집계 거동을 예측할 수 없다. 명시 목록 확정이 선택이 아니라 필수가 됐다.
- `avg(B)·avg(A) ≠ avg(B·A)` 논문 명시. "E=0.5라 오차가 작다"는 선험 논거는 기각(라운드당
  수백 스텝) — **집계 오차 실측을 부록 1단 '필수'로 승격**, 이를 위해 마일스톤 라운드의
  클라이언트 어댑터 페이로드를 보존. FLoRA 재학습(부록 2단)은 G3 이후 여유 시에만.
- **vLLM 어댑터 동적 로드 경로 확정 기각**(멀티모달 LoRA 언어부 한정·동적 로드 버그 보고)
  — 평가는 병합본으로만. 병합은 4bit 베이스가 아니라 **bf16 비양자화 사본**에 수행.
- **병합·평가 체크포인트 생성 절차를 5칸 공통 고정 9번째 항목으로 승격 제안** — 통합형 두
  칸과 배경지식 단계가 `merge_final` 동일 함수를 쓰도록 강제.
- FedAvg 가중 n_k = 클라이언트별 D4 학습 페어 수. **단일 소스는 B의 `counts.json`(게이트
  #6 결정 F로 정식 산출물화) — manifest 파생 가정 폐기.**
- VLM 라운드 곡선은 **val loss**(teacher-forcing, 생성 없음) 기준 서버 evaluate(게이트 #6
  결정 B). 생성 기반 지표는 최종 채점 1회 — 곡선이 더 필요하면 마일스톤 어댑터 사후 산출.
  글로벌 평가셋 접근은 최종 1회. (12a 함정 #3의 "분리·연합에만 라운드별 평가 유지" 문구는
  이 기준으로 대체.)

**검증(요점):** 키 순서·shape·dtype fail-fast assert 군, 집계 전후 norm 로깅, 병합 왕복
등가 테스트(병합본 vs 베이스+어댑터 로드 출력 대조), 집계 오차 실측(부록 1단). 전문 12a.

## 4-4. 함정 #4 — VLM bbox 좌표계

**채택: 기본 규약 NORM_1000(분모 1000 고정) + 플레인 JSON `bbox_2d` 타깃. 선언은 코드가
아니라 configs 데이터(CoordCfg). 합성 도형 카나리아 argmax 판별이 통합형 학습 착수의
하드 게이트 — 문서보다 실측 우선.** 판별 결과가 선언과 다르면 CoordCfg의 `coord_space`
값만 교체(Qwen3.5의 규약이 공개 자료에 미기재이므로 이 구조가 필수다).
어휘는 **`coord_space`로 통일**(게이트 #6 결정 F — 종전 표기 `convention` 폐기).

- **저장 계층은 원본 픽셀 단일.** 모델 좌표는 어떤 파일에도 저장 금지. 변환은 C 소유
  의존성 제로 리프 모듈 `vlm/coords.py`의 정변환·역변환 2함수로만 통과.
- ABS_RESIZED 판별 시 리사이즈 치수는 **재계산 금지** — smart_resize를 coords.py에 자체
  구현도 import도 하지 않고, 프로세서 실호출 산출물(`image_grid_thw` × patch_size 복원값)을
  인자로만 받는다. 정합 대조는 별도 파일럿 스크립트로 분리.
- **D 트랙은 같은 모듈을 import하고 C의 골든 픽스처를 재실행** — 채점기 자체는 bbox_px만
  받는 좌표계 무지 상태 유지(자체 재구현 금지 명분).
- 카나리아는 5칸 평가가 아니라 사전 점검이므로 재시도 금지 원칙(greedy 1회)과 무충돌 —
  프로브 프롬프트는 본 실험 프롬프트와 별도 산출물(파일 고정 + 해시).
- 예측 JSON에는 역변환 완료된 원본 픽셀 `bbox_px`만 기록 + 메타(`coord_space`,
  `coord_cfg_hash`) — §3 가정 9에 메타 필드 허용 추가.
- RIAWELC 백업 시 bbox 계열 전 경로 비활성(null 전파, §1-6).

**검증(요점):** 라운드트립 속성 테스트 IoU(원본, 왕복) ≥ 0.99(극단 종횡비·경계 픽스처 포함)
/ 골든 픽스처 회귀 / 카나리아-1: 파인튜닝 전 제로샷 grounding 소표본 argmax 판별(IoU
0.0x대 = 규약 가정 오류 신호, 학습 착수 차단) / 학습 첫 검증 IoU 분포 감시. 전문 12a.

---

# 5. 16GB 실측 계획 (RTX 5060 Ti 16GB, sm_120)

## 5-1. 방법

- 전제: A의 환경 부트스트랩(cu128, `torch.cuda.is_available()` 확인) 완료 후 착수.
- 공통 프로토콜: 칸별 대표 구성으로 **100 step 파일럿** → `torch.cuda.max_memory_allocated`,
  step 시간 기록 → GPU-일 추정치 갱신(총괄 리스크 등록부의 18~27 GPU-일 재추정 입력).
- OOM 시 조정 사다리 **[게이트 #6 결정 C로 재작성 — `max_pixels`는 어떤 단계에서도 손대지
  않는다(Q15 동결)]**:
  검출 = 배치 크기↓ (해상도 640 유지) /
  VLM = 배치 크기↓ → gradient accumulation↑(유효 배치 유지) → gradient checkpointing
  (기본 켬 — §2-4) → **(최후) 모델 크기** 재검토(총괄 안건). 그래도 안 되면 48GB 이관.
  **구조적 방어: 프로파일 yaml에 `min/max_pixels` 키 자체를 두지 않는다** — base.yaml에만
  존재해 override가 물리적으로 불가능하게 한다.
- **유효 배치·해상도 등 "5칸 공통 고정" 항목은 칸별로 다르게 조정할 수 없다** — 한 칸이
  낮추면 전 칸이 같이 낮춘다. 이것이 16GB 실측을 먼저 하는 이유다.

## 5-2. 칸별 예상과 병목

| 구간 | 16GB 예상 | 병목 후보 | 안 되면 |
|---|---|---|---|
| 분리·로컬/중앙 (YOLO11s 640 b32 AMP) | **여유** (6~8GB 추정) | 없음 예상 | — |
| 분리·연합 (+Ray) | 단일 클라이언트와 동일 peak (순차 실행) | **라운드 간 GPU 메모리 미해제**(Ray actor 수명) — 실측 필수 | actor 라운드마다 종료·재생성 |
| 배경지식 학습 (4B QLoRA, seq 1k~2k) | 가능 추정 | activation | accum↑ |
| 통합·중앙 (4B QLoRA + 비전) | **경계** (10~15GB 추정: 4bit 가중 ~3GB + activation) | **비전 토큰 수**(max_pixels는 동결 — 조정 불가), grad ckpt 필수. flash-attn: sm_120/Windows 휠 부재 실측 확인 → SDPA 폴백 확정 | §5-1 사다리(배치→accum→ckpt) 소진 후에도 OOM이면 48GB 이관. 병목 = activation임을 기록 |
| 통합·연합 | 클라이언트 순차라 peak는 중앙과 동일 + 병합은 CPU RAM | 라운드 반복의 로드/언로드 시간(성능 아님) | — |
| 평가 추론 (vLLM 4B bf16 + 어댑터) | 가능 (8~10GB + KV) | `gpu_memory_utilization` 조정 | 배치 크기↓ |

## 5-3. 산출물

`configs/gpu_16gb.yaml`·`configs/gpu_48gb.yaml`에 실측 확정값 기입(값은 configs에만, 문서는
링크), 실측 로그는 MLflow. 5칸 각각 "16GB에서 도는가 / 무엇을 줄였는가 / 병목이 무엇인가"
한 줄 표를 파일럿 보고로 총괄에 제출한다.

---

# 6. 열린 질문 (전부 안전 기본값 — 총괄 답변 전까지 이 값으로 진행)

다관점 병렬 검증 판정으로 Q1~Q3은 기본값이 **확정안으로 승격**됐다(게이트 추인 대기). 함정별 세부
열린 질문 **27건**(전부 기본값 부여)은 부록 12a의 함정별 목록에 있다 — 게이트 안건.

| # | 질문 | 안전 기본값 |
|---|---|---|
| Q1 | 검출 FL에서 EMA 가중치를 집계 대상으로 하는가 | **확정(§4-1):** 검출 3칸 EMA 공통 비활성, raw fp32 가중치를 교환·집계·채점. stock last.pt 미사용 — "last 채점" = 최종 epoch/라운드 raw state_dict(선택 절차 전무). 게이트 추인 요청 |
| Q2 | LR 스케줄을 라운드 사이에 이어가는가(글로벌 epoch 오프셋) | **확정(§4-1):** 전역 cosine(N=100) 단일 스케줄을 `start_epoch=r×E` resume 오프셋으로 연속 — 중앙 칸과 LR 궤적 동일(V2 이진 검증) |
| Q3 | FL 라운드 모니터링의 평가 데이터 | **닫힘(게이트 #6 결정 B·E, 정본 7-4 정정 완료):** 라운드별 곡선 = **val**(학습 풀 10%, 묶음 단위·층화, 로깅 전용). 글로벌 평가셋 = 최종 채점 1회. 구현설계 3-3(BN 재추정) 정정 제안은 §4-2대로 유지(별도 안건) |
| Q4 | 검출 conf 임계(판정부 입력 필터) | 0.25 고정, **5칸 공통**, 파일럿 후 확정하되 칸 간 차등 금지 |
| Q5 | Qwen3.5 4B 네이티브 멀티모달 여부(통합형·판정부 체크포인트 공유) | **닫힘 (2026-08-21, HF 실물 config 확인 — `23_모델확정_Qwen35_C.md`).** 네이티브 멀티모달 확정(`Qwen3_5ForConditionalGeneration` + `vision_config` + 비전 토큰 4종 + `Qwen3VLProcessor`). **통합형·판정부가 동일 체크포인트를 공유하며 백본 비동일 교란 변수는 발생하지 않는다.** 배포 리포는 `Qwen/Qwen3.5-4B` 단일(`-Instruct` 변형 없음). 좌표 규약은 여전히 미기재이므로 §4-4 카나리아 게이트가 최종 판정 |
| Q6 | VLM E=0.5 epoch의 스텝 환산 정의 | **단일 공식으로 통일(E 리뷰 Minor 1 반영):** `S_k = max(1, round(0.5 × n_k / B_eff))`, `n_k`는 **B의 counts.json** 값. §2-5 회계 매트릭스 검증도 같은 식을 쓴다(12a 함정 #3의 별도 표기는 이 식으로 대체). R=6이면 총 노출 3.0 epoch 등가. 파일럿에서 확정, R·E·스텝 수 논문 기재 |
| Q7 | FL에서 시드·셔플의 라운드별 배정 | 전역 시드 s 고정, 라운드 r·클라이언트 c의 데이터로더 셔플 시드 = hash(s, r, c) — 재현 가능·라운드마다 상이 |
| Q8 | 검출 증강 구성 | 파일럿 전 임시 = Ultralytics 기본값. 파일럿에서 mosaic+flip-only와 비교 후 하나로 확정(구현설계 2-5), **세 칸 동일 고정** |

---

# 7. 완료 기준 (이 스펙의 게이트 통과 조건)

1. §3 가정 13건이 A·D 스펙과 대조되어 계약 #2·#4가 얼었다.
2. §4 함정 4건의 처방이 Critical 지적 없이 통과했다.
3. §6 기본값 중 게이트가 뒤집은 항목이 이 문서와 configs에 반영됐다.
