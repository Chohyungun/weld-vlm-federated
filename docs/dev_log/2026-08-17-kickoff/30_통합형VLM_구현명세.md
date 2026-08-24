# 통합형 VLM 학습·연합 구현 명세

2026-08-21 · 세 관점 설계안 + 3렌즈 적대 검증(9회) 종합본
**이 문서가 통합형 구현의 단일 소스다.** 12_spec_C·12a와 어긋나는 부분은 제2부·제3부에 판정 근거와 함께 적었고, 문면 개정이 필요한 항목은 제7부에 표시했다.

대상: 5칸 중 **통합·중앙**, **통합·연합** 두 칸. 배경지식 SFT 2단(D3-(b) QA 2만, D3-(c) 판정추론 1만)은 5칸 공통 출발점이므로 같은 루프·같은 게이트를 통과한다.

세 관점이 각각 무엇을 들고 왔는지:
- **관점 1(학습 루프)** — 프레임워크 없는 직접 루프, 16GB 메모리 예산, 스텝 예산 회계
- **관점 2(연합·집계)** — buffer 실측, LoRA 집계 대수, 통신량 원장, 교환 규약
- **관점 3(출력·좌표)** — 좌표 규약 단일화, 생성·파싱 표기, 카나리아 게이트

적대 검증에서 **세 설계안 모두 `survives: false`**였다. 아래는 fatal 처방을 전부 반영하고 관점 충돌을 판정한 뒤 남은 것이다.

---

# 제1부 — 하드 게이트 (착수 전 통과 필수)

**게이트는 문서상 선행 조건이 아니라 실행 조건이다.** 산출물(`outputs/gates/*.json`)이 없거나 `passed=false`거나 내용 해시가 어긋나면 학습 진입점이 **첫 실행문에서 거부**한다. 우회 경로를 만들지 않는다 — 게이트 검사를 `if` 뒤에 두지 않고, 완화 스위치를 인자로 노출하지 않는다.

## 게이트 전체 지도

| ID | 이름 | 무엇을 증명하는가 | 막는 것 | 가중치 필요 |
|---|---|---|---|---|
| **G0** | 버전 핀 | 실효 라이브러리 버전이 잠금과 같다 | 메이저 드리프트로 아래 게이트 전부가 다른 물건을 잰다 | 아니오 |
| **G1** | 구조 프로브 | 모델 텐서 3집합(P/B/X) 전수 분류 | 이름 규칙 추정으로 buffer를 판정하는 것 | 아니오 (meta) |
| **G2** | **교환 폐포 감사** | "어댑터만 교환한다"가 집합 등식으로 성립 | 클라이언트에 남은 상태를 아무도 집계하지 않는 것 | **예 (NF4 실로드)** |
| **G3** | 프로세서 실측 | factor·size·grid 단위가 실호출 산출물과 정합 | 전처리가 두 칸에서 갈리고 흔적이 없는 것 | 예 |
| **G4** | 어댑터 부착 동결 | 248 모듈 부착, 비전 0건, 실현 해시 봉인 | linear_attention 24개 층 조용한 누락 | 예 |
| **G5** | **카나리아-1** | 좌표 규약이 실측으로 판별된다 | 좌표계 붕괴(IoU 0.938→0.055) | 예 |
| **G6** | 골드 자기채점 | 타깃 왕복이 예산 안, `train_index` 봉인 | 학습량 등가의 분모가 두 칸에서 갈리는 것 | 아니오 (NORM_1000) |
| **G7** | 자원 프로브 | 최악 배치·스텝 시간·커널 실체 | 본런 중 OOM, 15일짜리 계획을 모른 채 착수 | 예 |
| **G8** | 루프 등가 | grad accum ≡ 큰 배치, LR 궤적, 증인 4종 | 학습량 등가가 수식으로만 성립하는 것 | 아니오 (더미) |
| **G9** | val 예산 파일럿 | eos·truncated·처리량 | 평가셋으로 `max_new_tokens`를 고르는 것 | 예 |

의존: `G0 → G1 → G3 → {G4, G5(1a)} → G2 → G6 → G7 → G8 → [배경지식 SFT] → G5(1b) → 본학습 → G9 → eval`.

지시에 명시된 두 게이트가 **G2**(SSM persistent buffer 열거 검증)와 **G5**(카나리아-1)다. 먼저 적는다.

---

## G2 — SSM persistent buffer 열거 검증 → **교환 폐포 감사**

### 실측 사실 (관점 2가 meta device로 이미 확인, `Qwen/Qwen3.5-4B` 실 config, 총 4,539,265,536 파라미터)

| 항목 | 실측 |
|---|---|
| `named_buffers()` 전수 | **3개뿐** — `visual.rotary_pos_emb.inv_freq`[16], `language_model.rotary_emb.inv_freq`[32], `.original_inv_freq`[32] |
| 셋 다 | `persistent=False`, fp32, **state_dict에 없음** |
| `rope_parameters.rope_type` | `"default"` → forward 중 불변 |
| `state_dict − named_parameters()` | `['lm_head.weight']` — buffer가 아니라 `tie_word_embeddings=True`의 **별칭** |
| `linear_attn.conv1d` | `nn.Conv1d` 24개, **학습 파라미터**. buffer 아님 |
| `linear_conv_kernel_dim=4` | buffer가 아니라 conv1d **커널 폭**이었다 |
| `layer_types` | `linear_attention` 24 : `full_attention` 8 (`full_attention_interval=4`) |

**결론: 이 모델에 지속 buffer는 없다.** conv state·SSM state는 모듈 buffer가 아니라 forward 시점 캐시 객체(`Cache`)에 산다.

### 그런데 이 게이트 정의를 그대로 쓰면 안 된다 — 세 렌즈가 같은 급소를 찔렀다

1. **학습 모드 + `use_cache=False`에서 5스텝 전후 diff는 반드시 0이다.** 통과할 수밖에 없는 게이트로 논문에 사실 주장을 싣는 것은 게이트가 없는 것보다 나쁘다.
2. **`state_dict ∩ named_buffers` 정의는 bnb 양자화 상태를 구조적으로 못 본다.** `Linear4bit._save_to_state_dict`가 `weight.absmax`·`quant_map`·`nested_absmax`·`quant_state.bitsandbytes__nf4`를 `register_buffer` 없이 state_dict에 직접 주입한다. 즉 G2가 유일하게 뭔가를 찾을 수 있는 단계(NF4 실로드)에서 교집합이 여전히 ∅다.
3. **진짜 위험은 buffer가 아니라 "학습되는데 교환되지 않는 파라미터"다.** `modules_to_save`가 비어 있지 않거나 target 목록에 MTP·게이팅 모듈이 잘못 들어가면, 그 파라미터는 클라이언트 로컬에서만 갱신되고 어댑터 dict에 실리지 않는다. 6라운드 뒤 "글로벌 모델"이 어느 클라이언트의 모델과도 다르고 그 사실이 어디에도 없다.
4. **`rope_deltas`는 buffer가 아니라 평범한 인스턴스 속성**이고 매 forward 덮어써진다. `named_buffers()` 시야 밖이다.

### 채택 정의 — P/B/X 3집합 + 교환 폐포 등식

```
P = {n for n, _ in model.named_parameters(remove_duplicate=False)}
B = {n for n, _ in model.named_buffers()}
X = set(model.state_dict()) - P - B          # "state_dict 잉여물"
```

`remove_duplicate=False`가 핵심이다. 기본값(`True`)은 tied weight를 dedup해 `lm_head.weight`를 buffer로 **오탐**한다(레지스트리 #12). 이 정의면 tied 별칭은 P에 남고 X에는 bnb quant state만 들어온다.

**게이트 조건 G2-1 ~ G2-8**

| # | 조건 | 실패 시 |
|---|---|---|
| G2-1 | `X`의 전 원소가 `{TIED_ALIAS, QUANT_ARTIFACT, CACHE_LIKE, CONST_DERIVED}` 중 하나로 분류된다. `UNKNOWN` 0건 | 착수 차단(유형 D, 재검토 안건) |
| G2-2 | `B` 중 persistent(`= B ∩ set(state_dict)`)인 것 0건 | 유형 A 처방 발동(아래) |
| G2-3 | **`{n for n,p in model.named_parameters() if p.requires_grad}` == 어댑터 페이로드 키 집합** (부분집합 아님, 완전 일치) | 통합·연합 착수 거부 |
| G2-4 | 교환 페이로드 키에 buffer·X 원소 0건 | 즉시 실패 |
| G2-5 | 라운드 1 종료 시 클라이언트 state_dict의 **비어댑터 키 전수가 서버 배포본과 bit-exact** | "어댑터만 바뀐다"의 유일한 직접 증거. 실패 시 착수 거부 |
| G2-6 | 3클라이언트 독립 로드의 `base_quant_digest` 동일 | "동일 base 사전 배포" 전제 붕괴 |
| G2-7 | 같은 base를 **두 번 양자화**해 `base_quant_digest` 동일 (NF4 결정론) | 실패 시 재검토 안건 |
| G2-8 | 학습 forward에서 `use_cache`가 실제로 falsy. 5스텝 프로브를 **학습 모드**와 **`use_cache=True` generate 경로** 양쪽에서 돌려 두 모드의 buffer·속성 목록 차이를 보고서에 기록 | 기록 누락 시 미완 |

**`base_quant_digest` 재정의**: `named_buffers` 기반이 아니라 **모든 `Linear4bit`의 packed `weight` 바이트 + `X ∩ QUANT_ARTIFACT`의 (키, dtype, shape, 바이트 sha256)** 을 정렬해 만든 sha256. 이렇게 해야 이 값이 실제로 실패할 수 있다.

**프로브 확장**: `named_buffers()`에 더해 **모든 서브모듈의 `vars(m)` 중 Tensor 값 속성**을 diff 대상에 넣는다 — 여기서 `rope_deltas`가 잡힌다.

**보고서 필수 필드**: `n_tensors_examined`. 0건 통과가 **"없어서"인지 "안 봐서"인지** 구분되게 한다.

### 발견 시 처방 (사전 등록)

| 유형 | 조건 | 처방 |
|---|---|---|
| A | persistent + 학습 중 변이 + 평가에 사용 | **교환 단위 편입 + 표본수 가중 평균** — 검출 BatchNorm 처방과 동일 논리. FedBN식 로컬 잔존은 금지. 통신량 표·논문 서술 수정 |
| B | persistent + 불변 | 조치 없음. digest 동일 assert만 |
| C | 비persistent + 변이 (dynamic rope 등) | 조치 없음. state_dict·교환·병합 어디에도 실리지 않음을 증명 |
| D | 미분류 | **착수 차단**, 재검토 안건 |

### 검출 BatchNorm 처방과의 관계 — 논리는 같고 기제가 반대다

- **같음**: "글로벌 모델 1개를 글로벌 평가셋으로 채점한다 → 평가에 영향을 주는 클라이언트 유래 상태는 전부 집계돼야 한다." 전제가 그대로 성립한다.
- **다름**: 검출은 `state_dict` 전체를 보내므로 Flower가 **모든 것을 자동으로 평균한다** → 위험은 "평균하면 안 될 통계를 조용히 평균했다". 통합형은 교환 단위가 state_dict의 진부분집합이라 **아무것도 자동으로 실리지 않는다** → 위험이 정반대로 "클라이언트에 남은 상태를 아무도 집계하지 않았고 지표에도 흔적이 없다"다. 같은 함정의 거울상이고, 그래서 게이트도 diff가 아니라 **집합 등식**이어야 한다.
- **세척**: 클라이언트가 매 라운드 base를 새로 로드하므로 buffer가 base 값으로 재생성된다. 단 이 보증은 Ray actor가 warm 재사용을 하지 않을 때만 유효하다 — **buffer digest로는 warm actor를 탐지할 수 없다**(위 (1)과 같은 이유로 항상 상수다). 탐지는 판정 12의 3종으로 한다.

**논문 부수 결과**: 분리형은 데이터 의존 통계(BN running stats)를 매 라운드 밖으로 내보내고 통합형은 내보내지 않는다. 비반출 서사에서 유리한 대비축이고, RQ2의 결과 중 하나로 쓴다. **단 그 문장은 G2-3·G2-5를 통과한 뒤에만 쓴다.**

### 실행

```bash
python scripts/gate_structure_probe.py                       # G1 (meta, CPU 수초)
python scripts/gate_exchange_audit.py --stage runtime --clients 0,1,2   # G2
```
산출물 `outputs/gates/exchange_audit_uni.json` → MLflow artifact + 태그 `exchange_gate=pass|fail`. **fail이면 `fl/server_vlm.py`가 기동 자체를 거부한다.**

---

## G5 — 카나리아-1 (좌표 규약 argmax 판별)

**통합형 학습 착수와 타깃 직렬화 동결의 하드 게이트.** 문서가 아니라 실측이 좌표 규약의 최종 판정자다. 함정 #4(Qwen2-VL 0~1000 정규화 vs Qwen2.5-VL 절대 픽셀, 불일치 시 IoU 0.938 → 0.055)가 이 게이트의 존재 이유다.

### 절차

합성 도형 19장(흰 배경, 단색 사각형/원 1개, 정답 bbox를 생성 시점에 정확히 앎, 시드 고정) → 이미지당 **greedy 1회** → 출력을 `{NORM_1000, ABS_RESIZED, ABS_ORIG}` 각각으로 역변환해 IoU → median으로 argmax.

`ABS_RESIZED` 후보의 리사이즈 치수는 **그 호출에서 나온 `image_grid_thw`로 복원**하며 재계산하지 않는다(`coords.ImageGeom.require_resized`가 이미 추정을 거부한다).

### 도형 집합 — 관점 3의 "업스케일 필수"를 기각한다

관점 3은 "1280×720에서 ABS_ORIG↔ABS_RESIZED 혼동 IoU 0.95 이상"을 근거로 면적 65,536 미만 업스케일(120×90)을 **필수**로 못 박았다. 적대 렌즈가 같은 venv에서 실측했다:

| 박스 | ORIG→RESIZED 혼동 IoU |
|---|---|
| (600,640,640,660) 하단 소형 | **0.156** |
| (0,0,1280,720) 전면 | 0.978 |

즉 **네이티브 해상도에서 작은 도형을 하단에 놓기만 해도 분리된다.** 축퇴는 큰 박스·상단 박스에서만 일어난다. 2.4배 업스케일 초소형 이미지는 제로샷 파싱률 0.9 미달로 게이트를 막을 위험이 높은 불필요한 경로다.

**채택 규칙** (`build_specs`가 코드로 강제):
- (a) **네이티브 RT 해상도(1280×720, 1234×707)에 짧은 변 20~40px 도형을 프레임 하단 60% 이상 위치에** — 필수
- (b) 한 축이 900~1100px인 해상도 배제 (`NORM_1000` ↔ `ABS_ORIG` 축퇴)
- (c) 극단 종횡비 2종(2560×140, 140×2560) — x/y 스왑 노출
- (d) factor 비배수(1234×707)
- (e) 면적 < `min_pixels` 업스케일 — **선택**(분리 여유 확보용, 파싱률이 떨어지면 뺀다)

### 판별 가능성 사전 검사 — 모델을 부르기 **전에**

```python
confusion_iou(spec, geom, a: CoordCfg, b: CoordCfg) -> float   # 순수 기하, 모델 불필요
assert_separable(specs, geoms, cfgs, *, max_confusion_iou=0.5, min_images=8) -> None
```
세 후보의 모든 쌍에 대해 혼동 IoU ≤ 0.5인 이미지가 8장 이상이어야 판정에 진입한다. 12a F14의 "2배 이상 차이 해상도 포함"이라는 **입력 조건**을 "실제로 갈라지는가"라는 **출력 조건**으로 바꾼 것이다.

`IoU`는 자체 구현하지 않고 `evaluation.metrics.localization.iou`를 import한다 — 카나리아와 채점기가 다른 IoU 정의를 쓰는 상태를 만들지 않는다.

### 통과 조건 5종 (전부 충족)

| # | 조건 | 이유 |
|---|---|---|
| 1 | `argmax == 선언 coord_space` | |
| 2 | `median IoU(argmax) ≥ 0.5` | |
| 3 | **`margin ≥ 0.15`** (1위−2위 median) | 축퇴 상태에서 1위가 우연히 선언과 같아 통과하는 경로 차단 |
| 4 | `파싱 성공률 ≥ 0.9` | |
| 5 | `eos_stop_rate ≥ 0.9` | **정지 토큰 미지정을 본런 이전에 잡는 유일한 관측 지점**(제3부 F20) |
| 6 | `assert_separable` 통과 | |

전 후보 실패(argmax IoU < 0.5) 시 **자동 폴백하지 않는다.** 12a의 후퇴안("직전 세대 Qwen3-VL로 후퇴")은 2026-08-21 확정 사실인 "통합형·판정부 동일 체크포인트 공유"와 정면 충돌한다 — 통합형만 모델을 바꾸면 해소됐던 백본 비동일 교란이 되살아나 RQ2 주 기여가 훼손된다. **총괄 에스컬레이션 전용 경로다.**

### 체크포인트 2벌 — 1a / 1b

| 태그 | 대상 | 목적 | 논문 |
|---|---|---|---|
| **1a** | 원 배포본 `Qwen/Qwen3.5-4B` | 규약 판별 (파인튜닝 **전** 제로샷) | 이쪽을 싣는다 |
| **1b** | 배경지식 SFT 2단 **병합본** | 통합형의 실제 출발점. argmax 규약이 텍스트 SFT로 바뀌지 않았음의 회귀 확인 | 부록 |

`require_canary_pass`가 대조하는 것은 **1b의 `base_ckpt_sha256`**이다. 1a만 봉인하면 `H(raw) ≠ H_merged`로 모든 통합형 학습이 거부되고, 현장의 자연스러운 대응은 대조 항목에서 `base_ckpt_sha256`을 빼는 것(게이트 약화)이다. 20장×2벌이라 비용이 사실상 0이다.

**1a와 1b의 `argmax_space`가 갈리면 그 자체가 재검토 사유다** — 텍스트 전용 SFT가 grounding 규약을 바꿨다는 뜻이므로.

### 게이트 키 — 프로브를 고쳐 통과시키는 경로를 막는다

```python
gate_key(*, base_ckpt_sha256, coord_cfg_hash, preproc_sha256, prompt_bundle_sha,
         probe_sha256, quantized_modules_sha256, base_yaml_sha256,
         env_fingerprint_sha, transformers_version) -> str
```

관점 3의 원안은 `probe_sha256`(프로브 프롬프트 + 도형 스펙)이 키에 **없었다.** 그러면 통과할 때까지 프로브를 고쳐 돌릴 수 있고, 성공 레코드가 실패 이력을 남기지 않은 채 같은 키를 덮는다. 좌표 규약 판정이 "실측"이 아니라 "통과할 때까지 조정한 실측"이 되어 하드 게이트의 증명력이 0이 된다. **키에 넣으면 프로브를 고칠 때 키가 바뀌어 이전 기록이 자동 무효가 되고, 두 키의 레코드가 디렉터리에 나란히 남아 재시도 이력이 보존된다.**

추가로 게이트 레코드에 **`attempt_seq`**(단조 증가)와 직전 시도 결과 요약을 필수 필드로 넣고, **시도 3판단 요청터는 승인 필드 없이 `require_canary_pass`가 통과를 거부**한다.

레코드 필드: `{passed, tag(1a|1b), argmax_space, iou_table, margin, parse_rate, eos_stop_rate, separability, attempt_seq, prev_attempts, mlflow_run_id, record_sha256}` + 위 gate_key 입력 전부.

`quantized_modules_sha256`이 키에 있는 이유: **카나리아-1과 본학습의 양자화 구성이 반드시 같아야 한다.** 비전 타워가 카나리아에서는 bf16, 본학습에서는 NF4라면 게이트가 검증한 좌표 규약이 본학습 모델의 규약이라는 보장이 없다. 위치 지표는 RQ2 결과표의 축이다.

### 프로브 프롬프트는 본 실험 프롬프트와 별도 파일·별도 해시

`vlm/prompts/canary_grounding_v1.txt`(영어 — Qwen 쿡북 어휘에 맞춰 제로샷 파싱률을 올린다). 카나리아는 5칸 평가가 아니므로 재실행이 허용되고, 본 프롬프트의 한 글자 동결과 섞이면 안 된다.

### 실행

```bash
python scripts/vlm_canary1.py --config configs/base.yaml --ckpt <raw>       # 1a
python scripts/vlm_canary1.py --config configs/base.yaml --ckpt <merged> --tag 1b
```
기대값을 구현 출력으로 갱신하는 스위치를 제공하지 않는다(`run_fixtures.py`와 같은 원칙). 실패는 실패로 끝나고, 처방은 `coord_space` 값 교체 또는 프로브 수정 후 **키를 바꿔** 재실행이다.

---

## 나머지 게이트 요약

### G0 — 버전 핀
`pyproject.toml`의 `transformers>=4.51`을 **잠금과 일치하는 정확 핀**으로 좁힌다. `uv.lock`은 win32/linux에 transformers **5.15.0**, darwin에 5.8.1을 잠갔고 모델 config의 `transformers_version`은 `4.57.0.dev0`이다 — 하한만 걸린 지금 상태는 메이저가 갈린 채 두 기계가 다른 물건을 돌릴 수 있다. `torch`·`peft`·`bitsandbytes`·`trl`·`accelerate`·`flwr[simulation]`·`ray` 전부 정확 핀 + `uv lock` 커밋. 실효 버전을 MLflow 필수 태그로 남기고 전 셀 단일값 assert. **Flower API가 레거시인지 Message인지도 여기서 실측 확정한다**(판정 14).

### G1 — 구조 프로브 (가중치 불필요)
`AutoConfig`(로컬 config.json) → meta device 인스턴스화 → P/B/X 열거. CPU 수초. `scripts/gate_structure_probe.py`. **이미 실행해 통과했다**(위 실측 표). 회귀 스냅샷 테스트로 고정해 transformers 버전이 올라가 구조가 바뀌면 여기서 깨지게 한다.

### G3 — 프로세서 실측 (P1~P6)
| # | 검사 | 잡는 것 |
|---|---|---|
| P1 | `factor == patch_size × merge_size` (실측 대조, **32도 28도 코드에 없다**) | 하드코딩 |
| P2 | `pixel_values.shape[0] == t*h*w` | grid 축 순서 |
| P3 | `input_ids`의 `image_token_id` 개수 == `t*h*w // merge_size**2` | **grid가 patch 단위인지 merged 단위인지** |
| P4 | 복원 `rw·rh`가 factor 배수이고 `min_pixels ≤ rw*rh ≤ max_pixels` | 호출 인자 override |
| P5 | `abs(rw/rh − orig_w/orig_h) ≤ factor 반올림 허용치` | EXIF 회전·manifest W/H 오기 |
| P6 | `size.longest_edge` 동결값이 `pick_longest_edge.py` 산출 P99.5와 정합 | ceil 식 5% 과대 추정(레지스트리 #21) |

P3가 결정적이다. `geom_from_grid`의 `resized_h = grid_h × patch_size` 가정은 **selfcheck도 오라클 주입도 반증할 수 없다**(encode와 decode에 같은 geom을 넣으므로 2배 틀려도 왕복 오차 0). 프로세서 실호출 교차검증만이 이걸 잡는다.

산출물 `outputs/gates/processor_facts.json`의 sha256이 `preproc_sha256`이고 `gate_key` 입력이다.

### G4 — 어댑터 부착 동결
`scripts/dump_target_modules.py` 1회 실행 → `configs/target_modules_uni.json` 동결. 부착 모듈 **248**, 파라미터 **32,464,896**, fp32 **129,859,584 B**를 학습 시작 시 assert. `visual|vision|merger` 0건. `realized_adapter_sha256`(판정 21) 봉인.

### G6 — 골드 자기채점 + `train_index` 봉인
D4 페어 전수를 `encode → serialize → parse → decode` 실경로로 통과시켜 **좌표별 절대 오차 ≤ `coords.roundtrip_budget_px(geom, cfg)`, 위반 0건**. 합격 기준이 IoU가 아닌 이유는 제3부 F16에 있다. 동시에 `outputs/vlm/train_index_{coord_cfg_hash}.jsonl`(폐기 후 페어 목록 + 드롭 사유·건수) 생성·sha256 봉인 → **통합 2칸의 `n_k_effective` 단일 소스**. 클라이언트별 폐기율 상한 1%(전역 아님)와 클라이언트 간 편차 1%p를 함께 검사한다.

### G7 — 자원 프로브 (B1~B5)
| # | 프로브 | 판정 |
|---|---|---|
| B1 | **길이 상위 1% 샘플만으로 구성한 최악 배치** 20 step, `max_memory_allocated`/`reserved` | 평균 길이 파일럿은 꼬리에서 죽는다 |
| B2 | 같은 base 2회 양자화 digest 동일 | G2-7과 공유 |
| B3 | micro-pass 시간 중앙값 × 총 micro-pass 수 → GPU-일 | 15~21일이면 착수 전 총괄 |
| B4 | 실효 커널 경로 기록(`env_fingerprint`) | 통합 2칸이 다른 수치 경로로 도는 것 |
| B5 | `triton-windows` + `fla-core`로 fast path가 켜지는가 | 켜지면 공통 고정 9번에 편입 |

**B1을 통과하지 못하면 본학습을 시작하지 않는다.** 추론 OOM에는 사다리가 없으므로(제4부 4-20) 여기가 유일한 발견 지점이다.

### G8 — 루프 등가
더미 모델(GPU 불필요)로: `accum k×b`의 누적 그래디언트가 `batch k·b` 단일 스텝과 **1e-5 이내 일치**(마이크로배치마다 총괄 토큰 수가 다른 불균형 케이스 필수), R라운드로 쪼갠 `step_offset` 재개 LR 궤적이 단일 cosine 닫힌형과 전 스텝 일치(1e-12), 증인 4종 일치, 첫 스텝 직후 학습 파라미터 496개 전수 `p.grad is not None and p.grad.abs().max() > 0`.

### G9 — val 예산 파일럿
**`split='val'` 전용 진입점**(`vlm/pilot_budget.py`)에서만 `max_new_tokens`를 정한다. `truncated_rate == 0` **and** `eos_stop_rate == 1.0`을 동시에 확인한 뒤에만 eval을 허용하고, `images/s`·p50/p99 latency·peak VRAM·예상 총 wall clock을 게이트 레코드에 남긴다. 확정된 `max_new_tokens`는 `gen.effective_sha256`으로 봉인되어 이후 변경이 게이트를 무효화한다.

---
# 제2부 — 관점 충돌 판정

세 관점이 어긋난 지점 22건. "둘 다 맞다"로 닫지 않는다. 각 판정에 **채택 / 기각 / 근거**를 적는다.

## 판정 1 — 학습 프레임워크: TRL SFTTrainer vs 직접 루프 → **직접 루프**

| | TRL 경로 (스펙 §2-1 문면) | 직접 루프 (관점 1) |
|---|---|---|
| 숨은 기본값 표면 | TRL 전처리 + `Trainer` 전 표면 상속 | 우리가 쓴 코드만 |
| 레지스트리 "다음 수색 표면" 6건 중 | 3건(`save_total_limit`, 재개 상태 복원, `save_embedding_layers`)이 **이 경로에서만** 발생 | 해당 없음 |
| 라운드 무상태 | 라운드마다 Trainer 재생성 + 스케줄러 수동 주입 + "Trainer가 몰래 하는 일" 재확인 | `train_round` 단일 함수, detection이 이미 증명한 패턴 |
| VL 손실 | **`accepts_loss_kwargs=False`라 `num_items_in_batch`를 구조적으로 무시**(레지스트리 #7) | `labels=None` + 자체 정규화 |
| 샘플러 | `Trainer.__init__`의 `set_seed` + generator 없는 `RandomSampler` → **fresh Trainer가 매번 같은 순열**(레지스트리 #20) | 단일 가상 순열(판정 8) |
| 잃는 것 | — | `trainer_state.json` 독립 증인 |

**채택: 직접 루프.** 잃는 독립 증인은 증인 4종(판정 19 아래)으로 더 강하게 대체된다. 새로 지는 위험은 grad accum 손실 정규화 하나뿐이고 G8이 1e-5로 잠근다. **통제 가능한 하나의 위험 대 감사 불가능한 열 개의 기본값이면 교환은 명확하다.**

단 스펙 §2-1 문면과 다르므로 **승인 항목(Q2)**. 승인 전 기본값으로 진행하되 `run_steps`의 공개 표면을 프레임워크 중립으로 유지해 되돌릴 수 있게 둔다. 코퍼스 담당이 TRL을 쓴다면 레지스트리 #6은 여전히 유효하므로 처방을 폐기하지 않는다. `datasets`(Arrow)도 쓰지 않는다 — 캐시 무효화가 조용히 어긋나는 표면을 하나 더 만든다.

## 판정 2 — 손실 정규화 정의 → **shift 후 총괄 토큰 총합 분모, `vlm/loss_norm.py` 단일 소유**

세 값이 서로 다르다:
- 관점 1: 마이크로배치 **평균의 합**을 피하고 "총괄 토큰 수 가중 합 ÷ 총 총괄 토큰 수"
- 프레임워크 렌즈: VL 클래스 `.loss`는 **마이크로배치 내 토큰 평균**만 준다. 분리형 판정부가 쓸 `ForCausalLM`은 **토큰 가중 합/총합**을 받는다 → **같은 체크포인트인데 두 구조가 다른 목적함수를 최적화한다**
- 불변 렌즈: 토큰 평균 + FedAvg `n_k` 가중이면 클라이언트 k의 토큰이 `1/τ_k` 가중을 받아 **통합·중앙(균일)과 통합·연합이 다른 목적함수**

**채택**:
```python
# vlm/loss_norm.py — 학습 루프·감사·테스트·분리형 판정부가 이 함수만 부른다
shift_labels = labels[..., 1:]
shift_logits = logits[..., :-1, :]
loss = Σ_micro CE_sum ÷ Σ_micro (shift_labels != -100).sum()
```
- `model(**batch, labels=...)`의 `.loss`를 **쓰지 않는다.** `labels=None`으로 호출해 logits만 받는다.
- 분모는 **shift 후** 카운트다. shift 전에 세면 마지막 위치가 총괄 토큰인 샘플(= 배치 내 최장 샘플, 동적 패딩이라 매 배치 반드시 1개 이상)마다 분모가 1씩 과대해지고, 배치마다 방향이 달라 G8의 1e-5 테스트가 재현 없이 실패한다.
- **분리형 판정부에도 같은 함수를 주입한다**(TRL이면 `compute_loss_func`). 두 구조의 정규화식 동일성을 교차 테스트로 잠근다 — RQ2가 주 기여인데 손실 정규화 비대칭이 통제되지 않은 교란으로 들어가면 두 구조 모두 재학습이다.
- `effective_train_config`에 `loss_reduction`·`denominator_rule="shift_labels"`·`model_class`·`accepts_loss_kwargs` 4필드.

## 판정 3 — FedAvg 가중 단위: `n_k` vs `T_k`(총 총괄 토큰) → **`n_k_effective` 기본, τ 비 초과 시에만 전환**

토큰 평균을 버리고 판정 2의 총합 분모를 쓰면 클라이언트 목적함수가 이미 토큰 단위로 균일해지므로 `1/τ_k` 왜곡의 주 경로가 닫힌다. 남은 것은 라운드 단위 가중이다.

**채택**: 파일럿 100 step에서 클라이언트별 `τ_k`(샘플당 평균 총괄 토큰 수)를 실측하고 회계 매트릭스에 `supervised_tokens` 컬럼을 추가한다. `max(τ_k)/min(τ_k) ≤ 1.02`면 `n_k_effective` 유지, 초과하면 `T_k` 전환을 **총괄에 제안**(검토 결정 F 개정 항목이므로 자동 전환 금지). 어느 쪽이든 `τ̄_w/τ_k` 왜곡 계수를 클라이언트별로 논문에 싣는다.

## 판정 4 — `n_k` 소스: `counts.json` vs 필터 후 → **`n_k_effective`(`train_index`)**

확정 사항은 `counts.json`(길이 필터 **전**)을 단일 소스로 동결했다. 그대로 두면:
- `S_k`의 분모와 감사 규칙 ④의 분모가 **같은 값**이라 감사가 3.000을 반드시 출력하는 **항진명제**가 된다
- 통합·중앙은 필터 후 풀로 3 epoch, 통합·연합은 필터 전 수로 계산한 `S_k`를 돈다 → 총 갱신이 폐기율만큼 어긋난다. RQ1 회복률의 분자가 부풀려진다
- 폐기는 긴 샘플(결함 많은 이미지)에 집중되고 **C3(알루미늄)이 구조적으로 손해**를 본다 — 하필 RQ3의 주인공

**채택**: `n_k_declared`(counts.json)와 `n_k_effective`(G6의 `train_index`)를 **둘 다 회계에 기록**하고, `plan_steps`의 분모와 FedAvg 가중은 `n_k_effective`를 쓴다. 감사 규칙 ④″로 `n_k_declared − dropped_long_samples == n_k_effective` 정확 일치를 검사한다. 결정 F 개정이므로 **총괄 안건(Q6)**.

필터는 통합 2칸이 **같은 순수 함수**(tokenizer sha + chat_template sha + max_seq_len + prompt sha + processor cfg sha를 인자로 받는)를 쓰고, 산출물은 런 시작 **전에** 한 번 동결한다.

## 판정 5 — 역변환 정수화: "각 방향 1회" vs 0회 → **역변환 정수화 0회**

관점 3의 핵심 원칙("정수화 각 방향 1회")을 **기각한다.** 적대 렌즈가 `.venv` 실행으로 확인한 산술:

> NORM_1000, 1280×720, GT `x1=2px` → `to_model` 1.5625 → `quantize` 2 → `to_px` 2.56 → `quantize` **3**. 오차 1.0px, `roundtrip_budget_px` = 0.64px. **위반.**
> W=1280에서 정수 x값의 21.9%, W=1600에서 37.5%가 예산 초과. W=2500에서 최대 오차 1.75px > 예산 1.25px.

즉 **G6(골드 자기채점)이 정상 코드에서 통과 불가능**해지고, 마감 압력 하의 자연스러운 처방은 "예산을 늘린다"이며 그 순간 좌표계 붕괴를 잡는 유일한 사전 게이트가 무력화된다. 부수 피해로 원본 폭이 1000 미만인 이미지에서 인접 모델 좌표 2개가 같은 정수로 뭉개져 정상 예측이 `bbox_invalid` 오답이 되고, 손실이 소형 결함(기공·슬래그)에 집중돼 RQ2를 계통 편향시킨다.

**근거 3중**: (a) `roundtrip_budget_px`는 `run_fixtures.py` L64-85가 증명하듯 **역방향 정수화 없는 경로**를 위해 정의됐다. (b) `evaluation.schema.Defect.bbox_px`는 `tuple[float, float, float, float]`다(실물 확인). (c) 13_spec_D §3-4 실패모드 8이 "내부 float 유지, 정수화는 어댑터 경계 최종 1회"로 이미 확정했다.

**채택**: `decode_boxes_to_px`는 `to_px` 1회만. `bbox_px`는 float 그대로 기록. 정수 표기가 필요하면 D 어댑터의 최종 1회.

## 판정 6 — 좌표 스냅 소유: C vs 채점기 → **채점기 단독** (관점 3 유지)

계약 #4 §3-4가 "경계 이탈 ≤1px 스냅"을 채점기 권한으로 이미 못 박았다. C가 같은 일을 하면 보정이 두 곳이 되고, 클램프는 IoU를 올리는 방향으로만 작동하므로 답을 고쳐 주는 것이다. C는 `validate_box_px`로 **판정·계수만** 하고 값은 그대로 낸다. `snap_to_bounds`는 `vlm/` 어느 파일에서도 호출하지 않으며 AST 테스트로 부재를 강제한다. 이탈량 분포는 `failrates.json`에 남겨 D가 자기 스냅 발생률과 대조할 수 있게 한다.

## 판정 7 — `CoordCfg` 확장 vs `preproc_sha256` → **확장하지 않는다** (fatal 지적 부분 기각)

불변 렌즈가 "`CoordCfg`에 `size.shortest_edge`/`longest_edge`가 없어 `coord_cfg_hash`가 리사이즈 키에 눈이 멀었다"를 **fatal**로 올렸다. **실물 확인 결과 부분적으로 틀렸다**:

```python
# vlm/coords.py L72-105 (실물)
class CoordCfg:  coord_space, min_pixels, max_pixels, factor, norm_denominator, transformers_version
def canonical_json(self): return json.dumps(asdict(self), sort_keys=True, ...)
```
`min_pixels`/`max_pixels`/`factor`가 이미 해시 입력이다. Qwen3.5의 `size={shortest_edge, longest_edge}`와 Qwen2.5-VL의 `min_pixels/max_pixels`는 **둘 다 면적 단위**이고 라이브러리가 같은 역할로 매핑하므로, `coord_runtime`이 `shortest_edge→min_pixels`, `longest_edge→max_pixels`로 **고정 매핑**하는 한 `longest_edge`가 바뀌면 `coord_cfg_hash`도 바뀐다. 시나리오는 성립하지 않는다.

**남는 진짜 구멍 2개**는 인정한다: (a) `max_seq_len`은 해시에 없다. (b) **호출 시점 kwargs override**는 객체 속성을 읽는 어떤 검사로도 안 잡힌다(판정 17).

**채택**: `coords.py`를 손대지 않는다. 이미 `SHA256SUMS`(e2bed000…)·골든 픽스처 12/12·5칸 `coord_cfg_hash`에 묶여 있어 한 글자만 바뀌어도 전 게이트 재봉인이 따라붙는다. 대신 **`preproc_sha256`**(런타임 `processor.image_processor.to_dict()` + `max_seq_len` + factor 실측의 sha256)을 `REQUIRED_TAGS`와 `check_cells_identical` 비교 필드에 넣고, `base.yaml` 자체의 sha256을 `gate_key`에 봉인한다. **같은 방어를 리프 수정 없이 얻는다.** (Q1 — 총괄)

## 판정 8 — 표본 노출: 라운드별 재셔플(스펙 Q7) vs **단일 가상 순열**

스펙 Q7은 라운드 r의 셔플 시드를 `hash(s,r,c)`로 정한다. E=0.5와 결합하면 각 라운드가 **새 순열의 앞 절반**만 소비한다:
- 특정 샘플이 6개 순열 모두에서 뒷절반에 떨어질 확률 `(1/2)^6 = 1.5625%` → 가장 작은 클라이언트에서 **전체 페어의 약 1.6%가 한 번도 학습되지 않는다**
- 노출 횟수 분산 6·0.25 = 1.5 (sd 1.22). 통합·중앙은 정확히 3회
- 검출 칸은 E=2(정수)라 이 문제가 없다 → **분수 E를 쓰는 통합·연합에만 생기는 비대칭**이 RQ2에 그대로 실린다
- 프레임워크 렌즈가 더 나쁜 변형을 실행으로 확인했다: fresh Trainer가 **매번 같은 순열**을 내면 고유 샘플이 50%다. 그런데 `R·S_k·B_eff/n_k`는 정확히 3.0이라 감사가 통과한다

**채택**: 클라이언트마다 길이 `R·S_k·B_eff = 3·n_k`의 **단일 가상 순열**을 시드 하나로 만들고, 라운드 t는 전역 커서 `[(t−1)·S_k·B_eff, t·S_k·B_eff)` 구간을 잘라 쓴다(경계에서 다음 epoch 순열로 넘어감). **LR을 전역 오프셋에서 재개하는 것과 정확히 같은 논리를 데이터 스트림에도 적용한다.** 모든 샘플이 정확히 3회(나머지 < B_eff) 노출되어 중앙 칸과 표본 수준까지 등가가 된다.

회계 컬럼 `sample_cursor_start`·`sample_cursor_end`·`sample_index_digest`·`exposure_min`·`exposure_max`, 감사 규칙 ⑭: `exposure_max − exposure_min ≤ 1` **and** `coverage == 1.0` **and** 같은 클라이언트의 `sample_index_digest`가 라운드마다 서로 다름. 스펙 Q7 개정이므로 **총괄 안건(Q7)**.

## 판정 9 — `micro_batch ↔ grad_accum`을 "레시피 중립 레버"로 볼 것인가 → **아니다**

관점 1은 이 교환을 "허용, 실사용만 기록"으로 분류했다. 두 렌즈가 각각 반박했다:
- LoRA `dropout > 0`이면 드롭아웃 마스크가 마이크로배치 텐서 shape로 뽑히므로 micro=2/accum=16과 micro=1/accum=32는 **다른 난수열을 소비**한다. 통합형은 1시드라 반복 측정으로 흡수할 수단이 없다
- 판정 2의 총합 정규화를 쓰더라도 총괄 위치 한정 로짓 경로는 micro≥2에서 의미가 달라진다(판정 11)

**채택**: (a) **LoRA `dropout = 0.0` 고정** — 어댑터 설정은 5칸 공통 고정 2번이므로 게이트 결정이고, 0으로 두면 micro_batch가 난수 소비 측면에서 진짜로 중립이 된다. (b) **`micro_batch = 1` 고정**(판정 10). (c) 실제로 쓴 레버 조합의 sha256을 **`lever_fingerprint`**로 `REQUIRED_TAGS`에 추가하고 `check_cells_identical` 비교 필드에 넣어, 두 칸이 다른 레버로 돌면 채점 전에 실패시킨다.

## 판정 10 — 시작값 micro=2 vs micro=1 → **micro=1 / accum=32. micro=2는 사다리에 없다**

관점 1의 메모리 표가 지배항에서 2배 이상 틀렸다. 두 렌즈의 독립 실측이 일치한다:

| 관점 1 표 | 실측 |
|---|---|
| vocab "약 15만" | **248,320** (`configuration_qwen3_5.py` L82) |
| embed/lm_head 언급 없음 | **각 1.19 GB bf16** — `nn.Embedding`은 bnb 변환 대상이 아니고 lm_head는 기본 보호. tied라도 둘 다 상주할 수 있다 |
| 로짓+CE "1.5~2.5 GB" | `ForCausalLMLoss`가 **무조건 `logits.float()`**(`loss_utils.py` L58-59). 격리 실측: B=1/S=1300 → **4.22 GB**, B=1/S=2048 → **6.64 GB**, B=2/S=1300 → **8.43 GB** |
| 가용 16GB | `mem_get_info` idle free **14.72 GB** (Windows WDDM 1.21 GB 상주) |

micro=2/S=1300은 **확정 OOM**이다. 그리고 동적 패딩이라 2048 토큰 샘플이 처음 뽑히는 스텝에서 **비결정적으로** 터진다.

**채택**: 시작값 micro=1/accum=32(유효 배치 32 불변). 사다리에서 micro=2 제거. 예산 상한 14.7 GB. 길이 버킷팅으로 최대 길이 배치가 언제 오는지를 결정적으로 만들고, G7-B1이 가장 긴 버킷부터 돈다.

## 판정 11 — 총괄 위치 한정 로짓: 선택 레버 vs **착수 전제**

관점 1은 "테스트 통과 전에는 켜지 않는다"는 선택 레버로 뒀다. 판정 10의 실측이면 이걸 안 켜고는 S=2048이 14.72 GB 한계선 위아래다. 총괄 200/1300 토큰이면 4.22 GB → 약 0.65 GB로 **3.5 GB를 번다.**

**채택: 착수 전제로 승격**하되 세 조건을 붙인다.
1. **공식 경로만 쓴다.** `modeling_qwen3_5.py` L1779가 `slice_indices = slice(...) if isinstance(logits_to_keep, int) else logits_to_keep`이므로 `logits_to_keep`에 **위치 LongTensor**를 넘기면 모델 내부 수술 없이 된다.
2. **`micro_batch == 1`을 강제한다.** 인덱스 텐서는 dim=1에 대해 **배치 전 행에 동일하게** 적용된다. 샘플 A 총괄 [900,1100], 샘플 B 총괄 [400,600]이면 합집합 401~701 위치를 통과시켜야 하고 절감이 사라진다. 루프 진입에 `assert cfg.micro_batch == 1 or not use_supervised_logits`.
3. **`labels=`를 넘기지 않는다.** `ForCausalLMLoss`는 labels를 받으면 내부에서 pad+shift를 하므로, 이미 gather된 logits에 원본 labels를 넘기면 **예외 없이 1토큰 밀린 손실**이 나온다. gather 위치에 맞춰 미리 shift한 타깃으로 직접 CE를 부른다.

**기준 경로 고정**: 손실 일치 테스트의 reference는 **프레임워크 자체 경로**(`model(**batch, labels=labels).loss`)여야 한다. 우리가 쓴 나이브 CE를 기준으로 삼으면 같은 off-by-one을 공유해 통과한다. 인덱스 이름을 `hidden_positions = label_positions - 1`로 강제한다.

## 판정 12 — warm actor 탐지: buffer digest vs **3종 실측**

관점 2는 fit 진입 buffer digest 대조를 warm actor 오염 탐지기로 5곳에 배치했다(§0·§5·#9·회계 ⑥·테스트). **§0이 스스로 증명한 사실 때문에 이 탐지기는 원리적으로 발동하지 않는다** — persistent buffer 0개, 나머지 3개는 `rope_type="default"`라 불변이므로 digest는 항상 같은 상수다. 게다가 `client_resources={"num_gpus": 1.0}` + GPU 1장이면 Ray는 액터를 **정확히 1개** 만들고 18회 fit을 그 프로세스에서 순차 실행한다 — warm 재사용은 가능성이 아니라 **강제된 구성**이다.

**채택**: buffer digest는 불변량 감사용으로만 남기고, 탐지는 3종으로 한다.
1. fit 진입 즉시 `assert torch.cuda.memory_allocated() < 64<<20` (cold면 0에 가깝고 warm이면 GB 단위)
2. 프로세스 전역 단조 카운터 `_BUILDS`와 `os.getpid()`를 셀에 기록 → 감사에서 `(pid, _BUILDS)` 쌍이 셀마다 유일하고 fit 순서와 1:1 증가
3. `device_map={"": 0}` 하드코딩(**`"auto"` 금지** — accelerate가 `mem_get_info`로 배치를 정하므로 warm 잔존이 클라이언트마다 다른 실행 그래프를 만든다) + 로드 후 `assert {p.device for p in model.parameters()} == {cuda:0}`

추가로 fit 진입에 **`torch.cuda.reset_peak_memory_stats()`**. 없으면 `max_memory_allocated`가 액터 수명 전체의 단조 최댓값이라 두 번째 셀부터 전부 같은 값을 보고해 VRAM 근거가 증발한다.

## 판정 13 — 어댑터 wire dtype: bf16 왕복 vs **fp32 전 구간**

관점 2는 "캐스트 지점 정확히 2곳(bf16↔fp32)"을 규약으로 뒀다. PEFT는 `autocast_adapter_dtype=True`(기본)에서 lora_A/lora_B를 **fp32로 만든다**. 즉 `to_wire`의 bf16 입력 assert가 라운드 1 첫 fit에서 터지고, 그걸 믿고 bf16 캐스트를 넣으면 매 라운드 글로벌 어댑터가 **상대오차 ~2e-3**로 절삭된다. 그런데 부록 1단이 재려는 양 `Σ p_k δB_k δA_k`는 로컬 갱신의 2차항으로 통상 0.1% 이하다 — **측정 노이즈가 측정 대상보다 크다.** 그 상태로 낸 ε 표는 bf16 반올림 잔차를 클라이언트 교차공분산이라고 보고하게 되고, "두 경로로 계산해도 같다" 테스트는 두 경로가 같은 오염을 받으므로 통과한다.

**채택**: 어댑터를 **fp32로 끝까지 유지**한다(PEFT 기본 동작이 이미 그렇다). 캐스트 지점 **0곳**. `to_wire`의 assert를 `dtype is torch.float32`로, `from_wire`는 캐스트 없이 fp32 주입. wire fp32 결정과 통신량 129,859,584 B는 불변이다. 테스트를 `test_어댑터가_전_구간_fp32다`(서버 init → wire → 주입 → 학습 후 반환 dtype 전수)로 교체한다.

부수 정정: "정렬 + float64 누적"의 근거를 **"비트 재현성"이 아니라 "서버 집계 단계의 결정론 — Ray 완료 순서가 결과에 새어들지 않음"**으로 바꿔 쓴다. 클라이언트 학습은 SDPA 폴백·bnb dequant·recompute 때문에 비트 결정론이 아니다.

## 판정 14 — 라운드 실패 방어: `accept_failures=False` vs **전략이 직접 검사**

flwr **1.33.0**(설치본. 2026-08-23 트랙 C 실측으로 정정 — 원 표기 1.34)의 Message API `FedAvg` 생성자에는 **`accept_failures` 파라미터가 없다**(`fraction_train`/`min_train_nodes`/`min_available_nodes`/…). `aggregate_train`은 `valid, _ = self._check_and_log_replies(...)`로 **에러를 버린 뒤** 남은 것으로 집계하고, 흔적은 INFO 로그 한 줄이다. 반대로 레거시 `flwr/server/strategy/fedavg.py`에는 `accept_failures: bool = True`가 있다. 관점 2의 설계는 `configure_fit`/`aggregate_fit`(레거시 이름)을 쓰면서 Message API 배선을 전제해 **둘 중 어느 쪽에서도 호출되지 않는다.**

**채택**: G0에서 API를 실측 확정하고 문서에 못 박는다. 어느 쪽이든 방어는 **전략 코드가 직접** 한다:
```python
valid, errors = self._check_and_log_replies(replies, is_train=True)
if errors: raise UniFedRoundFailure(...)
if len(valid) != len(cfg.client_ids): raise UniFedRoundFailure(...)   # 에러 없이 덜 샘플링된 경우
```
`configure_train`에서도 `sample_nodes` 반환 길이 != K면 라운드 시작 전에 실패(`sample_nodes`가 `min_available_nodes = max(min_available_nodes, sample_size)`로 덮어써 명시값을 조용히 무시할 수 있다).

**`fraction_evaluate=0.0`, `min_evaluate_nodes=0`을 인자 노출 없이 하드코딩한다.** 기본 1.0이면 `Server.fit()`이 매 라운드 `evaluate_round`를 무조건 호출해 (a) 클라이언트 evaluate 구현 시 NF4 모델을 3회 더 빌드해 `model_builds == 1` assert가 깨지고 (b) 미구현 시 매 라운드 failures 3건이 `aggregate_evaluate`에서 조용히 삼켜져 "실패 라운드 0건" 주장이 거짓이 된다. `aggregate_evaluate`에도 fit과 대칭인 실패 검사를 넣는다.

`aggregate_arrayrecords`는 쓰지 않는다 — reply 도착 순서대로 fp32 누적한다. 우리 정렬·float64 집계기를 쓰는 이유를 주석이 아니라 테스트로 고정한다. 레지스트리 #2·#10 갱신(#19).

## 판정 15 — 통신량 게이트: "정확 일치" vs **2열 분리**

관점 2의 `assert_expected`(실측 바이트 == 파라미터수×4 **정확 일치**)는 Flower 직렬화와 양립 불가능하다. 실측: 496 텐서, `sum(nbytes)` = 129,859,584 B, flwr 직렬화 후 = **129,923,072 B**(npy 헤더 496 × 128 = 63,488 B). 라운드 1 첫 기록에서 예외가 나고 `accept_failures` 방어와 맞물려 런이 전혀 시작되지 않는다. `sum(nbytes)`로 우회하면 이번엔 "실사용을 기록한다"를 어긴다.

**채택**: `CommsRecord`를 2열로 쪼갠다.
- **`tensor_nbytes`** = Σ nbytes → **정확 일치 게이트 대상**(임베딩 동봉·잉여물 혼입을 잡는 원래 목적은 여기서 달성)
- **`wire_nbytes`** = 직렬화 실측 → 논문 표에 싣는 수치. 게이트는 `0 ≤ wire − tensor ≤ 256 × n_tensors`

결과표의 "통합 129.86 MB vs 분리 37.60 MB" 대비는 **양쪽 다 `tensor_nbytes`로 통일**하고(detection `serialize.payload_nbytes`도 `sum(nbytes)` 정의다) 각주에 "직렬화 헤더 오버헤드는 통합형 0.049%, 분리형 <0.01%로 결론에 영향 없음"을 적는다.

## 판정 16 — 부록 2단 SVD: dense vs **QR 코어**

`ΔW = Σ_k p_k B_k A_k`의 랭크는 정의상 `K·r = 48` 이하다. dense를 거칠 이유가 없다. 상세는 제4부 4-22.

## 판정 17 — `target_modules` 표기: 전체 경로 정규식 vs **접미사 앵커**

관점 2는 경로 앵커 정규식(`model\.language_model\.layers\.\d+\.…`)을 제안했다. 프레임워크 렌즈가 반박했다: `Qwen3_5ForConditionalGeneration`의 디코더 경로는 `model.language_model.layers.*`이고 `Qwen3_5ForCausalLM`(분리형 판정부)은 `model.layers.*`다. 전체 이름을 동결하면 **판정부에서 매칭이 0건**이 되어 PEFT가 죽거나(운 좋은 경우) 급히 다른 sha256의 목록을 뜨게 된다 — 5칸 공통 고정 2번을 "같다"고 증명할 수단이 사라진다. 더 나쁜 경로: 배경지식 어댑터를 텍스트 클래스에 얹을 때 접두사가 달라 `set_peft_model_state_dict`가 **0개 로드하고 조용히 성공**한다.

**채택**: **2구성요소 접미사 앵커 목록**으로 동결한다.
```
linear_attn.in_proj_qkv | linear_attn.in_proj_z | linear_attn.in_proj_b | linear_attn.in_proj_a | linear_attn.out_proj
self_attn.q_proj | self_attn.k_proj | self_attn.v_proj | self_attn.o_proj
mlp.gate_proj | mlp.up_proj | mlp.down_proj
```
PEFT의 `endswith` 매칭과 정확히 맞고 두 클래스에서 같은 집합을 고른다. **표준 Qwen 레시피(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`)는 32개 층 중 full_attention 8개 층에만 붙고 linear_attention 24개 층을 통째로 건너뛴다** — 부착 128/248, 어댑터 파라미터 21.2M/32.5M, **35%가 에러 없이 사라진다**(레지스트리 #11). `in_proj_*`·`out_proj`가 목록에 반드시 있어야 하는 이유다.

그 위에:
- `assert_adapter_placement`가 **해석된 전체 이름 집합**을 돌려주고 동결 목록과 **완전 일치**(부분집합 아님)를 검사. 부착 모듈 수 248, `visual|vision|merger` 0건
- 교차 테스트 `set(uni_names에서 'language_model.' 제거) == set(sep_names)` — **하나의 sha256이 두 구조를 모두 덮는다는 증명**
- `dump_target_modules.py`에 **"학습 손실 경로 도달 여부"를 forward 후크로 실측하는 열**을 추가한다. `mtp_num_hidden_layers=1`(MTP 헤드)의 projection은 이름·클래스 모두 평범한 Linear라 목록에 들어가기 쉽고, MTP 손실을 안 켜는 우리 학습에서 grad가 영원히 None이다. 죽은 어댑터는 `assert_adapter_placement`·`lora_B norm==0`·payload 바이트 검사를 **전부 통과**한다
- `'all-linear'` 문자열은 이 도구 안에만 존재한다

## 판정 18 — 프로파일 override 금지 검사: 키 부재 vs **diff 화이트리스트**

관점 1·3이 겨눈 표면이 틀렸다. 실물 `configs/gpu/16gb.yaml`·`48gb.yaml`에 `size` 계열 키는 애초에 없고, 대신 **`vlm.max_seq_len_qa: 1024`·`max_seq_len_reason: 2048`·`detection.imgsz: 640`·`per_device_batch`·`grad_accum`·`vlm.load_in_4bit: true`가 들어 있다.** 값 동일성을 강제하는 코드는 없고 주석만 있다(48gb.yaml: "16gb와 동일해야 한다 — 5칸 전처리 고정 조건"). 그리고 `max_seq_len_qa: 1024`는 **레지스트리 #6의 바로 그 숫자**이며, `run_background_sft`가 이 값을 읽으면 5칸 공통 출발점인 배경지식 병합본이 선언의 절반 길이로 학습된다 — 모든 칸이 같은 오염된 출발점을 공유하므로 어떤 교차 칸 검사도 원리적으로 감지할 수 없다.

**채택**:
1. `max_seq_len_qa`·`max_seq_len_reason`을 프로파일에서 **삭제하고 `configs/base.yaml`로 이관**(두 프로파일 값이 같으므로 손실 없음). `vlm` 블록을 `train:`/`eval:`로 쪼개고 `eval:`에는 양자화 키를 두지 않는다(판정 22)
2. 검사를 **프로파일 diff 화이트리스트**로 바꾼다 — 값이 달라도 되는 키를 `{profile, hardware.*, device, per_device_batch, grad_accum, workers, detection_concurrency, vllm_*, driver}`로 명시 열거하고 **그 밖의 모든 키가 값까지 동일**함을 assert. 현재 두 파일은 이 검사를 그대로 통과한다 — 지금 넣으면 공짜다
3. 유효 배치 `per_device_batch × grad_accum` 동일성은 파생 검사
4. `size` 계열 부재 검사는 프로파일뿐 아니라 **칸별 override 5개와 `pyproject.toml [tool.flwr.app.config]`까지** 확장

## 판정 19 — `FIXED_UNI` 검사 대상: "덮으려 한 키" vs **"실제로 쓰일 값"**

detection의 `FIXED_OVERRIDES` 검사는 `set(extra_overrides) & set(FIXED_OVERRIDES)` 교집합뿐이다(실물 L139-147). 통합형에서 그 패턴을 그대로 베끼면 정작 `FIXED_UNI`가 이름을 올린 값들이 `extra_overrides`가 아니라 `PairSpec`(max_seq_len, coord_cfg, prompt_file)·`LoopCfg`(micro_batch, grad_accum, lr, betas, seed)로 들어와 **검사를 지나간다.** 게다가 스펙 §2-2가 명시한 실행 방식이 `flwr run --run-config`로 pyproject를 override하는 것이라 상시 통로가 열려 있다.

**채택**: `train_round` 진입 첫 부분에서(**transformers import 전**) **병합·해석 완료된 설정 전체를 `base.yaml`과 대조**한다.
```python
for k in FIXED_UNI:
    if resolved[k] != base_yaml[k]: raise ValueError(...)
```
`PairSpec`·`LoopCfg`는 자유 생성자를 막고 **`from_fixed(base_yaml, overrides)` 팩토리만 공개**해, `FIXED_UNI` 키에 base.yaml 아닌 값이 들어오면 생성 자체가 실패하게 한다. 무거운 optional 의존성 없이 설정 오류를 알아낼 수 있어야 한다는 detection의 원칙은 그대로 유지한다.

## 판정 20 — 칸 간 대조: 5칸 전수 동일 vs **구조 내 동일 + 필드 확장**

`tracking.mlflow_local.check_cells_identical`이 비교하는 필드는 실물에서 **4개뿐**이다(`base_ckpt_sha256`, `coords_sha256`, `coord_cfg_hash`, `rag_snapshot_sha256`). `prompt_sha256`은 `REQUIRED_TAGS`에 있으면서도 **칸 간 대조 대상이 아니다.** 그래서 통합·중앙과 통합·연합이 서로 다른 프롬프트·전처리·커널·양자화로 돌아도 전부 초록이다.

**채택**: `CellFingerprint`를 확장하고 비교를 **구조 내 동일**로 바꾼다.

| 추가 필드 | 잡는 것 |
|---|---|
| `prompt_sha256` | label_map 변경 후 프롬프트 재생성 드리프트 |
| `preproc_sha256` | 전처리(판정 7) |
| `env_fingerprint_sha` | 커널 백엔드(공통 고정 9번) |
| `runtime_fingerprint` | dtype·양자화·attn 구현(판정 22) |
| `quantized_lm_modules_sha256` | lm_head 양자화 여부(제3부 F4) |
| `realized_adapter_sha256` | 실제 부착 결과(판정 21) |
| `lever_fingerprint` | micro/accum·로짓 최적화 조합(판정 9) |
| `manifest_sha256`·`verdict_mode` | 데이터·모드 |
| `pixels_actual_sha256` | 실제 리사이즈 산출물(판정 17의 호출 override) |

프롬프트는 구조별로 다를 수밖에 없으므로 "5칸 전수 동일"로는 검사 자체가 불가능하다. `uni_central ↔ uni_fed` 바이트 동일, `sep_*` 3칸끼리 동일로 나눈다. **불일치 시 채점 거부.**

## 판정 21 — 어댑터 설정 증명: 선언 파일 해시 vs **실현 산출물 해시**

`lora_cfg_sha256`(yaml 파일 해시)의 "전 셀 단일값 assert"는 단일 프로세스 Ray 시뮬레이션에서 **반증 불가능하다** — 세 클라이언트가 같은 머신의 같은 파일을 읽으므로 정의상 항상 같다. 이 검사가 실패하는 우주는 없다. 반면 조용히 깨지는 경우는 많다: PEFT 버전 상향으로 매칭 규칙이 바뀌어 부착이 248이 아니게 되는 경우, `lora_dropout`·`use_dora`·`use_rslora`·`lora_bias`·`target_parameters`·`trainable_token_indices` 미기재 필드의 기본값이 바뀌는 경우.

**채택**:
```python
realized_adapter_sha256 = sha256(
    json.dumps(peft_config.to_dict(), sort_keys=True) + "\n" +
    "\n".join(sorted(n for n, m in model.named_modules() if isinstance(m, LoraLayer))) + "\n" +
    "\n".join(f"{k}:{tuple(v.shape)}:{v.dtype}" for k, v in sorted(get_peft_model_state_dict(model).items()))
)
```
한 컬럼이 부착 248 assert, 파라미터 수, dtype 사고, PEFT 드리프트를 전부 덮고 **실제로 실패할 수 있다.** yaml 해시는 남기되 "설정 파일 동일성"이라고만 부르고 "어댑터 설정 고정의 증명"이라고 부르지 않는다. `lora_uni.yaml`에는 위 6개 필드를 전부 명시값으로 적는다.

## 판정 22 — 평가 정밀도·양자화 → **bf16 병합본 단일 경로**

어디에도 고정돼 있지 않은데 프로파일에는 이미 `vlm.load_in_4bit: true`가 학습/평가 구분 없이 적혀 있다. 두 위험이 겹친다:
- **학습 4bit 비전 타워 / 평가 bf16 비전 타워** — NF4 블록 양자화 오차의 RMS는 가중치 대비 0.5~1%이고 r=16 LoRA의 `‖ΔW‖/‖W‖`는 통상 1e-3~1e-2다. **base 교체 섭동이 파인튜닝이 만든 변화보다 크거나 같다.** 그리고 이 열화는 통합형에만 걸린다(판정부는 텍스트 입력이라 비전 타워를 안 탄다) → "통합형 grounding이 약하다"의 귀속이 불가능해진다
- **칸 간 갈림** — 통합·중앙만 4bit로 평가하면 좌표가 몇 px 갈리고 `latency_ms`가 2~3배 차이 난다. latency는 RQ2 처리속도 지표라 "구조 비교"가 "로딩 옵션 비교"가 된다

**채택**: (a) **`llm_int8_skip_modules`에 `visual` 포함**(제3부 F4의 합성 방식) — 비전 타워 bf16 유지 비용은 0.64 GB 대 NF4 0.16 GB, **차액 0.48 GB**로 예산 안이다. 이러면 학습과 평가가 같은 비전 타워를 쓴다. (b) 언어부 평가는 **bf16 병합본 단일 경로**. `assert_eval_runtime(model)`이 `model.dtype is bfloat16`, 양자화기 부재, `_attn_implementation`을 **모든 하위 config에서** 실측 assert한다. (c) `configs/gpu/*.yaml`의 `vlm`을 `train:`/`eval:`로 쪼개고 `eval:`에 양자화 키를 두지 않는다 — **어길 수단 제거.** (d) 부록에 층별 `‖W_bf16 − dequant(W_nf4)‖_F / ‖W_bf16‖_F`를 ε′_l과 **같은 축에** 실어 "집계 오차가 양자화 격차 대비 어느 정도인가"를 독자가 판단하게 한다.

---
# 제3부 — 적대 검증 fatal 지적과 처방

9회 적대 검증(3설계 × 3렌즈)에서 **세 설계안 전부 `survives: false`**였다. fatal 등급 지적을 중복 제거해 23건으로 정리했다. 각 건에 **처방**과 **이 명세 어디로 갔는지**를 적는다. 마지막 두 절에 **버린 설계**와 그 이유, **재검증 후 기각한 fatal 주장**을 적는다.

## fatal 23건과 처방

| # | fatal 지적 | 왜 치명적인가 | 처방 | 반영 위치 |
|---|---|---|---|---|
| **F1** | 손실 토큰 평균 × FedAvg `n_k` 가중 → 통합·중앙과 통합·연합이 **수학적으로 다른 목적함수**를 최소화. 클라이언트 k의 토큰이 `1/τ_k` 가중을 받는다 | 모든 게이트가 초록인데 RQ1 회복률이 이 차이로 계산된다. C3 토큰이 0.71배로 눌리면 RQ3 이득이 그만큼 사라진다 | 총합 분모 정규화(판정 2) + `supervised_tokens` 회계 컬럼 + τ 비 1.02 게이트(판정 3) | 판정 2·3, 4-3, 4-19 |
| **F2** | `Qwen3_5ForConditionalGeneration.accepts_loss_kwargs = False`, `forward`가 `**kwargs`를 `loss_function`에 전달하지 않음(L1700-1701, L1826) → VL 클래스는 `num_items_in_batch`를 **구조적으로 무시**. 분리형이 쓸 `ForCausalLM`은 정상 전달 | **같은 체크포인트인데 프레임워크 클래스가 두 구조에 다른 목적함수를 준다.** RQ2 주 기여에 통제되지 않은 교란 | `labels=None` forward + `vlm/loss_norm.py` 단일 공식을 분리형에도 주입 + 교차 테스트 | 레지스트리 #7, 판정 2, 4-3 |
| **F3** | 손실 shift가 우리 책임이 되는데 off-by-one이 **조용히** 실패한다 — 손실이 0으로 붕괴하며 학습은 끝까지 돈다 | `stop_reason='budget'`, 증인 일치, 회계 전 셀 통과, `AuditReport.ok=True`. 실패는 평가에서 바닥 점수로만 나타나고 **"통합형이 분리형보다 나쁘다"가 논문 결론이 된다** | 손실 일치 테스트의 **reference를 프레임워크 자체 경로**로 못 박음 + 라벨 한 칸 밀린 배치에서 **어긋나는지**도 확인 + 첫 50 step loss가 `log(vocab)=12.4`에서 시작하는지 assert, 200 step 내 0.05 미만이면 **run 실패** | 판정 11, 4-15, G8 |
| **F4** | `BitsAndBytesConfig.llm_int8_skip_modules`를 **명시하는 순간** transformers 기본 보호(lm_head·tied·output embeddings)가 통째로 폐기(`quantizers/base.py` L233-247, `quantizer_bnb_4bit.py` L129-131이 `add_default_skips`를 안 넘김) | `["visual"]`을 적으면 vocab 248,320×2,560 = **635.7M 출력층이 NF4로 눌린다.** `None`으로 두면 비전 타워 전체가 NF4가 되어 카나리아가 검증한 좌표 규약이 본학습 모델의 규약이 아니게 된다. **"5칸이 같은 sha256에서 출발한다"가 거짓이 된다** | skip 목록을 손으로 적지 않고 **합성**: `get_keys_to_not_convert(meta_model) ∪ {"visual"}`. 로드 후 `Linear4bit` 집합 sha256(`quantized_lm_modules_sha256`) + `lm_head` 비양자화 + tie 유지(`data_ptr` 동일) + `visual.*` Linear4bit 0건 assert. **`gate_key`에 포함** | 레지스트리 #8, 4-10, G5, 판정 20·22 |
| **F5** | `BitsAndBytesConfig` 실제 인자명은 `bnb_4bit_*`. `quant_type=`·`use_double_quant=`·`compute_dtype=`는 **예외도 경고도 없이 무시**되고 fp4 / fp32 compute / double-quant 없음으로 돈다 | 논문 방법 절의 "NF4"가 사실과 다르고, fp32 compute가 활성 메모리를 배로 만들어 F7의 예산을 무너뜨린다. `base_quant_digest` 단일값 assert는 **"전 셀이 똑같이 틀렸다"를 증명할 뿐** | `configs/quant_uni.yaml` 5필드 명시 + `quantization_config.to_dict()` 키 집합 **완전 일치** 대조(교집합 아님) + 실체 3종 assert(`quant_state.quant_type`, `base_layer.compute_dtype`) | 레지스트리 #9, 4-4, 4-10 |
| **F6** | `from_pretrained`의 `dtype` 기본값이 v5에서 `"auto"`(v4는 fp32) → 비양자화 2.4 GB의 정밀도를 리포 `config.json`에 위임 | 리포가 갱신되면 +2.4 GB로 조용히 OOM, 반대로 두 칸이 다른 리비전을 타면 정밀도가 갈리는데 `base_ckpt_sha256`은 동일하게 찍힌다. dtype 히스토그램을 로깅해도 **선언값이 없으면 대조 대상이 없다** | `QLoraSpec.dtype='bfloat16'` 명시 + `declared_dtype` vs 히스토그램 대조 + 비양자화 파라미터 중 bf16 아닌 것 1건이라도 있으면 실패. `base_ckpt_sha256`에 `config.json` 포함 | 레지스트리 #10, 4-10 |
| **F7** | 로짓 메모리 추정이 3~4배 과소. vocab 248,320(가정 "약 15만"), `ForCausalLMLoss`가 **무조건 `logits.float()`**, lm_head·embed 비양자화 각 1.19 GB, 가용은 16이 아니라 **14.72 GB** | "micro=2로 시작", "통합·중앙은 16GB에서 돈다"가 실측 이전에 이미 틀렸다. 동적 패딩이라 2048 토큰 샘플이 처음 뽑히는 스텝에서 **비결정적으로** 터진다 | micro=1/accum=32(판정 10), 총괄 위치 한정 로짓을 **착수 전제로 승격**(판정 11), 길이 버킷팅, G7-B1 최악 배치 프로브 | 판정 10·11, G7, 9-3 |
| **F8** | 언어부 32층 중 **24층이 linear_attention**인데 Windows sm_120에 고속 커널이 없다(triton 부재, `causal-conv1d`·`fla` Windows 휠 0). 폴백은 fp32 캐스트 + `for i in range(1, chunk_size)` 63회 파이썬 루프 | 실측: linear 0.1509 s/층 대 full 0.0611 s/층(**2.47배**). 32층 4.11 s/마이크로배치 × accum 32 = **132 s/옵티마이저 스텝** → 통합 3종 합계 **15 GPU-일**(bf16 하한), NF4·비전·CE 포함 19~21일. **GPU-일은 "아직 모르는 값"이 아니라 이미 계획을 무효화하는 값이다** | **착수 전 백엔드 확정**(G7-B5): `triton-windows` + `fla-core` 시도 → 되면 `env_fingerprint`에 편입, 안 되면 폴백 확정 사실과 총 GPU 시간을 근거로 48GB 이관을 **파일럿 이전에** 결정. 어느 쪽이든 통합 2칸은 같은 백엔드에서 돌고 이관 시 둘 다 재실행 | G7, 판정 22, Q11·Q12, 9-5 |
| **F9** | PEFT 접미사 매칭 + 표준 Qwen 레시피 → **linear_attention 24개 층 전부 누락**. 부착 128/248, 어댑터 파라미터 21.2M/32.5M, **35%가 에러 없이 사라진다** | 학습은 정상적으로 돌고 loss도 내려간다. 어댑터 실효 용량이 선언보다 작은데 지표에 흔적이 없다 | `in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj` 포함한 접미사 앵커 목록(판정 17) + 248 완전 일치 + `realized_adapter_sha256` + **forward 후크로 손실 경로 도달 여부 실측**(죽은 MTP 어댑터 탐지) | 레지스트리 #11, 판정 17·21, G4 |
| **F10** | buffer 게이트가 **구조적으로 실패할 수 없다.** 학습 모드 + `use_cache=False`면 5스텝 diff는 반드시 0. 게다가 `state_dict ∩ named_buffers` 정의는 bnb quant state를 못 본다(`register_buffer` 없이 state_dict 직접 주입) | **통과할 수밖에 없는 게이트를 통과시키고 논문에 사실 주장을 싣는 것이 게이트가 없는 것보다 나쁘다.** `base_quant_digest`가 rope 상수 3개의 해시가 된다 | P/B/X 3집합(`remove_duplicate=False`) + **교환 폐포 등식**(G2-3) + 라운드 1 비어댑터 키 bit-exact(G2-5) + `base_quant_digest` 재정의 + `n_tensors_examined` 보고 | 레지스트리 #12·#13, **G2** |
| **F11** | `S_k`의 분모 `n_k`와 감사 규칙 ④의 분모가 **같은 counts.json**이라 감사가 3.000을 반드시 출력하는 **항진명제** | 폐기율이 클라이언트마다 다르면(C1 0.2% / C3 3.0%) 전역 상한 1%를 통과하면서 실현 노출이 갈리고, 하필 C3가 손해를 본다. 어느 지표에도 흔적이 없다 | `n_k_declared`/`n_k_effective` 이원화(판정 4) + G6에서 `train_index` 사전 봉인 + **클라이언트별** 폐기율 상한 + 감사 ④′④″④‴ | 판정 4, G6, 4-14, 4-19 |
| **F12** | 라운드마다 시드를 다시 심는 fresh Trainer가 **같은 순열**을 낸다(실행 확인: 3회 모두 `[28563, 2910, 12526, …]`). E=0.5면 데이터의 **앞 절반만 6번** 본다. Q7의 라운드별 재셔플도 1.56%를 통째로 빠뜨린다 | `R·S_k·B_eff/n_k`는 정확히 3.0이라 감사 통과. 분리형 검출은 정수 epoch이라 100% 커버리지 → **"구조 차이"로 보고할 숫자가 데이터 커버리지 차이가 된다** | 단일 가상 순열 + 전역 커서(판정 8) + `sample_index_digest`·`exposure_min/max`·`coverage` 감사 | 레지스트리 #20, 판정 8, 4-15, 4-19 |
| **F13** | 글로벌 어댑터 초기값을 서버가 손으로 만들면(`A=kaiming, B=0`) 통합·중앙의 PEFT 기본 초기화와 갈린다. `kaiming_uniform_(a=sqrt(5))`와 `a=0`은 **초기 A 스케일이 2.449배** 차이 | `B=0`이라 첫 스텝 `∂L/∂B ∝ A`이므로 어댑터가 움직이는 속도가 2.449배 다르다. 유일한 검사(`B norm==0, A norm>0`)는 **A를 어떤 분포로 뽑아도 통과** | 초기값을 손으로 만들지 않는다 — 시드 고정 후 `get_peft_model(base, LoraConfig(**lora_uni.yaml))`을 meta/CPU에서 1회 만들어 `get_peft_model_state_dict()`를 그대로 내보낸다(**통합·중앙과 문자 그대로 같은 코드 경로**) + `outputs/gates/adapter_init_{cell}.json` **바이트 동일** assert | 4-18, 4-19 |
| **F14** | 전처리에 대한 **교차 칸 증인이 없다** — `check_cells_identical`이 4필드만 보고 `prompt_sha256`조차 대조 대상이 아니다 | 통합 2칸이 서로 다른 입력 해상도·프롬프트로 학습됐는데 선언된 모든 게이트가 초록이고, RQ2 통합형 열이 내부적으로 비교 불가가 된다 | `CellFingerprint` 9필드 확장 + 구조 내 동일 대조(판정 20). **단 "`coord_cfg_hash`가 리사이즈 키에 눈이 멀었다"는 실물 재검증 결과 부분 기각**(판정 7·아래 재검증 절) | 판정 7·20 |
| **F15** | 이미지 토큰 상한이 어디에도 고정돼 있지 않다. 배포 `size.longest_edge = 16,777,216`은 이미지 1장에 최대 **16,384 비전 토큰**을 허용 | S=16,384면 로짓 스택만 **40.68 GB** — 즉시 OOM. 반대로 3000×1000 RT 필름은 2,930 비전 토큰이라 `max_seq_len` 2048 필터가 데이터를 대량 폐기한다. `size.*`는 5칸 공통 고정이라 **한 칸을 돌린 뒤 낮추면 그 칸이 무효** | `size.longest_edge = 1,048,576`(4,096 패치 = **1,024 비전 토큰**) 사전 등록, `max_seq_len` 2048 유지 → 텍스트 예산 1,024. 결정 자체를 코드로(`pick_longest_edge.py`가 매니페스트 전수를 실제 `smart_resize`에 통과) — **`ceil` 식은 banker's rounding 때문에 약 5% 과대 추정**(레지스트리 #21) | Q16, G3-P6, 4-7 |
| **F16** | 역변환에 `quantize`를 넣으면(`to_px → quantize`) **`roundtrip_budget_px`를 구조적으로 초과**한다. 실측: 1280×720, GT `x1=2` → 오차 1.0px 대 예산 0.64px. W=1600에서 정수 x값의 37.5%가 위반 | **G6(학습 스텝 0 이전 하드 게이트)가 정상 코드에서 통과 불가능**해지고, 자연스러운 처방은 "예산을 늘린다" — 좌표계 붕괴를 잡는 유일한 사전 게이트가 무력화된다 | 역변환 정수화 **0회**(판정 5). `bbox_px`는 float. 계약 #4가 이미 `tuple[float,…]`이고 D §3-4가 "정수화는 어댑터 경계 최종 1회"로 확정했다 | 판정 5, 4-6, 4-20 |
| **F17** | `truncated_but_parsable`로 `max_new_tokens`를 정한다고 했는데 그 카운터를 내는 유일한 진입점이 **eval split을 읽는 `run_eval`**이고, `limit=None` 부분 실행 손잡이까지 있다 | 평가셋으로 5칸 공통 고정 항목(디코딩)을 튜닝하는 경로가 설계 안에 완비. `eval_access.json`이 덮어쓰기라 **사전 접근 기록이 사라진다.** 불변조건 1-4 위반 = 논문 철회 사유 | `run_eval`에서 `limit` **제거**. 예산 파일럿은 생성자에서 `split=='eval'`을 막는 `vlm/pilot_budget.py` 전용. `eval_access.jsonl` **append-only 해시 체인**(2줄 이상이면 D가 채점 거부). `max_new_tokens`를 `gate_key`에 포함 | G9, 4-20 |
| **F18** | `resized_h = grid_h × patch_size` 가정을 **어떤 검사도 반증할 수 없다.** selfcheck도 오라클 주입도 encode/decode에 같은 geom을 넣으므로 merge_size 배 틀려도 왕복 오차 0 | grid가 merged 단위면 복원 resized가 2배 어긋나 카나리아 ABS_RESIZED 후보가 IoU 0 근처로 깔리고, 세 후보 전부 실패 시 **모델 후퇴 경로(8/21 확정 "동일 체크포인트 공유" 위반)가 발동**한다 | G3-P2·P3: `pixel_values.shape[0] == t*h*w` **and** `image_token_id` 개수 == `t*h*w // merge_size**2`. 프로세서 실호출 교차검증만이 이걸 잡는다 | G3, 4-6 |
| **F19** | `llm_int8_skip_modules=None`이면 비전 타워 98개 Linear이 전부 NF4. 그런데 `merge_final`은 bf16 base | NF4 오차 RMS 0.5~1% > LoRA `‖ΔW‖/‖W‖` 1e-3~1e-2 → **base 교체 섭동이 파인튜닝 변화보다 크다.** 통합형에만 걸려 "통합형 grounding이 약하다"의 귀속이 불가능 | `visual` skip 명시(F4 합성 방식, 차액 0.48 GB) + bf16 병합본 단일 평가 경로 + 양자화 격차를 ε′_l과 같은 축에 부록 수록(판정 22) | 판정 22, 4-10, Q17 |
| **F20** | `generation_config.json` 부재를 "위험 없음"으로 읽었다. 이 파일이 없다는 것은 **벤더가 선언한 eos_token_id 목록이 없다**는 뜻이기도 하다. `add_generation_prompt` 기본값도 **False** | eos가 `<|endoftext|>` 단일 id면 모델이 JSON을 끝내고도 안 멈춰 전 샘플이 `length` → "length면 truncated 오답" 규칙이 **28,000 레코드 전건을 오답으로** 만든다. `add_generation_prompt` 누락이면 생성문이 빈 문자열. 두 경우 모두 **"통합형이 구조적으로 열등하다"와 구별되지 않는다** | eos를 chat template **차분에서 유도**해 집합으로 넘기고 `terminator_ids ⊆ gc.eos_token_id` assert. `stop_reason`을 길이가 아니라 **마지막 토큰 정체**로 판정. `assert_generation_prompt`(두 렌더 차분 비어있지 않음 + 실사용이 그 접미사로 끝남). **G5 통과 조건 5번 `eos_stop_rate ≥ 0.9`**, G9에서 `== 1.0` | G5, G9, 4-12, 4-20 |
| **F21** | "greedy 1회라 백엔드가 바뀌어도 출력 동일"이 SSM 하이브리드에서 성립하지 않는다. `use_kernel_func_from_hub_with_fallback`이 `try/except`로 **로그도 경고도 없이** 커널을 교체하고, `self.norm`은 fla 설치 여부에 따라 **클래스 자체가 바뀐다**. `_attn_implementation`은 32층 중 **8층만** 지배 | batch=1을 "패딩이 argmax를 뒤집는다"로 고정하면서 훨씬 큰 교란인 커널·GPU 교체는 열어 뒀다. 통합 2칸이 다른 환경에서 돌면 차이의 일부가 **"연합 때문"이 아니라 "커널 때문"**인데 MLflow 태그는 동일하다 | **`env_fingerprint`를 5칸 공통 고정 9번째 항목으로 승격**(아래 표) + `check_cells_identical` 대조 필드 + 이관 시 통합 2칸 동시 이관·둘 다 재실행 + 학습 풀 고정 50장 재현 카나리아(두 환경 생성문 바이트 동일) | 레지스트리 #14, 판정 21·22, 4-10 |
| **F22** | 28,000장 batch=1 단일 프로세스에 **재개 경로가 없는데** 평가셋 접근은 전 실험 통틀어 1회다. 3 s/장이면 23시간, 8 s/장이면 62시간 연속 실행 | 크래시하면 전량 재실행이고 그것이 두 번째 eval 접근이다. **실무에서 유일하게 실행 가능한 선택지가 "`eval_access.json`을 조용히 덮어쓴다"가 되어 격리 증거가 위조된다** | 이미지마다 append+flush + `progress.json`(6해시) + 종료 시 image_id 정렬 정규화본 rewrite. `eval_access.jsonl`을 **재개가 표현 가능한 사실**로 만든다(`reason: initial|resume`). 재개 시 6해시 불일치면 거부. 학습 루프도 K 스텝마다 재개 체크포인트(조기 종료도 best도 아니다 — val을 아예 안 읽는다) | 4-15, 4-20, G9 |
| **F23** | flwr **1.33.0**(설치본, 트랙 C 실측 정정) Message API `FedAvg`에 **`accept_failures`가 없고** `aggregate_train`이 에러 응답을 버린 뒤 남은 것으로 집계. `fraction_evaluate` 기본 1.0 | 라운드 4에서 C1이 OOM으로 죽어도 서버가 2클라이언트로 재정규화해 진행하고 정상 종료한다. R×E=N이 깨졌는데 전략이 초록 | 전략이 직접 검사(판정 14) + `fraction_evaluate=0.0`·`min_evaluate_nodes=0` 하드코딩 + `aggregate_evaluate` 대칭 방어 + 회계 빈 셀 = run 무효 | 레지스트리 #19, 판정 14, 4-19 |

## 버린 설계와 이유

| 버린 것 | 어느 관점 | 왜 버렸나 |
|---|---|---|
| TRL `SFTTrainer` 채택(스펙 §2-1 문면) | 스펙 | 숨은 기본값 표면이 두 겹, 라운드 무상태와 반대 방향, VL 클래스가 손실 kwargs를 무시(F2), fresh Trainer 샘플러(F12) → 판정 1 |
| `optimizer.state[p]['step']`을 **단일** 독립 증인으로 | 관점 1 | torch는 **grad를 받은 파라미터에만** state를 만든다(`adam.py` L151의 `if p.grad is not None` 안에서 초기화). 죽은 어댑터가 전 검사를 통과한다. → 증인 4종 + 전 학습 파라미터 전수 검사 + state **부재**를 별도 실패 사유로 |
| `audit_buffers(probe_steps=5)` 이름 diff 게이트 | 관점 2 | 구조적으로 실패할 수 없다(F10) → 교환 폐포 등식(G2) |
| `micro_batch↔grad_accum`을 레시피 중립 레버로 | 관점 1 | LoRA dropout 난수 소비 + 로짓 최적화 의미 변화 → 판정 9(dropout 0.0, micro=1 고정, `lever_fingerprint`) |
| 시작값 micro=2 / accum=16 | 관점 1 | vocab 248,320 + fp32 upcast 실측이면 확정 OOM(F7) → 판정 10 |
| `max_pixels`를 OOM 사다리에 포함 | 초기 스펙 | 검토 결정 C로 이미 금지. 사다리에서 제거되어 있고 이 명세도 유지 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 관점 1 | Windows에서 **무동작**(`CUDAAllocatorConfig.h` L35-45가 무조건 false) → `garbage_collection_threshold:0.8,max_split_size_mb:512` + 길이 버킷팅(레지스트리 #15) |
| 서버가 손으로 만드는 글로벌 어댑터 초기값 | 관점 2 | PEFT 기본 초기화와 스케일이 2.449배 갈릴 수 있다(F13) → PEFT 경로 재사용 + 바이트 동일 게이트 |
| 어댑터 wire의 bf16↔fp32 캐스트 2지점 | 관점 2 | 측정 노이즈(2e-3)가 측정 대상(집계 오차 2차항, ~1e-3)보다 크다 → 판정 13(fp32 전 구간, 캐스트 0지점) |
| `assert_expected` 바이트 **정확 일치** 단일 게이트 | 관점 2 | flwr npy 헤더 496×128 B 때문에 런이 시작되지 않는다 → 판정 15(2열 분리) |
| buffer digest를 warm actor 탐지기로 | 관점 2 | 정의상 항상 상수라 발동하지 않는다(판정 12) → 3종 실측 |
| `milestone_rounds(6, (.25,.5,.75)) = (2,3,5)` | 관점 2 | **최종 라운드 페이로드 보존 경로가 없어** 부록 1단의 '최종' 행과 2단 전체가 산출 불가 → `∪ {R}` 강제, 저장 1.56 GB(이미 예산에 있음) |
| dense `ΔW` SVD | 관점 2 | 랭크가 정의상 ≤48인데 층당 94 MB·248층 2~4시간 → QR 코어(판정 16) |
| 역변환 `quantize`("정수화 각 방향 1회") | 관점 3 | G6가 정상 코드에서 통과 불가(F16) → 정수화 0회(판정 5) |
| 카나리아에 면적<65,536 업스케일 **필수** | 관점 3 | 축퇴 전제가 실측과 다르다(하단 소형 도형 혼동 IoU 0.156). 파싱률 미달로 게이트를 막을 위험만 크다 → 선택으로 강등, 하단 소형 도형이 필수 |
| `render_prompt_sha` 상수성 **단일** assert | 관점 3 | 이미지 pad 확장 때문에 2번째 샘플에서 죽거나(런 중단), 템플릿 적용 문자열이면 항상 통과해 아무것도 못 잡는다 → 2값 분리(4-8) |
| `parse_error` 어휘 "문자열까지 동일" 테스트 | 관점 3 | 실물 `ParseError`는 7개, target_format은 4개라 즉시 빨간불이고 통과시키려면 통합형이 낼 수 없는 어휘(`join_missing`)를 넣게 된다 → **부분집합 + 신규 문자열 0** |
| `run_eval(..., limit=None)` | 관점 3 | 평가셋 부분 실행 손잡이는 격리와 양립 불가(F17) → 제거 |
| 프로파일 yaml "키 부재" 검사 | 관점 1·3 | 실물이 `max_seq_len_qa`를 갖고 있어 **공허하게 통과**한다 → diff 화이트리스트(판정 18) |
| `model.merged_adapters == []` 가드 | 관점 2 | `BaseTunerLayer` 클래스 속성이라 `PeftModel`에서 AttributeError → `get_model_status().merged_adapters`(레지스트리 #22) |

## 재검증 후 기각·강등한 fatal 주장 2건

**합의가 정답의 증거가 아니듯, 적대 지적도 그 자체로 참이 아니다.** 실물 대조로 확인했다.

1. **"`CoordCfg`에 `size.shortest_edge`/`longest_edge`가 없어 `coord_cfg_hash`가 눈이 멀었다" → 부분 기각.** `vlm/coords.py` L72-105 실물에 `min_pixels`·`max_pixels`·`factor`가 있고 `canonical_json`이 `asdict` 전체를 해시한다. 두 표기 모두 **면적 단위**이므로 `coord_runtime`의 고정 매핑(`shortest_edge→min_pixels`, `longest_edge→max_pixels`)만 지키면 시나리오가 성립하지 않는다. **남는 진짜 구멍**(`max_seq_len` 미포함, 호출 시점 kwargs override)만 `preproc_sha256`과 `encode_inputs` 단일 경로로 막는다(판정 7·17). `coords.py`는 수정하지 않는다.
2. **"`state_dict − named_parameters`가 `lm_head.weight`를 buffer로 오탐하니 그 컬럼을 버려야 한다" → 강등.** 오탐의 원인은 컬럼이 아니라 `remove_duplicate=True` 기본값이다. `remove_duplicate=False`로 빼면 tied 별칭은 정확히 제외되고 **나머지 비파라미터 state_dict 항목(bnb quant state)은 남는다** — 그 컬럼이 양자화 산출물을 볼 수 있는 유일한 창이므로 버리면 안 된다(레지스트리 #12·#13, G2).

## 5칸 공통 고정 — 8종에서 **9종**으로

불변조건 3-3의 8종에 **실행 환경**을 9번째로 추가한다. F21이 근거다 — 커널 백엔드가 바뀌면 greedy여도 출력이 갈리는데 기존 8종 어디에도 그 축이 없다.

| # | 항목 | 통합형에서의 실체 | 증인(태그) | 갈리면 |
|---|---|---|---|---|
| 1 | 기본 모델 체크포인트 | 배경지식 SFT 병합본. `config.json` 포함 해시 | `base_ckpt_sha256`, `base_quant_digest`, `quantized_lm_modules_sha256` | 채점 거부 |
| 2 | 어댑터 설정 | r=16, α=32, **dropout 0.0**, 접미사 앵커 248 모듈, `modules_to_save=None`, `use_rslora/use_dora=False` | **`realized_adapter_sha256`**(선언 파일 해시 아님) | 채점 거부 |
| 3 | 최적화 | AdamW, lr·betas·wd·warmup·단일 cosine, 유효 배치 32, `max_grad_norm`(+`error_if_nonfinite=True`) | `effective_train_config`, `grad_norm_max`·`clip_hit_count` | run 무효 |
| 4 | 전처리 | `size.shortest_edge/longest_edge`(base.yaml 단독), `max_seq_len` 2048, 이미지 로드 규칙 | **`preproc_sha256`**, `pixels_actual_sha256` | 채점 거부 |
| 5 | 프롬프트 | `unified_v1.txt` + chat template + 표기 규칙 | `prompt_sha256`(= 3자 번들 해시) | 채점 거부 |
| 6 | 검색 설정 | 통합형은 검색 없음 — 그 자체가 대비축. `rag_snapshot_sha256`은 provenance로 기록 | `rag_snapshot_sha256` | — |
| 7 | 디코딩 | greedy 1회, `num_beams=1`, 샘플링 노브 None, `max_new_tokens` 동결, **`batch_size=1`**, eos 집합 | `gen.effective_sha256`, `gen.eos_token_ids` | 채점 거부 |
| 8 | 시드 | `derive_seed(base, r, k) = base + 10007r + 101k`(`fl/seeding.py` 단일 소유), 통합형 1시드 | `seed`, `sample_index_digest` | run 무효 |
| **9** | **실행 환경** | GPU·torch·transformers·peft·bnb 버전, triton/fla/causal-conv1d 가용성, 실효 커널 바인딩, `layer_types` 집계, `_attn_implementation` **전 하위 config** | **`env_fingerprint_sha`**, `runtime_fingerprint`, `lever_fingerprint` | **통합 2칸 재실행** |

---
# 제4부 — 모듈별 구현 명세

**공통 규칙 4개.**
1. `vlm/` 아래 어떤 모듈도 **최상단에서 `transformers`·`torch`·`peft`·`bitsandbytes`를 import하지 않는다.** 함수 안 지연 import + 덕타이핑(detection이 `fed_trainer.py` 하나에 ultralytics를 가둔 것과 같은 배치). 현 `.venv`에 이 패키지들이 없어도 기존 760건 스위트가 계속 돌아야 하고, 새 테스트도 가짜 프로세서 객체로 오프라인 검증돼야 한다.
2. **모든 파일 IO는 `encoding="utf-8", newline="\n"` 명시.** 이 머신 실측 `locale.getpreferredencoding(False) == 'cp949'`, 전역 `core.autocrlf=true`, `.gitattributes`는 `* text=auto eol=lf`. 인자 없는 `open`/`read_text`/`write_text`는 AST 테스트로 금지한다 — 한국어 프롬프트가 해시는 통과하고 내용만 모지바케되는 경로가 실재한다.
3. `vlm/coords.py`와 `vlm/coords_fixtures/`는 **손대지 않는다.** 착수 전·후 `python vlm/coords_fixtures/run_fixtures.py` 12/12 통과와 `coords.py` sha256 불변을 확인한다.
4. 선언이 아니라 **실사용을 기록하고, 불일치를 실패로 승격한다.**

---

## 4-1. `vlm/target_format.py` — 표기의 단일 소유자 (리프, stdlib 전용)

**목적**: 학습 타깃 직렬화와 추론 출력 파싱을 **같은 파일**이 소유한다. 좌표계 붕괴의 절반은 변환 산식이 아니라 "타깃을 쓴 코드와 출력을 읽는 코드가 다른 파일에서 각자 진화하는 것"에서 나온다. `coords.py`가 `to_model`/`to_px`를 한 파일에 둔 것과 같은 논리를 표기 형식에 적용한다.

```python
TARGET_FORMAT_ID = "json_bbox2d_v1"
TARGET_KEYS  = ("defects", "verdict", "cited_clauses")
DEFECT_KEYS  = ("iso_code", "bbox_2d")
VERDICTS     = ("합격", "불합격", "판정불가")
TARGET_PARSE_ERRORS = ("no_json", "json_decode", "schema_violation", "bbox_invalid")

def serialize_target(pred: ModelPrediction) -> str
def extract_json_block(text: str) -> tuple[str | None, str | None]
def parse_prediction(text: str) -> tuple[ModelPrediction | None, str | None]
def format_sha256() -> str
```

**핵심 결정**
- 타깃 JSON은 `{"defects":[{"iso_code":"2011","bbox_2d":[x1,y1,x2,y2]}],"verdict":"불합격","cited_clauses":[...]}` 단일 형태. `json.dumps(..., ensure_ascii=False, separators=(",",":"), sort_keys=False)` 고정.
- **추출은 관대하게, 검증은 엄격하게.** 코드펜스·선행/후행 산문 제거는 중괄호 카운팅으로 첫 번째 균형 최상위 객체를 취하고, 그 뒤 필드 검증은 **어떤 보정도 하지 않는다**(계약 #4 §2-4를 코드 경계로 구현한 지점).
- `parse_error` 어휘는 계약 #4 `ParseError`의 **부분집합**이어야 한다(동일이 아니다). 실물 `ParseError`는 7개이고 그중 `unknown_iso_code`는 `postprocess` 소유, `join_missing`·`truncated`는 다른 계층이다. 통합형이 낼 수 없는 3개를 **명시 제외 목록**으로 코드에 적어 의도를 문서가 아니라 코드에 남긴다.
- **모델 좌표(`bbox_2d`)는 이 모듈의 dataclass 안에서만 존재한다. 디스크로 나가는 함수를 제공하지 않는 것이 "모델 좌표 저장 금지"의 구현이다.**
- **D가 이 모듈을 import한다.** 계약 #4 §2-4가 D 어댑터를 추출 주체로 적어 두어 추출기가 두 벌이 될 위험이 있다. stdlib 전용 리프로 만들어 그 경로를 구조적으로 없앤다. 이행 전까지는 C가 파싱 결과를 레코드에 함께 실어 D가 동등성 assert를 할 수 있게 한다.

**테스트** `tests/test_vlm_target_format.py`

| 테스트 | 검증 대상 |
|---|---|
| `test_리프원칙_stdlib만_import한다` | AST |
| `test_serialize_parse_왕복_항등` | |
| `test_추출관대성_코드펜스_선행산문_중첩중괄호_후행텍스트` | |
| `test_최상위객체가_2개면_첫번째를_취한다` | 규칙 고정 |
| `test_검증엄격성_여분키_enum위반_bbox원소수` | 보정 0회 |
| `test_parse_error_어휘가_계약_ParseError의_부분집합이다` | 동일이 아니라 부분집합 |
| `test_통합형이_낼수없는_어휘가_제외목록에_명시돼있다` | |
| `test_VERDICTS가_evaluation_schema_Verdict와_같다` | pydantic import 없이 테스트가 보증 |
| `test_디스크_쓰기_API가_없다` | AST |

---

## 4-2. `fl/seeding.py` — 시드·스텝 예산 공식의 단일 소유자 (신규)

**목적**: 파생 시드와 스텝 예산 공식을 검출·VLM·평가·논문이 각자 적으면 그 순간 두 벌이 된다.

```python
def derive_seed(base_seed: int, round_idx: int, client_idx: int) -> int   # base + 10007r + 101k
def plan_steps(n_k_effective: int, effective_batch: int = 32, epoch_frac: float = 0.5) -> int
def realized_epoch_equiv(n_k_effective: int, effective_batch: int, rounds: int, epoch_frac: float) -> float
def milestone_rounds(R: int, fracs: Sequence[float]) -> tuple[int, ...]
```

**핵심 결정**
- `derive_seed`를 여기로 옮기고 `detection.round_runner`는 **재수출만** 남긴다(기존 import 경로·테스트 불변). 두 모듈이 같은 값을 내는지 확인하는 테스트를 함께 둔다.
- ```python
  plan_steps = max(1, int(math.floor(epoch_frac * n_k_effective / effective_batch + 0.5)))
  ```
  **파이썬 내장 `round`를 쓰지 않는다.** `vlm/coords.py`의 `quantize`가 바로 그 이유로 내장 `round`를 배제하고 `floor(v+0.5)`를 정본 규칙으로 선언해 놓았다("banker's rounding이라 0.5가 값에 따라 위아래로 갈린다. 채점 재현성이 서브픽셀 정밀도보다 우선"). `0.5·n_k/32 = n_k/64`이므로 `n_k ≡ 32 (mod 64)`인 클라이언트에서 정확히 x.5가 나오고, 논문 담당이 `numpy.round`나 `math.floor(x+0.5)`로 독립 재계산하면 논문 수치와 실제 run이 어긋난다.
- **분모는 `n_k_effective`**(판정 4). 감사가 학습과 같은 함수를 부르는 것은 증명이 아니므로, 진짜 방어선은 **잔차 검사(`≤ R·K/2 = 9`)**임을 문서에 명시한다.
- `milestone_rounds`는 **최종 라운드를 무조건 포함**: `tuple(sorted(set([ceil(R*p) for p in fracs] + [R])))`. R=6 → (2,3,5,6). 이게 없으면 부록 1단의 '최종' 행과 2단 전체가 산출 불가다.

**테스트** `tests/test_fl_seeding.py`
- `test_detection과_fl이_같은_시드를_낸다`
- `test_plan_steps가_banker_rounding을_쓰지_않는다` — `n_k ∈ {7968, 8032}` 경계 케이스
- `test_plan_steps가_coords_quantize와_같은_규칙이다`
- `test_최종라운드가_항상_마일스톤에_포함된다`
- `test_잔차가_R곱K나누기2를_넘지_않는다`

---

## 4-3. `vlm/loss_norm.py` — 손실 정규화의 단일 소유자 (리프)

**목적**: 판정 2. 통합형·분리형·감사·테스트가 이 함수 하나만 부른다. 수식이 두 벌이 되면 RQ2가 오염된다.

```python
def supervised_token_count(labels: "Tensor") -> int          # shift 후 카운트
def shift_for_causal(logits, labels) -> tuple["Tensor", "Tensor"]
def ce_sum_and_count(logits, labels, *, ignore_index: int = -100) -> tuple["Tensor", int]
def normalize(ce_sums: Sequence["Tensor"], counts: Sequence[int]) -> "Tensor"
def denominator_rule() -> str                                 # "shift_labels"
```

**핵심 결정**
- 분모는 **`(labels[..., 1:] != -100).sum()`**. shift 전에 세면 배치 내 최장 샘플마다 1씩 과대해지고 배치마다 방향이 달라 G8의 1e-5 테스트가 재현 없이 실패한다(`trainer.py` L2155 주석이 같은 함정을 인정한다).
- torch는 함수 안에서만 쓰고 카운트 계산은 순수 함수로 분리해 GPU 없이 테스트한다.
- 분리형 판정부 주입 경로를 함께 제공한다(`compute_loss_func` 또는 수동 정규화).

**테스트** `tests/test_vlm_loss_norm.py`
- `test_분모가_shift_후_카운트다` — 마지막 위치가 감독인 샘플에서 1 차이가 나는지
- `test_accum_kxb가_batch_kb와_1e-5_이내다` — **불균형 케이스 필수**
- `test_기준경로가_프레임워크_loss다` — reference == `model(**batch, labels=...).loss`
- `test_라벨을_한칸_밀면_기준과_어긋난다` — 방어선 작동 증명
- `test_통합형과_분리형이_같은_함수를_쓴다` — 교차 테스트

---

## 4-4. 설정 파일 — `configs/base.yaml` · `quant_uni.yaml` · `lora_uni.yaml` · `target_modules_uni.json`

**`configs/base.yaml`** (신규, A와 공동 소유)
```yaml
vlm:
  coord:      {coord_space: NORM_1000, norm_denominator: 1000, target_format: json_bbox2d_v1}
  processor:  {patch_size: 16, merge_size: 2, factor: 32,
               size: {shortest_edge: 65536, longest_edge: 1048576}}   # Q16 사전 등록
  max_seq_len: 2048
  max_seq_len_qa: 2048          # 프로파일에서 이관 (판정 18)
  max_seq_len_reason: 2048
  effective_batch: 32
  micro_batch: 1                # 판정 10
  grad_accum: 32
  generation: {batch_size: 1, attn_implementation: sdpa, max_new_tokens: null}  # G9에서 동결
  prompt:     {file: vlm/prompts/unified_v1.txt}
  canary:     {n_images: 19, seed: 0, min_median_iou: 0.5, min_parse_rate: 0.9,
               min_eos_stop_rate: 0.9, min_margin: 0.15,
               max_confusion_iou: 0.5, min_separable_images: 8}
```
- **여기 적힌 숫자는 선언이고 코드가 쓰는 값은 프로세서 실물이다.** `build_coord_cfg`가 둘을 대조해 다르면 죽는다.
- `size` 계열 키는 **이 파일에만** 존재한다. 프로파일·칸별 override·`[tool.flwr.app.config]` 어디에도 두지 않는다(판정 18).

**`configs/quant_uni.yaml`** (신규) — 레지스트리 #9 처방
```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true
bnb_4bit_quant_storage: uint8
llm_int8_skip_modules: AUTO_PLUS_VISUAL     # get_keys_to_not_convert(meta) ∪ {"visual"}
dtype: bfloat16                              # from_pretrained 명시 (#10)
```
인자명이 `bnb_4bit_*`가 아니면 **예외도 경고도 없이 무시**된다. `to_dict()` 키 집합 **완전 일치** 대조로 오타를 잡는다.

**`configs/lora_uni.yaml`** (신규) — 통합 2칸 + 배경지식 + 분리형 판정부가 같은 파일을 읽는다
```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.0          # 판정 9 — 명시. 미기재 시 기본값이 파일 해시에 안 남는다
bias: none
use_rslora: false
use_dora: false
lora_bias: false
init_lora_weights: true
modules_to_save: null
target_parameters: null
trainable_token_indices: null
adapter_name: default
target_modules: $file:configs/target_modules_uni.json
```
`α/r`이 클라이언트마다 같아야 행렬별 평균이 정의된다. 매 fit마다 `realized_adapter_sha256`을 보고하고 서버가 전 셀 단일값을 assert한다(판정 21).

**`configs/target_modules_uni.json`** (신규, G4에서 동결)
```json
{"suffixes": ["linear_attn.in_proj_qkv","linear_attn.in_proj_z","linear_attn.in_proj_b",
              "linear_attn.in_proj_a","linear_attn.out_proj",
              "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
              "mlp.gate_proj","mlp.up_proj","mlp.down_proj"],
 "exclude_regex": "^model\\.visual\\.",
 "n_modules": 248, "lora_param_count": 32464896, "wire_nbytes_fp32": 129859584,
 "transformers_version": "...", "model_sha256": "...", "generated_by": "...", "sha256": "..."}
```

**테스트** `tests/test_vlm_fixed_uni.py`, `tests/test_configs_profiles.py`
- `test_프로파일_diff가_화이트리스트_밖에서_동일하다` (판정 18)
- `test_16gb와_48gb의_키집합이_완전히_같다` — 48gb.yaml 주석이 요구하는데 검사가 없었다
- `test_프로파일에_max_seq_len_size_pixels_접두사_키가_없다` — 정규식 스캔(완전 일치 아님)
- `test_유효배치_per_device곱grad_accum이_두_프로파일에서_같다`
- `test_eval_블록에_양자화_키가_없다` (판정 22)
- `test_FIXED_UNI를_extra_overrides로_덮으면_ValueError` — **transformers 미설치 환경에서도 돈다**
- `test_해석된_실효값이_base_yaml과_다르면_실패한다` (판정 19)
- `test_PairSpec_LoopCfg가_from_fixed_팩토리로만_생성된다`

---

## 4-5. `vlm/fixed.py` — `FIXED_UNI`와 실효값 대조

```python
FIXED_UNI: dict[str, Any]
def assert_fixed(resolved: Mapping[str, Any], base_yaml: Mapping[str, Any]) -> None
def assert_size_not_overridden(paths: Sequence[Path]) -> None
def profile_diff_whitelist() -> frozenset[str]
```
검사 대상은 "호출자가 명시적으로 덮으려 한 키"가 아니라 **"실제로 쓰일 값"**이다(판정 19). `transformers` import **전에** 돌아야 한다 — 설정 오류를 알아내는 데 무거운 optional 의존성이 필요할 이유가 없다(detection의 같은 원칙).

---

## 4-6. `scripts/vlm_processor_probe.py` + `vlm/coord_runtime.py` — 리프와 프로세서 실물의 유일한 접합면

**목적**: 프로세서에서 읽어야만 알 수 있는 값을 여기서 한 번 읽어 `CoordCfg`·`ImageGeom`으로 바꾸고, 선언과 실사용의 불일치를 즉시 실패로 만든다. 리프는 리프로 두고 더러운 일은 접합면 한 곳에 모은다.

```python
def read_processor_facts(processor, *, transformers_version=None) -> ProcessorFacts
def build_coord_cfg(declared: Mapping, facts: ProcessorFacts) -> CoordCfg
def encode_inputs(processor, *, facts: ProcessorFacts, images, text) -> dict   # 유일한 호출 경로
def extract_grid_thw(features, key="image_grid_thw") -> tuple[int, int, int]
def geom_from_grid(orig_w, orig_h, grid_thw, patch_size, merge_size) -> ImageGeom
def encode_boxes_for_target(boxes_px, geom, cfg) -> tuple[list[Box], list[int]]
def decode_boxes_to_px(boxes_model, geom, cfg) -> list[tuple[float, float, float, float]]
def load_image(path: Path) -> "PIL.Image"          # 16bit→8bit 규칙, mode 화이트리스트
def preproc_sha256(facts, max_seq_len) -> str
```

**핵심 결정**
- **factor를 하드코딩하지 않는다.** `patch_size`·`merge_size`를 실물에서 읽고 `factor == patch × merge` **관계만** assert. 코드·테스트 어디에도 32도 28도 판정에 쓰이지 않는다. 가짜 프로세서 14/2(→28)와 16/2(→32) **양쪽이 통과**하고, 선언 factor가 어긋나면 `CoordError`.
- **size 이중 표기 흡수**: Qwen3.5는 `size={shortest_edge, longest_edge}`, Qwen2.5-VL은 `min_pixels/max_pixels`. 둘 다 면적 단위이므로 `shortest_edge→CoordCfg.min_pixels`, `longest_edge→CoordCfg.max_pixels` **고정 매핑**(판정 7의 전제). 동시 존재 시 값 대조, 다르면 실패. 원시 키 이름·값은 `ProcessorFacts.size_keys`에 보존해 MLflow에 그대로 남긴다 — `CoordCfg` 필드명은 Qwen2.5-VL 시절 유산이라 이름만 보고 판단하면 안 된다.
- **facts를 속성이 아니라 실호출에서 역산한다**(판정 17): 프로브 이미지 2장(면적 하한 미만 1장, 상한 초과 1장)을 실제로 통과시켜 `image_grid_thw`로 복원한 면적이 하한/상한에 붙는지로 **실효** 상·하한을 확인한다. 속성값·size 딕트·실측 셋이 어긋나면 `CoordError`. 이 프로브는 G3-P2·P3와 같은 호출 한 번으로 끝난다.
- **`encode_inputs`가 프로세서를 부르는 유일한 경로**다. 픽셀 관련 kwargs는 `facts`에서만 구성한다. `vlm/` 전체 AST에서 `min_pixels`·`max_pixels`·`size`·`shortest_edge`·`longest_edge`·`resized_height`·`resized_width`를 **호출 kwarg로 넘기는 코드의 부재**를 검사한다 — 이 override는 yaml 검사로 원리적으로 못 본다.
- **매 호출 3줄 assert**(비용 0): grid가 merge 배수 / 복원 면적이 `[min_pixels, max_pixels]` 안 / grid 종횡비가 원본 종횡비와 factor 반올림 허용치 안(EXIF 회전·manifest W/H 오기·축 뒤바뀜을 한 번에).
- `smart_resize`를 **import도 구현도 하지 않는다.** `ABS_RESIZED`의 리사이즈 치수는 `image_grid_thw` 복원값만 인자로 받는다(`coords.ImageGeom.require_resized`가 이미 추정을 거부한다). AST로 심볼 부재 강제.
- **정변환 `to_model → quantize` 1회, 역변환 `to_px` 1회(정수화 없음)**(판정 5). 두 함수를 한 파일에 나란히 두어 눈으로 확인 가능하게 한다.
- 정변환에서 `is_degenerate`인 박스는 **넓혀 살리지 않고 드롭**하고 인덱스를 돌려준다. 한 샘플의 결함이 전부 드롭되면 타깃이 '결함 없음'이라는 거짓 라벨이 되므로 **샘플 폐기 신호**를 호출자에게 준다.
- `load_image`: 16bit 그레이스케일 RT 원본의 8bit 변환 규칙을 명시 함수로 두고 mode 화이트리스트를 assert. **분리형(cv2/Ultralytics) 경로와 같은 규칙**이어야 RQ2가 오염되지 않는다(Q5, A·C 공동).

**테스트** `tests/test_vlm_coord_runtime.py`

| 테스트 | 검증 대상 |
|---|---|
| `test_가짜프로세서_14x2와_16x2가_모두_통과한다` | factor 하드코딩 부재 |
| `test_선언_factor_불일치시_CoordError` | |
| `test_size_이중표기_매핑과_동시존재시_값대조` | 판정 7 전제 |
| `test_grid_축순서가_t_h_w다` / `test_grid키_부재시_가용키를_찍고_실패` | |
| `test_image_token_개수가_grid곱_나누기_merge제곱과_같다` | **G3-P3, merge 단위 오해 차단** |
| `test_pixel_values_행수가_t곱h곱w다` | G3-P2 |
| `test_복원면적이_min_max_pixels_안에_있다` | 호출 override 탐지 |
| `test_종횡비_불일치시_실패` | EXIF·manifest 오기 |
| `test_ABS_RESIZED에서_resized_미제공시_CoordError` | |
| `test_encode_decode_거울검사_정수화가_각_1회와_0회다` | 판정 5 |
| `test_W가_997_1233_1280_1600_2500에서_왕복오차가_예산이내다` | **F16 회귀 고정** |
| `test_퇴화타깃_드롭인덱스와_전결함드롭시_샘플폐기신호` | |
| `test_vlm전체_AST에_smart_resize_snap_to_bounds_심볼이_없다` | |
| `test_vlm전체_AST에_픽셀_kwarg_호출이_없다` | 판정 17 |
| `test_인자없는_open_read_text_write_text가_없다` | 인코딩 |

---

## 4-7. `scripts/pick_longest_edge.py` — 동결값 결정을 코드로

**목적**: 사람이 식을 옮겨 적는 경로를 없앤다. 매니페스트 해상도 전수를 **프로세서의 `smart_resize`에 직접 통과**시켜 비전 토큰 분포를 산출하고 P99.5를 보고한다.

`ceil(h/32)×ceil(w/32)` 식은 틀렸다 — transformers는 `round(x/factor)*factor`이고 파이썬 `round`는 banker's rounding이다. 실측 대조: 1280×720 → 실제 1280×704(**880 토큰**), ceil 식 920(+4.5%). 2000×1000 → 실제 1984×992(**1,922 토큰**), ceil 식 2016(+4.9%). 약 5% 과대 추정으로 필요보다 한 단계 낮은 해상도를 **영구 동결**하게 된다(레지스트리 #21).

산출물(해상도 히스토그램 + 토큰 수 분위수 + 선택값)을 G3-P6 첨부물로 남기고 sha256을 `base.yaml`에 기록한다. `factor`는 같은 도구가 프로세서에서 읽어 28/32 혼동을 원천 차단한다.

---

## 4-8. 프롬프트 — `vlm/prompts/*` · `scripts/build_unified_prompt.py` · `vlm/prompt_store.py`

```python
def load_prompt(name, *, root=PROMPT_DIR) -> PromptAsset      # read_bytes → 해시 대조 → utf-8 decode
def chat_template_sha256(processor_or_tokenizer) -> str
def prompt_bundle_hash(prompt_sha, chat_template_sha, target_format_sha) -> str   # → 태그 prompt_sha256
def prompt_render_sha(processor, messages) -> str             # 자리표시자 1개 렌더, 상수여야 함
def assert_generation_prompt(processor, messages) -> None
def vision_token_count(features) -> int                       # 샘플마다 로깅
```

**핵심 결정**
- 프롬프트는 코드가 아니라 파일. `vlm/prompts/SNAPSHOT.sha256`에 본 프롬프트 + 카나리아 프로브 + chat template 해시를 기록하고, `load_prompt`가 **읽는 즉시 대조**해 불일치면 예외. "읽을 수 있는데 다른 내용"인 상태로 진행할 방법을 주지 않는다.
- **`read_bytes()`로만 읽고 해시 대조 후 `.decode("utf-8")` 성공까지를 로드 조건에 넣는다.** cp949 기본 디코드는 UTF-8 한글을 대부분 예외 없이 다른 글자로 바꾼다 — 해시는 통과하고 모델이 본 문자열만 깨진다.
- 라벨 하드코딩 금지(불변조건 1-8)와 파일 동결을 동시에 만족시키기 위해 `build_unified_prompt.py`가 `configs/label_map.yaml` + `target_format` 상수에서 **생성**하고 커밋한다. 재생성 바이트 동일성 테스트가 label_map 드리프트를 잡는다. **단** 첫 통합형 run 이후에는 `PROMPT_FROZEN` 마커 존재 시 build 스크립트 자체가 거부한다 — "테스트가 깨지면 재생성"이라는 반사 행동을 물리적으로 막는다. `label_map_sha256`을 `gate_key`에 넣어 label_map 변경이 카나리아를 자동 무효화하게 한다.
- 프롬프트에는 **대표 ISO 코드만** 열거한다(alt 출력 빈도를 낮춘다). 정규화는 `postprocess`가 한다(4-20).
- **`render_prompt_sha`를 2값으로 쪼갠다**(관점 3의 단일 assert 기각):
  - `prompt_render_sha` = **이미지 자리표시자 1개**짜리 `apply_chat_template(tokenize=False)` 해시 → **상수 assert**. 이것이 프롬프트 번들에 들어가고 chat template 드리프트를 잡는다.
  - 샘플마다 `input_ids`의 `image_token_id` 개수 == `prod(grid_thw) // merge_size**2` **등식 assert** → 프롬프트 경로와 좌표 경로를 묶는 유일한 교차검증. 확장 후 토큰 수는 `vision_tokens`로 별도 로깅.
  - 프로세서 확장 결과를 상수로 assert하면 두 번째 이미지에서 런이 죽는다(1280×720은 880개, 227×227은 64개).
- **`assert_generation_prompt`**: `add_generation_prompt=True/False` 두 렌더의 차분이 비어 있지 않고 실사용 렌더가 그 차분 접미사로 끝나는지 검사(문자열 하드코딩 없이). 기본값 False로 렌더하면 생성문이 빈 문자열이 되고 전 칸이 `no_json`이다(F20).
- 카나리아 프로브는 **별도 파일 + 별도 해시**(`canary_grounding_v1.txt`, 영어).
- 길이 캐시 키에 **chat template sha256**을 포함한다 — {prompt sha, processor cfg sha, coord_cfg_hash, image sha}만으로는 템플릿(7,756자, tokenizer_config 소유)을 덮지 못한다.

**테스트** `tests/test_vlm_prompts.py`
- `test_SNAPSHOT과_파일해시_일치` / `test_재생성_결과와_바이트_동일`
- `test_프롬프트_ISO코드목록이_label_map과_일치한다`
- `test_PROMPT_FROZEN_마커가_있으면_build가_거부한다`
- `test_prompt_render_sha가_샘플간_상수다`
- `test_image_token_개수_등식이_샘플마다_성립한다`
- `test_add_generation_prompt_차분이_비어있지_않다`
- `test_카나리아_프로브가_본프롬프트와_다른_파일이다`
- `test_통합형_2칸이_같은_prompt_sha256을_로깅한다`
- `test_read_bytes로_읽고_utf8_decode에_실패하면_예외`

---

## 4-9. `scripts/gate_structure_probe.py` · `fl/exchange_audit.py` · `scripts/gate_exchange_audit.py` — G1·G2

```python
TENSOR_SOURCES = ("PARAM", "BUFFER", "SD_EXTRA")
CLASSES = ("CONST_DERIVED", "QUANT_ARTIFACT", "CACHE_LIKE",
           "TRAINING_MUTATED_STAT", "TIED_ALIAS", "UNKNOWN")

@dataclass(frozen=True)
class TensorRecord:
    name: str; source: str; shape: tuple[int, ...]; dtype: str
    persistent: bool; in_exchange: bool; mutated_by_training: bool | None
    digest: str; klass: str; reason: str

def partition_tensors(model) -> dict[str, set[str]]        # P / B / X, remove_duplicate=False
def tied_aliases(model) -> dict[str, str]
def classify(name, *, source, persistent, mutated) -> tuple[str, str]
def probe_mutation(model, step_fn, *, steps: int = 3, include_generate: bool = True) -> dict[str, bool]
def attr_tensors(model) -> dict[str, "Tensor"]             # vars(m) 순회 — rope_deltas 포착
def base_quant_digest(model) -> str                        # packed weight + X∩QUANT_ARTIFACT
def audit_exchange_closure(model, adapter_keys: Sequence[str]) -> ExchangeReport
def gate(report: ExchangeReport) -> None                   # G2-1 ~ G2-8
def write_report(report, path: Path) -> Path
```

**핵심 결정**은 제1부 G2에 전부 적었다. 구현 시 놓치기 쉬운 것만 다시 적는다.
- `named_parameters(**remove_duplicate=False**)` — 기본값이면 tied `lm_head.weight`가 buffer로 오탐된다.
- 분류 1차 기준은 `(source, persistent, mutated)` 조합이고 **이름은 사유 문자열에만** 쓴다.
- 프로브는 **합성 픽스처 배치**로 돈다. 학습 풀을 건드리면 게이트 자체가 미기록 소비를 만든다(buffer 변이 판정에 실데이터가 필요하지 않다).
- 보고서에 `n_tensors_examined`를 적는다.
- `--stage structure`는 meta device(CPU 수초, 가중치 불필요), `--stage runtime`은 NF4 실로드 + 3스텝 프로브 + 2회 양자화 결정론 + 3클라이언트 digest 대조.

**테스트** `tests/test_exchange_audit.py`
- `test_tied_weight가_TIED_ALIAS로_분류되고_buffer로_오탐되지_않는다`
- `test_qwen35_구조_스냅샷` — buffer 3개·전부 비persistent·지속 buffer 0개 회귀 고정
- `test_학습중_변이_stat이_있으면_게이트가_막는다` (토이 모델에 running stat 등록)
- `test_UNKNOWN이_1건이라도_있으면_막는다`
- `test_교환폐포_등식이_깨지면_막는다` — requires_grad 집합 ≠ 어댑터 키 집합
- `test_비어댑터_키가_bit_exact가_아니면_막는다`
- `test_bnb_quant_state가_X집합에_잡힌다` — `remove_duplicate=False` 뺄셈
- `test_base_quant_digest가_rope_상수의_해시가_아니다`
- `test_generate경로와_학습모드의_buffer목록_차이가_기록된다`
- `test_n_tensors_examined가_보고서에_있다`

---

## 4-10. `vlm/model_io.py` — 모델 로드·어댑터 부착·실사용 기록

**목적**: `transformers`/`peft`/`bitsandbytes`를 만지는 **유일한** 모듈. `vlm` 안에서 여기만 무거운 의존성을 갖는다.

```python
@dataclass(frozen=True)
class QLoraSpec:
    base_ckpt: Path; base_ckpt_sha256: str
    quant_cfg: Path; quant_cfg_sha256: str
    lora_cfg: Path; lora_cfg_sha256: str
    target_modules_file: Path; target_modules_sha256: str
    dtype: str = "bfloat16"; attn_implementation: str = "sdpa"

def build_quant_config(spec) -> "BitsAndBytesConfig"
def load_base_4bit(spec, *, device_map={"": 0}) -> "PreTrainedModel"
def attach_adapter(model, spec, *, seed: int) -> "PeftModel"
def assert_adapter_placement(model, spec) -> AdapterPlacement
def freeze_vision(model) -> FreezeReport
def realized_adapter_sha256(model) -> str
def adapter_state_fp32(model) -> dict[str, np.ndarray]
def load_adapter_state(model, state) -> InjectionReport
def effective_train_config(model, processor, optimizer, scheduler) -> dict[str, Any]
def env_fingerprint() -> dict[str, Any]
def runtime_fingerprint(model) -> str
def assert_eval_runtime(model) -> None
def merge_final(...) -> Path
```

**핵심 결정**
- **`llm_int8_skip_modules`를 손으로 적지 않고 합성한다**(F4): meta device 인스턴스로 `get_keys_to_not_convert(model)`을 얻어 `∪ {"visual"}`. 로드 **직후** 전수 검증 — `Linear4bit` 집합 sha256(`quantized_lm_modules_sha256`), `type(model.lm_head).__name__ == "Linear"` **and** `weight.dtype is bfloat16`, `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()`(tie 유지), `model.visual` 하위 `Linear4bit` **0건**.
- `dtype=torch.bfloat16` **명시**(#10). `"auto"`에 맡기지 않는다. 비양자화 파라미터 중 bf16이 아닌 것이 1건이라도 있으면 실패.
- `prepare_model_for_kbit_training`을 **통째로 부르지 않는다.** 그 안의 셋(`config.use_cache=False`, gradient checkpointing `use_reentrant=False`, `enable_input_require_grads`)만 명시 수행한다 — 헬퍼의 dtype 승격 규칙이 버전 의존이고 `text_config.mamba_ssm_dtype=float32`와 상호작용하는 것이 문서화돼 있지 않다(레지스트리 #7 후보). 대신 **파라미터 dtype 히스토그램**을 로깅하고 `declared_dtype`과 대조한다.
- **gradient checkpointing 하드 게이트**: `enable_input_require_grads()` + `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`를 명시하고 **첫 옵티마이저 스텝 직후** `n_grad = sum(1 for p in trainable if p.grad is not None and p.grad.abs().sum() > 0); assert n_grad == 496`. `use_reentrant=True`(PyTorch 기본) + 동결·양자화 임베딩이면 backward가 세그먼트 안으로 안 들어가 **lora_B가 영원히 0**이고, loss는 유한하고 스텝은 정확히 돌고 회계는 전 셀 통과한다.
- `attach_adapter`: `target_modules`는 접미사 앵커 목록(판정 17). `modules_to_save=None`, 어댑터 이름 `default` 고정. **`'all-linear'` 문자열은 이 저장소에서 `dump_target_modules.py` 안에만 존재한다.**
- `assert_adapter_placement`: **해석된 전체 이름 집합**과 동결 목록의 **완전 일치**(부분집합 아님). 부분집합만 보면 목록에 있는데 안 붙은 모듈을 놓친다. `visual|vision|merger` 0건. 모듈 수 248·파라미터 32,464,896 assert.
- `freeze_vision`: `requires_grad=False`뿐 아니라 **forward 후크로 비전 출력의 `requires_grad == False`**를 확인한다. 후자가 참이어야 비전 그래프가 만들어지지 않아 활성 메모리 절감이 선언이 아니라 실측이 된다.
- `load_adapter_state`: **`set_peft_model_state_dict`로만** 주입(런타임 키의 `.default.` 세그먼트 때문에 `load_state_dict` 직접 사용 시 strict=False가 조용히 0개 로드). 반환 `_IncompatibleKeys`를 **버리지 않고** `unexpected_keys == [] and missing_keys == []` assert. 주입 직후 `get_peft_model_state_dict`를 다시 떠서 **키 집합 일치 + 전 키 `torch.equal` 왕복 항등**(스칼라 norm이 아니라 bit 단위). `low_cpu_mem_usage=False` 고정 — `assign=True` 경로는 Parameter 객체를 교체하므로 옵티마이저를 주입보다 먼저 만들면 고아 텐서를 갱신하게 된다.
- `adapter_state_fp32`: `get_peft_model_state_dict(..., save_embedding_layers=False)` **명시**. `"auto"`는 판정 과정에서 **HF Hub로 네트워크 요청**을 보낸다(레지스트리 #16) — 비반출 서사와 정면 충돌. 이 호출은 `fl/adapter_io.py` 안 단일 진입점에서만 하고 다른 모듈의 직접 호출을 grep 테스트로 금지한다. 프로세스에 **`HF_HUB_OFFLINE=1` 강제**.
- `env_fingerprint()`: GPU 이름·capability, torch/transformers/peft/bnb/flwr 버전, `has_triton()`, `find_spec` (fla, fla_core, causal_conv1d, kernels, triton) 결과와 버전, **실효 커널 바인딩**(`torch_chunk_gated_delta_rule` 언랩 후 `__module__ + '.' + __qualname__`, `causal_conv1d_fn`), `type(m.norm).__name__`, `layer_types` 히스토그램, **모든 하위 config의 `_attn_implementation`**(root/vision/text). 단일 최상위 태그는 증거로 인정하지 않는다 — 32층 중 8층만 지배한다.
- `assert_eval_runtime`: `model.dtype is bfloat16`, `model.hf_quantizer is None`, 전 하위 config `_attn_implementation == "sdpa"`(판정 22).
- `merge_final`: 진입에서 `is_loaded_in_4bit is False`, `quantization_config is None` assert(4bit 병합은 dequant→재양자화 손실). 병합 카운터를 **agg_tag별로** 기록하고 `merge_calls[tag] == 1` + `merge_calls_during_rounds == 0`을 독립 규칙으로 감사한다. 병합 여부 확인은 `get_model_status().merged_adapters`(`PeftModel.merged_adapters`는 AttributeError, 레지스트리 #22). 같은 자리에서 `active_adapters == ["default"]`, `requires_grad == {"default": True}`도 확인해 어댑터가 비활성·동결된 채 도는 사고를 잡는다.
- `effective_train_config`: `quantization_config.to_dict()` 키 집합 **완전 일치** 대조, `_attn_implementation` 전 경로, optimizer 이름·lr·betas·wd, `size.*`, factor, `max_seq_len`, 학습 파라미터 수·바이트, dtype 히스토그램, `loss_reduction`·`denominator_rule`·`model_class`·`accepts_loss_kwargs`, `linear_attn_backend` 블록. **선언과 다르면 호출자가 실패시킨다.**

**테스트** `tests/test_vlm_model_io.py`, `tests/test_vlm_effective_config.py`
- `test_skip목록을_지정해도_lm_head가_양자화되지_않는다` — **F4 회귀**
- `test_tie가_유지된다_data_ptr_동일`
- `test_visual하위_Linear4bit가_0건이다`
- `test_quant_config_키집합_완전일치_대조` — `quant_type=` 오타를 잡는다
- `test_bnb_4bit_quant_type이_nf4이고_compute_dtype이_bfloat16이다` — 실체에서 되읽음
- `test_declared_dtype과_히스토그램이_대조된다`
- `test_어댑터가_동결목록과_완전일치한다` / `test_목록에_있는데_안붙어도_실패한다`
- `test_visual_vision_merger_0건`
- `test_save_embedding_layers_False_명시` / `test_get_peft_model_state_dict_직접호출이_단일진입점_밖에_없다`
- `test_주입_반환값의_unexpected_missing이_비어있다` / `test_주입후_왕복이_bit_동일하다`
- `test_키를_뒤섞거나_1개_누락시키면_중단된다` — 의도적 셔플 주입
- `test_라운드0_lora_B_norm이_0이고_lora_A_norm이_양수다`
- `test_첫스텝_직후_496개_전부_grad가_있고_0이_아니다` — **gradient checkpointing 회귀**
- `test_비전출력의_requires_grad가_False다` — forward 후크
- `test_env_fingerprint에_커널_바인딩과_전_하위config_attn이_있다`
- `test_assert_eval_runtime이_4bit_모델을_거부한다`
- `test_merge가_4bit_base에서_거부된다` / `test_merge_calls가_agg_tag별로_1이다`
- `test_effective_config에_필수키가_전부_있고_불일치_주입시_예외`

---

## 4-11. `scripts/dump_target_modules.py` — 부착 대상 확정 도구 (1회 실행, G4)

```bash
python scripts/dump_target_modules.py --ckpt <path> --out configs/target_modules_uni.json
```
- `all-linear` 전개 결과를 **부모 모듈 클래스명**과 함께 덤프해 SSM/linear-attention 계열과 비전 계열을 사람이 눈으로 분류한 뒤 sha256으로 동결한다. `'all-linear'` 문자열이 존재하는 유일한 장소다.
- **"학습 손실 경로 도달 여부" 열을 forward 후크로 실측**해 사람 분류의 오분류를 잡는다(MTP 헤드처럼 이름·클래스가 평범한 Linear인데 grad가 영원히 None인 모듈).
- 두 클래스(`ForConditionalGeneration` / `ForCausalLM`) 모두에서 전개해 **접미사 목록이 두 구조에서 같은 집합을 고르는지** 산출물에 기록한다.
- 산출물에 부착 모듈 수·파라미터 수·fp32 바이트·`transformers_version`·`model_sha256`을 함께 기록하고 학습 시작 시 assert. **버전 핀이 바뀌면 재실행 필수**(`mtp_num_hidden_layers=1`이 살아나는 버전에서 MTP 층이 생길 수 있다).

**테스트** `tests/test_target_modules.py`
- `test_표준_접미사_리스트는_linear_attention층을_건너뛴다` — 128/248·21.2M/32.5M 회귀 고정. 레지스트리 #11이 되살아나면 즉시 깨진다
- `test_확정_목록이_248개에_붙고_visual은_0건이다`
- `test_두_클래스에서_같은_집합을_고른다` — `set(uni - 'language_model.') == set(sep)`
- `test_손실경로_미도달_모듈이_보고서에_표시된다`

---

## 4-12. `vlm/generation.py` — 디코딩 고정과 실효 설정 기록

```python
def build_generation_config(*, decode: DecodeCfg, terminator_ids, pad_token_id) -> "GenerationConfig"
def assert_greedy(gc) -> None
def resolve_eos(processor) -> tuple[int, ...]
def stop_reason(sequences, prompt_len, eos_ids) -> Literal["eos", "length"]
def effective_generation_tags(model, gc, decode) -> dict[str, str]
```

**핵심 결정**
- **백지 `GenerationConfig()`**에서 시작해 `do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=<동결>`만 세우고 `temperature/top_p/top_k/typical_p/penalty_alpha`는 **None으로 명시 해제**. `generate(..., generation_config=gc)`로 매번 명시 전달하고 `model.generation_config`도 이 객체로 교체한다(레지스트리 #5).
- `assert_greedy`에 **완화 스위치를 두지 않는다.** 스위치는 언젠가 켜진다.
- **eos를 추측하지 않는다**(F20): 학습 타깃 직렬화 끝에 붙는 어시스턴트 종료 문자열을 chat template **차분에서 유도**해 토크나이즈하고, `gc.eos_token_id = sorted({tokenizer.eos_token_id} | terminator_ids | config.eos_token_id)`. `assert set(gc.eos_token_id) >= terminator_ids`.
- **`stop_reason`을 길이가 아니라 마지막 토큰 정체로 판정한다**: `sequences[0, -1].item() in eos_ids`면 `eos`. 길이로 판정하면 정확히 마지막 허용 위치에서 EOS를 뱉은 정답이 오답이 되고, 발생률이 출력 길이 분포에 비례해 **판정문이 긴 칸이 구조적으로 손해**를 본다.
- **`batch_size = 1`을 디코딩 고정 항목에 편입한다.** left-padding 배치 생성은 패딩에 따라 logits가 미세하게 달라져 greedy argmax가 뒤집힐 수 있다 — 배치 크기가 결과를 바꾸면 "디코딩 5칸 공통 고정"이 문면으로만 남는다. D의 latency 측정이 batch=1을 요구하는 것과도 일치한다.
- `effective_generation_tags`: `model.generation_config.to_diff_dict()` **실측**을 태그로 남기고 정렬 JSON의 sha256을 `gen.effective_sha256`으로 함께 남긴다(5칸 대조를 한 값으로). `gen.eos_token_ids`·`gen.eos_tokens`(디코딩된 문자열)·`gen.config_json_present=false`(#5b 부재 확인의 증거를 겸한다).
- **D의 `rag/judge`도 이 빌더를 import한다.** 통합형 전용으로 두면 판정부가 별도 `GenerationConfig`를 만들어 "디코딩 5칸 공통 고정"이 문면으로만 남는다.

**테스트** `tests/test_vlm_generation.py`
- `test_백지_config가_greedy다` / `test_샘플링_노브가_전부_None이다`
- `test_model_generation_config에_do_sample_True가_박혀있어도_bind후_통과한다`
- `test_batch_size가_1이_아니면_실패한다`
- `test_stop_reason이_마지막토큰_정체로_판정된다` — 길이 동률 케이스에서 `eos`
- `test_terminator_ids가_eos집합에_포함되지_않으면_실패한다`
- `test_effective_태그에_실측값과_eos목록과_sha256이_있다`
- `test_generation_config_json_부재_존재를_탐지한다`
- `test_judge가_같은_빌더를_쓴다` — import 경로 테스트

---
## 4-13. `vlm/collate.py` — 콜레이터

```python
class PairCollator:
    def __call__(self, batch) -> dict[str, "torch.Tensor"]
def mask_labels(input_ids, prompt_ids, mm_token_type_ids) -> "torch.Tensor"
def bucket_by_length(samples, lengths, *, n_buckets: int) -> list[list[int]]
```

**핵심 결정**
- **프로세서가 낸 키는 하나도 버리지 않는다.** `assert set(proc_out) <= set(collated)`. `mm_token_type_ids`가 빠지면 `modeling_qwen3_5.py` L1487-1492가 `ValueError("Multimodal data was passed … but mm_token_type_ids is missing")`로 첫 스텝에 죽는다. 프로세서는 이 키를 돌려주지만 `padding: False`가 기본이라 **패딩은 우리 몫**이다 — `input_ids`와 **같은 길이로 텍스트 값(0)** 우측 패딩하고 `len(mm_token_type_ids) == len(input_ids)`를 절단 센티널과 같은 위치에서 assert.
- **라벨 마스킹 검증은 `input_ids == image_token_id`가 아니라 `mm_token_type_ids != 0` 기준**으로 한다 — 모델이 멀티모달 위치를 판정하는 것과 같은 신호를 쓴다. `<|vision_start|>`/`<|vision_end|>`는 `image_token_id`가 아니므로 별도 assert가 필요하다. 예측이 자명한 토큰이 총괄 집합에 섞이면 손실 분모가 부풀고(판정 2와 직결) 토큰 정확도가 무의미하게 오른다.
- 마스킹은 문자열 검색이 아니라 **프롬프트-only 인코딩의 prefix 일치 assert**로 만든다. 샘플별 총괄 토큰 비율을 로깅한다.
- `position_ids`를 직접 만들어 넘기지 않는다 — L1487 검사를 우회해 3D M-RoPE 대신 1D 위치로 학습이 돌고, 좌표 grounding이 주 타깃인 통합형에서 카나리아는 통과한 채 본학습만 망가진다. 검증용으로 `language_model`에 forward 후크를 걸어 `position_ids.shape[0] == 3`을 확인한다.
- 동적 패딩(**패킹 미사용** — 노출 횟수 등가 유지). **절단 센티널**: 선계산 길이 != 조립된 `input_ids` 길이면 즉시 실패(프로세서 드리프트·캐시 노후 탐지). `max(len) <= max_seq_len` assert.
- **길이 버킷팅**(판정 10): 최대 길이 배치가 언제 오는지를 결정적으로 만들고 할당 크기 종류를 줄인다. 패킹이 아니므로 노출 횟수 등가는 유지된다.

**테스트** `tests/test_vlm_collator.py`
- `test_프로세서_출력키가_전부_보존된다`
- `test_mm_token_type_ids가_있고_input_ids와_길이가_같다` / `test_누락시_forward가_ValueError`
- `test_프롬프트_구간_라벨이_전부_-100이다` (prefix 일치)
- `test_mm_token_type_ids가_0이_아닌_위치가_전부_-100이다`
- `test_vision_start_end_위치도_-100이다`
- `test_감독토큰수가_0보다_크다` / `test_동적패딩_길이가_배치내_최대다`
- `test_선계산길이와_조립길이_불일치시_절단센티널이_예외를_던진다`
- `test_position_ids가_3D다` — forward 후크
- `test_길이버킷이_노출횟수를_바꾸지_않는다`

---

## 4-14. `vlm/pair_dataset.py` · `vlm/corpus_dataset.py` — 데이터

```python
@dataclass(frozen=True)
class PairSpec:
    pairs_jsonl: Path; counts_json: Path; snapshot_root: Path
    prompt_file: Path; coord_cfg: CoordCfg; max_seq_len: int = 2048
    @classmethod
    def from_fixed(cls, base_yaml, overrides) -> "PairSpec"      # 자유 생성자 금지 (판정 19)

def load_pairs(spec, *, split="train", client=None) -> list[PairSample]
def build_target_text(pair, geom, cfg) -> str
def precompute_lengths(samples, bundle, *, cache_path, exact_frac=0.01) -> LengthTable
def filter_by_length(samples, lengths, max_seq_len) -> tuple[list[PairSample], DiscardReport]
def write_consumed_ids(samples, path) -> Path
def virtual_permutation(n: int, *, seed: int, total: int) -> np.ndarray      # 판정 8
def slice_for_round(perm, *, round_idx, steps, effective_batch) -> np.ndarray
```

**핵심 결정**
- 로드 경로는 **A의 승인 로더 단일 경로**다: `load_snapshot(spec.snapshot_root)`(SNAPSHOT.sha256 검증 포함, **`verify=False` 금지**) → `split_view(manifest, 'train', client)` → `join_defects(snap)`. **`pandas.read_csv` 직접 호출 금지** — 빈 문자열이 NaN이 되어 '정상'과 '결측'이 섞인다.
- **이미지 경로·크기의 정본은 manifest다.** 페어 값은 대조용이고 불일치 시 실패. 이미지 파일 sha256도 manifest 값과 대조한다.
- **bbox 타깃은 페어가 아니라 `annotations.csv`에서 온다.** B 골격 키 목록(§4-5)에 bbox가 없고 원본 픽셀 bbox의 정본은 A 자산이다(불변조건 1-2·3-8). 조인은 `defect_instance_id ↔ ann_id = f"{image_id}#{i}"` — **이 사상 규칙이 계약 공백이므로 확정 전에는 assert로 막고 진행하지 않는다**(Q3).
- 좌표 변환은 `coord_runtime.encode_boxes_for_target` 한 곳에서만. 모델 좌표는 어떤 파일에도 저장하지 않는다. `verdict_mode == clause_only`면 합부 결론·margin 없는 변형 템플릿(B §4-7).
- `precompute_lengths`: **비전 토큰을 포함한 실측 길이.** 빠른 경로는 manifest W/H + 프로세서 자신의 리사이즈 함수로 `image_grid_thw`를 얻고, 무작위 1%는 프로세서 실호출로 교차검증해 `fast == real` assert. 캐시 키 = {prompt sha, processor cfg sha, **chat template sha**, coord_cfg_hash, image sha}.
- **절단이 아니라 폐기**(레지스트리 #6). `discarded_length.jsonl`(image_id, n_tokens, vision_tokens, text_tokens, reason) + 폐기 건수·**클라이언트별** 폐기율·히스토그램을 MLflow에. **폐기율 상한은 전역이 아니라 클라이언트별 1%**이고, 클라이언트 간 `n_after/n_declared` 편차가 1%p를 넘으면 게이트로 올린다(F11). `DiscardReport`는 클라이언트 × 결함코드 × 재질 교차표로 산출해 폐기가 클래스 균형을 흔드는지 카이제곱 한 줄로 사전 등록 검사한다.
- `virtual_permutation` + `slice_for_round`가 판정 8의 구현이다. `drop_last`로 예산이 조용히 줄어드는 경로를 제거하고, 정확히 `S_k × accum` 마이크로배치를 소비한다.
- **`split == 'eval'` 또는 `eval_subset`이 비어 있지 않은 행이 하나라도 들어오면 즉시 실패.**
- `write_consumed_ids`: **매 fit마다** `outputs/consumed/{cell}_s{seed}_r{t}_c{k}.txt`(append 금지, 정렬·중복 제거). `VlmCell`에 `consumed_ids_sha256`·`consumed_count`. 서버 종료 시 concat해 `consumed_image_ids.txt`. 이것이 없으면 **5칸 중 통합·연합만 누출 검증 불가 상태로 남는다**(스펙 §2-1 정식 산출물).
- `datasets`(Arrow) 미사용 — 캐시 무효화 표면을 하나 줄인다.

**`vlm/corpus_dataset.py`**: D3-(b) QA 2만 / D3-(c) 판정추론 1만의 **길이 필터·폐기 리포트 소유자**. `pair_dataset`은 D4 전용이라 배경지식 단계의 소유 모듈이 비어 있었다. `run_background_sft`도 `FIXED_UNI`·`StepBudget`·폐기 리포트·`effective_train_config`·회계 JSON을 산출해 **공통 출발점의 조기 종료 부재와 절단 0건을 증명 가능하게** 만든다.

**테스트** `tests/test_vlm_dataset.py`
- `test_split_eval_eval_subset_유입시_즉시_실패`
- `test_페어와_manifest의_client_split_크기_sha256_불일치시_실패`
- `test_pandas_read_csv_직접호출이_없다` (AST) / `test_verify_False_호출이_없다`
- `test_길이_2048은_유지_2049는_폐기` (경계)
- `test_절단이_일어나지_않는다` / `test_폐기건수와_사유가_정확히_집계된다`
- `test_클라이언트별_폐기율_상한과_편차_게이트`
- `test_프롬프트_sha가_바뀌면_길이캐시가_무효화된다` / `test_chat_template_sha도_캐시키에_있다`
- `test_비전토큰_빠른경로와_프로세서_실호출이_일치한다`
- `test_ann_id_사상규칙_미확정시_assert로_막는다`
- `test_R라운드_누적_노출이_모든_표본에_대해_정확히_3회다` — **판정 8 핵심**
- `test_consumed_image_ids가_정렬_중복제거되어_생성된다`
- `test_n_k를_manifest에서_파생하려_하면_실패한다` — `train_index`만 허용

---

## 4-15. `vlm/sft_loop.py` — 프레임워크 없는 학습 루프

**`transformers.Trainer`·`trl`을 import하지 않는다.** 스텝 예산·LR 오프셋 재개·grad accum·손실 정규화만 담당하고 데이터·모델은 인자로 받는다.

```python
@dataclass(frozen=True)
class LoopCfg:
    local_steps: int; total_steps: int; step_offset: int
    micro_batch: int; grad_accum: int
    lr: float; betas: tuple[float, float]; weight_decay: float
    warmup_ratio: float; max_grad_norm: float; seed: int
    log_every: int; ckpt_every: int
    @classmethod
    def from_fixed(cls, base_yaml, overrides) -> "LoopCfg"

def run_steps(model, batches, opt, sched, cfg, *, on_step=None) -> LoopResult
def build_scheduler(opt, cfg) -> "LambdaLR"
def supervised_logits_loss(model, batch, labels, *, use_supervised_positions: bool) -> tuple["Tensor", int]
def witnesses(opt, sched, trainable) -> Witnesses
def save_resume(path, *, model, opt, sched, rng, cursor, global_step) -> Path
def load_resume(path) -> ResumeState
```

**핵심 결정**
- **루프의 유일한 종료 조건은 예산 도달**(`StepBudget`, `stop_reason='budget'`). 그 외 종료는 run 무효.
- **NaN/inf 손실은 배치 스킵이 아니라 run 실패** — 스킵은 학습량을 조용히 줄이는 조기 종료의 변형이다.
- **`clip_grad_norm_(..., error_if_nonfinite=True)`**(레지스트리 #18). 기본 False면 inf 그래디언트에서 클립 계수가 0이 되어 **그 스텝의 모든 그래디언트가 조용히 0으로 눌리는데** 증인 3종이 전부 정상 증가한다. 라운드마다 모멘트를 리셋하는 연합 칸이 초반 큰 그래디언트를 더 자주 겪으므로 발생 빈도가 두 칸에서 다를 수 있고, 학습량 등가가 정확히 이 경로로 깨진다. 반환 `total_norm`을 스텝마다 기록해 회계에 `grad_norm_max`·`clip_hit_count`로 남기고, 발동률 비대칭 자체를 보고 항목으로 올린다.
- **손실은 `vlm/loss_norm.py`만 쓴다**(판정 2). `labels=None` forward.
- **총괄 위치 한정 로짓**은 착수 전제이되 `micro_batch == 1` 강제, `logits_to_keep`에 위치 LongTensor, `labels=` 미전달(판정 11). 기준 경로는 프레임워크 자체 손실.
- **LR**: 자기 총예산 `T_k = R·S_k`에 대한 **단일 cosine**을 `step_offset = (t−1)·S_k`에서 재개(`last_epoch = step_offset − 1`). 라운드마다 재시작하면 유효 LR 총량이 통합·중앙과 어긋나 학습량 등가 주장이 훼손된다(detection의 `scheduler.last_epoch = start_epoch − 1`과 같은 처방).
- **증인 4종** (하나라도 어긋나면 실패):
  1. 우리 카운터 `steps_ran`
  2. **전 학습 파라미터 전수** `int(optimizer.state[p]["step"]) == S_k`. **state 항목 부재를 0과 구분해 별도 실패 사유로** 명시한다 — torch는 grad를 받은 파라미터에만 state를 만들므로(`adam.py` L151) 죽은 어댑터가 아무 흔적 없이 통과한다
  3. `scheduler.last_epoch − step_offset == steps_ran`. 전역 예산의 증인은 라운드별 optimizer step이 아니라 회계의 `step_offset_in`·`last_epoch_out` 두 컬럼으로 `Σ_t (last_epoch_out − step_offset_in) == R·S_k`
  4. `sample_cursor_end − sample_cursor_start == steps_ran × B_eff` (판정 8)
- **재개 체크포인트**(F22): K 스텝마다 `last/step_{n}/`에 어댑터 + `optimizer.state_dict()` + `scheduler.state_dict()` + RNG + **샘플러 커서**를 저장한다. **조기 종료도 best 채점도 아니다 — 루프가 val 지표를 아예 읽지 않으므로 best 선택 경로가 생기지 않는다.** 저장 시점에도 `reject_best_checkpoint`를 걸고, 재개 시 3중 일치(`optimizer.state[p]['step'] == scheduler.last_epoch + 1 == 저장된 global_step`)를 assert한다. `stop_reason`을 `budget`/`resume`으로 구분하고 **최종 종료만 `budget`이어야 유효**로 친다. 회계에 재개 횟수·재개 스텝을 남겨 학습량 등가가 깨지지 않았음을 증명한다.
- 결정론: `cudnn.deterministic=True`, `CUBLAS_WORKSPACE_CONFIG`, `torch.are_deterministic_algorithms_enabled()` 로깅. **`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512`**(`expandable_segments`는 Windows 무동작, 레지스트리 #15). 다만 **bnb 4bit·SDPA 폴백·recompute는 비트 재현이 보장되지 않으므로** "시드 재현이지 비트 재현이 아니다"를 논문에 정확히 쓴다.

**테스트** `tests/test_vlm_loop_budget.py`, `tests/test_vlm_lr_schedule.py`
- `test_steps_ran과_optimizer_step과_scheduler와_커서가_일치한다`
- `test_증인_하나를_인위로_어긋내면_실패한다`
- `test_optimizer_state_항목_부재가_0과_구분되어_실패한다` — **죽은 어댑터**
- `test_NaN손실_주입시_배치스킵이_아니라_예외다`
- `test_inf_그래디언트에서_clip이_예외를_던진다` — `error_if_nonfinite`
- `test_stop_reason이_budget_이외_값을_가질_분기가_없다` (AST)
- `test_Trainer_SFTTrainer_EarlyStopping_load_best_metric_for_best_should_training_stop_save_total_limit_심볼이_없다` (소스 스캔)
- `test_step_offset_재개_LR이_단일cosine과_전스텝_일치한다` (1e-12)
- `test_라운드마다_재시작한_대조군은_이_테스트에_실패한다` — 방어선 작동 증명
- `test_라운드4에서_죽고_재개해도_증인_3중일치가_유지된다`
- `test_재개후_최종_stop_reason이_budget이다`

---

## 4-16. `vlm/round_runner.py` · `vlm/train_cell.py` — 통합 2칸 공통 유일 진입점

```python
def train_round(*, spec: QLoraSpec, pair_spec: PairSpec, loop_cfg: LoopCfg,
                adapter_in: Mapping[str, np.ndarray] | None,
                round_idx: int = 0, local_steps: int, total_steps: int,
                base_seed: int = 0, client_idx: int = 0, client: str | None = None,
                num_examples: int = 0, out_dir: Path,
                extra_overrides: dict[str, Any] | None = None) -> VlmRoundResult

def run_uni_central(cfg) -> Path                              # train_round(round_idx=0, local=total) 1회
def run_background_sft(cfg, stage: Literal["qa", "reason"]) -> Path
```

`VlmRoundResult`: `adapter, num_examples, round_idx, client_idx, steps_ran, samples_consumed, epochs_traversed, sample_cursor_start/end, sample_index_digest, seed, adapter_l2_A/B, payload_bytes, injection_digest, effective_config, discard_report, lr_trace, grad_norm_max, clip_hit_count, supervised_tokens, stop_reason, consumed_ids_path, model_builds, pid, max_mem_bytes, wall_s`

**핵심 결정**
- **통합·중앙은 R=1(`local_steps == total_steps`) 퇴화 케이스로 같은 함수를 통과한다.** 두 칸이 문자 그대로 같은 코드 경로를 탄다는 것이 RQ2 공정성 주장 자체다(detection의 `train_round`가 이미 증명한 패턴).
- **첫 줄에서 게이트를 검증한다**: `require_gate("exchange")` → `require_canary_pass(cfg, gate_key)` → `assert_fixed(resolved, base_yaml)` → `assert_size_not_overridden(profiles)` → `guard_no_cloud_logging()` + `assert_local_tracking()` + `guard_no_egress()`. **진입점 안에 있어야 우회가 불가능하다.**
- 무상태: 라운드마다 옵티마이저를 새로 만들고(모멘트 리셋 = FedAvg 표준, 논문 명시), 로컬 체크포인트를 저장하지 않는다(재개 체크포인트는 4-15의 별도 경로이고 라운드 경계에서 삭제).
- 어댑터 in/out은 **키드 `dict[str, np.ndarray]`(fp32)**. 리스트 순서 교환이 아니라 키가 페이로드에 실려 다니므로 '순서 뒤섞임' 실패 모드가 구조적으로 사라진다. `vlm`은 `flwr`를 import하지 않는다.
- `run_background_sft`: 3 epoch, 조기 종료 없음. **`FIXED_UNI`·회계·폐기 리포트를 그대로 통과한다.** 산출물은 병합 후 동결되어 5칸 공통 출발점이 되므로 여기가 오염되면 어떤 교차 칸 검사도 감지할 수 없다. 병합은 반드시 `ForConditionalGeneration`에서 수행하고, 병합본을 텍스트 클래스로 다시 로드했을 때 `_keys_to_ignore_on_load_unexpected = [r"^model.visual.*"]`로 비전 가중치가 버려지는 것이 **의도된 동작임을 로그로 확인**한다.
- 최종 산출 디렉터리명 `last/` 고정. 병합·채점·**저장** 시점 모두에서 `reject_best_checkpoint`.
- 회계 매트릭스 CSV·감사 JSON 산출을 **정상 종료 조건에 넣는다** — 감사 실패면 채점하지 않는다.
- 종료 시 `REQUIRED_TAGS`(확장본) 전량 채움.

**테스트** `tests/test_vlm_round_equivalence.py`, `tests/test_vlm_gate.py`
- `test_통합중앙과_통합연합이_같은_함수를_통과한다`
- `test_1클라이언트_R라운드가_수동리셋_단독학습과_손실궤적이_일치한다` (12a 검증 #4)
- `test_리셋없는_연속학습과의_차이로_모멘트_리셋_효과를_정량_분리한다`
- `test_카나리아_통과기록이_없으면_train_round가_시작을_거부한다`
- `test_coord_cfg_hash나_probe_sha가_다르면_거부한다`
- `test_교환폐포_게이트_fail이면_거부한다`
- `test_배경지식_SFT도_회계와_폐기리포트를_산출한다`

---

## 4-17. `scripts/vram_probe.py` · `scripts/step_time_probe.py` — G7

```bash
python scripts/vram_probe.py     --worst-bucket --steps 20    # B1
python scripts/step_time_probe.py --n 64                      # B3
```
- **B1은 길이 상위 1% 샘플만으로 구성한 최악 배치**를 돈다. 평균 길이로 돌린 파일럿은 꼬리에서 죽는다. `max_memory_allocated`/`max_memory_reserved`/**배치 최장 시퀀스 길이 분포**를 남기고, `reserved − allocated`를 단편화 실측치로 기록해 표의 1~2 GB 추정을 교체한다.
- B3는 동결된 `size` 캡·LoRA 설정으로 micro-pass 시간 중앙값을 재고 `총 micro-pass 수 × t_med`를 출력해 예산표와 대조한다. **100 step 파일럿을 기다리지 말고 9-5의 실측치(0.0611 / 0.1509 s/층)를 지금 GPU-일 재추정에 넣는다.**
- B4·B5는 `env_fingerprint()`를 산출물로 남긴다.
- **레버 분류**(OOM 대응):
  - **레시피 중립(허용, 실사용만 기록)**: gradient checkpointing(`use_reentrant=False`), 총괄 위치 한정 로짓·청크 CE, 길이 버킷팅, dataloader worker 수, 라운드 간 actor 정리·`empty_cache`, `garbage_collection_threshold`/`max_split_size_mb`
  - **레시피 변경(금지)**: `size.shortest_edge/longest_edge`, `max_seq_len`, 유효 배치, LoRA r/target, bf16→fp16, packing, **micro_batch**(판정 9)
  - **경계(전 칸 동시 적용 + 선언 시에만)**: paged AdamW
  - 사다리를 다 써도 안 되면 **48GB 이관이고 통합 2칸을 동시에 옮겨 둘 다 재실행한다**(Q11).

---

## 4-18. `fl/adapter_io.py` · `fl/adapter_aggregate.py` · `fl/comms.py` · `fl/checkpoint.py`

```python
# adapter_io.py — 교환 규약의 단일 소유자
AdapterSD = dict[str, np.ndarray]
def canonical_adapter_keys(state) -> tuple[str, ...]          # sorted 고정
def keys_sha256(keys) -> str
def shape_table(state) -> dict[str, tuple[int, ...]]
def to_wire(sd, *, keys, keys_hash) -> AdapterSD              # fp32 in, fp32 out — 캐스트 0회
def from_wire(wire, *, keys, keys_hash) -> "OrderedDict"
def assert_adapter_only(keys) -> None
def pair_index(keys) -> dict[str, tuple[str, str]]
def adapter_nbytes(wire) -> int
def inject(model, wire) -> InjectionReport                    # 단일 진입점

# adapter_aggregate.py — numpy만. GPU·peft 없이 전수 테스트가 돈다
def weighted_adapter_average(client_adapters, num_examples, *, keys, keys_hash,
                             shape_table, expected_clients) -> AdapterAggregationResult
def aggregation_residual(client_adapters, num_examples, *, pairs, scaling,
                         base_fro=None) -> dict[str, ResidualStat]
def adapter_norms(sd) -> dict[str, float]

# comms.py
class CommsLedger:  record / totals / per_round_per_client / to_csv / assert_expected
def bootstrap_cost(cell, *, base_param_count, itemsize) -> CommsRecord

# checkpoint.py
def save_global_adapter(adapter, path, *, meta) -> Path
def save_milestone(round_idx, adapter, client_payloads, out_dir) -> Path
def load_latest(out_dir) -> tuple[AdapterSD, int]
def assert_final_only(round_idx, rounds, agg_tag) -> None
```

**핵심 결정**
- **fp32 전 구간, 캐스트 0지점**(판정 13). `to_wire`/`from_wire`의 assert는 `dtype is float32`.
- `assert_adapter_only`: `visual|vision|merger|embed_tokens|lm_head|modules_to_save` 패턴 0건.
- 집계: **`sorted(client_id)` 순 float64 누적 → fp32**. 근거는 "비트 재현성"이 아니라 **"Ray 완료 순서가 결과에 새어들지 않음"**(판정 13). `order`를 결과에 남긴다.
- **assert 8종 fail-fast**: 키 해시 == canonical / 전 입력 dtype == float32 / shape == 동결표 / lora_A·B 짝 완전 대응 / 정규화 가중치 합 == 1.0(atol 1e-9) / isfinite 전수 / 출력 norm ≤ max(입력 norm)×(1+1e-6) / **`expected_clients` 집합 정확 일치(부분 참여 = 실패)**.
- **집계 대수는 정확한 등식이다** — 근사가 아니다:
  ```
  Σ p_k B_k A_k = B_g A_g + Σ p_k δB_k δA_k
  ⇒ avg(B)·avg(A) − avg(B·A) = − Cov_p(B, A)
  ```
  차이항은 클라이언트 갱신 편차의 **가중 교차공분산 하나**로 끝난다. K=1이거나 갱신이 동일하면 **정확히 0**임이 즉시 보인다(1클라이언트 항등 테스트의 이론적 근거). 전 클라이언트가 매 라운드 같은 글로벌 어댑터에서 출발하므로 이 항은 로컬 스텝 크기의 2차항이다. **크기는 선험적으로 주장하지 않고 실측한다.**
  - **`α/r`이 상수라는 것이 이 논증의 필수 조건이다.** 실효 갱신은 `ΔW = (α/r)·B·A`이고 `α/r`이 클라이언트마다 다르면 행렬별 평균의 정의 자체가 무너진다 → `realized_adapter_sha256`을 매 fit 보고·전 셀 단일값 assert.
  - 지표 2종: `ε_l = ‖B_g A_g − Σp B_k A_k‖_F / ‖Σp B_k A_k‖_F`(집계 오차의 상대 크기)와 `ε′_l = (α/r)·‖Σp δB δA‖_F / ‖W_base,l‖_F`(그 오차가 건드리는 가중치 대비 크기). **`ε_l`만 보고하면 "30% 오차"가 커 보이지만 그 ΔW 자체가 `W`의 0.1%라면 실질 무의미하다.** `ε′_l`이 그 판단을 준다.
- **통신량 2열**(판정 15): `tensor_nbytes`(정확 일치 게이트) / `wire_nbytes`(논문 수치). **1회 사전 배포 비용은 라운드 합계와 다른 행**으로 기록한다 — 합치면 통합형 13.75 GB가 라운드 통신량 4.67 GB를 압도해 결론이 뒤집히는데, 그 사실을 숨기지도 섞지도 않는다.
- `milestone_rounds`에 **최종 라운드 무조건 포함**. 마일스톤마다 클라이언트 어댑터 3벌을 동반 보존(1회 390 MB, 4지점 1.56 GB). 이게 없으면 부록 1단·2단이 **사후 복구 불가능**하다.
- `assert_final_only(round_idx, rounds, agg_tag)`: 병합 대상은 `round_idx == rounds`이고 채점 경로는 `agg_tag == 'matrix_avg'`만 허용.

**테스트** `tests/test_adapter_io.py`, `tests/test_adapter_aggregate.py`, `tests/test_comms_ledger.py`
- `test_어댑터가_전_구간_fp32다` (서버 init → wire → 주입 → 반환 dtype 전수)
- `test_교환키에_visual_merger_embed_lm_head가_없다` / `test_키해시_불일치는_즉시_실패`
- `test_tensor_nbytes가_파라미터수x4와_정확히_같다` / `test_wire_nbytes_헤더_상한` (496×128 B)
- `test_가중평균이_손계산과_일치한다` / `test_입력_딕셔너리_순서를_뒤섞어도_bit_exact다`
- `test_클라이언트가_1개면_집계가_항등이다` (atol 0)
- `test_잔차를_두_경로로_계산해도_같다` — `‖B_gA_g − Σp B_kA_k‖`와 `‖Σp δB δA‖`. **논문 대수 주장 자체를 코드가 증명한다**
- `test_가중치합이_1이_아니면_실패` / `test_출력norm이_입력norm_최댓값을_넘지_않는다`
- `test_fp32가_아닌_입력_shape표_불일치_짝불일치를_거부한다`
- `test_기대_클라이언트가_빠지면_집계하지_않는다`
- `test_사전배포_비용이_라운드_합계에_섞이지_않는다`
- `test_agg_tag가_다른_행은_주표에_들어가지_않는다`
- `test_최종라운드_클라이언트_페이로드_3벌이_디스크에_있다`

---

## 4-19. `fl/accounting_vlm.py` · `fl/client_vlm.py` · `fl/server_vlm.py` — 연합

**`VlmCell` 컬럼** (검출 `budget_audit.py`와 같은 판정 강도)
`round_idx, client_idx, participated, steps_ran, steps_planned, n_k_declared, n_k_effective, n_k_source_sha, dropped_long_samples, discard_rate, seed, micro_batch, grad_accum, effective_batch, samples_seen, tokens_seen, supervised_tokens, sample_cursor_start/end, sample_index_digest, exposure_min/max, coverage, lr_first, lr_last, step_offset_in, last_epoch_out, grad_norm_max, clip_hit_count, optimizer, arg_optimizer, betas, weight_decay, adapter_l2_A, adapter_l2_B, inject_rel_err, uplink_tensor_bytes, uplink_wire_bytes, downlink_*, keys_sha256, realized_adapter_sha256, base_ckpt_sha256, base_quant_digest, quantized_lm_modules_sha256, preproc_sha256, prompt_sha256, chat_template_sha256, env_fingerprint_sha, lever_fingerprint, max_seq_len_effective, truncated_count, buffer_digest_entry/exit, model_builds, pid, merged_adapters_at_exit, consumed_ids_sha256, consumed_count, resume_count, max_mem_bytes, wall_s`

**감사 규칙 15종** (하나라도 위반이면 run 무효, 머지 차단)

| # | 규칙 |
|---|---|
| ① | **셀 부재 자체가 실패** (R×3 전 셀). 실패 클라이언트가 로그에 아예 없는 경우까지 잡는다 |
| ② | `steps_ran == steps_planned` |
| ③ | 클라이언트별 `Σ steps_ran == R × S_k` |
| ④′ | 실현 epoch 등가 `R·S_k·B_eff / n_k_effective`가 3.0 ±1% |
| ④″ | `n_k_declared − dropped_long_samples == n_k_effective` 정확 일치 |
| ④‴ | **통합·중앙 총 옵티마이저 스텝 == `Σ_k R·S_k`** (잔차 ≤ 9). 두 칸 총 갱신 횟수 동일성을 숫자로 증명 |
| ⑤ | 해시 컬럼 전 셀 단일값 (`realized_adapter_sha256`·`base_*`·`preproc_*`·`prompt_*`·`env_*`·`lever_*`) |
| ⑥ | `model_builds == 1`, `(pid, _BUILDS)` 쌍이 셀마다 유일하고 fit 순서와 1:1 증가 (판정 12) |
| ⑦ | `lr_first(r)`가 단일 cosine 위 (1e-9), `Σ_t (last_epoch_out − step_offset_in) == R·S_k` |
| ⑧ | `merged_adapters_at_exit`가 전 셀 빈 리스트 |
| ⑨ | `arg_optimizer` 계열이 실사용 `optimizer`와 일치, `truncated_count == 0`, `max_seq_len_effective == 2048` |
| ⑩ | `n_k_effective`가 `train_index` sha와 일치 (manifest 파생 금지) |
| ⑪ | **`adapter_l2_B > 0` 전 셀** (gradient checkpointing 사고 탐지) |
| ⑫ | 라운드 t 글로벌 어댑터 norm != t−1 |
| ⑬ | `consumed_count == samples_seen`, 전 셀 합집합 ∩ eval == ∅ |
| ⑭ | `exposure_max − exposure_min ≤ 1` **and** `coverage == 1.0` **and** 같은 클라이언트의 `sample_index_digest`가 라운드마다 서로 다름 (판정 8) |
| ⑮ | `clip_hit_count` 비율의 칸 간 비대칭을 보고 항목으로 승격 |

**`record(cell)`이 호출될 때마다 CSV에 append-only 즉시 flush**하고 `from_csv`로 서버 기동 시 복원한다(F22). 재시작 후 감사가 "9개 셀이 비어 있다"로 run을 무효 판정하는 경로를 막는다. 중복 셀은 `(round, client)` 키로 마지막 것만 유효하되 중복 발생 자체를 로그에 남긴다.

`tests/test_accounting_parity.py`가 검출 `budget_audit.py`와 **같은 합성 셀 집합에 같은 판정**을 내는지 고정한다(코드 통합은 하지 않는다 — 머지된 검출 회계를 건드리는 위험이 이득보다 크다).

**`fl/client_vlm.py`**
- `build_client_model`을 **fit마다 새로** 만든다. 모듈 레벨 캐시·`lru_cache` 금지. `device_map={"": 0}` 하드코딩(**`"auto"` 금지**). fit 진입에 `reset_peak_memory_stats()` + `assert memory_allocated() < 64<<20` + `_BUILDS`·`pid` 기록(판정 12).
- 주입은 `adapter_io.inject` 단일 진입점. 학습은 `vlm.round_runner.train_round` 재사용 — **통합·중앙과 문자 그대로 같은 함수**.
- fit 종료 시 `get_model_status().merged_adapters == []`, 명시 teardown(`del → gc → empty_cache`), `max_memory_allocated` 기록.
- 조기 종료 장치 부재: `EarlyStoppingCallback` 미등록, `load_best_model_at_end=False`, `metric_for_best_model` 미설정. 정지 조건은 오직 `S_k` 스텝 도달.

**`fl/server_vlm.py`**
- 기동 시 `outputs/gates/exchange_audit_uni.json`·`canary1/{key}.json`의 `pass`를 확인하지 못하면 **예외**. 게이트를 실행 조건으로 만든다.
- `init_global_adapter`: **PEFT 경로 재사용**(F13) + `adapter_init_{cell}.json` 바이트 동일 게이트.
- `aggregate_train` 진입 즉시: `valid, errors = _check_and_log_replies(...)` → `if errors: raise` → `if len(valid) != K: raise` → client_id 정렬 → `n_k_effective` 대조 → 해시 단일값 대조 → `weighted_adapter_average` → comms·accounting 기록 → 체크포인트(판정 14).
- `fraction_train=1.0`, `min_train_nodes = min_available_nodes = K`, **`fraction_evaluate=0.0`, `min_evaluate_nodes=0`을 인자 노출 없이 하드코딩.** `configure_train`에서 `sample_nodes` 반환 길이 != K면 라운드 시작 전 실패.
- **서버 프로세스에서 모델을 만들지 않는다.** val loss는 (a) 지정 클라이언트 1개가 같은 액터 안에서 계산해 스칼라만 반환하거나 (b) 글로벌 어댑터를 디스크에 쓴 뒤 `subprocess`로 별도 프로세스에서 계산·종료. 액터가 R=6 내내 살아 CUDA 컨텍스트·워크스페이스를 쥐고 있으므로 드라이버에서 8 GB를 더 올리면 라운드 1 aggregate 직후 OOM이다. `server_vlm.py` 모듈 상단 `transformers` import 금지를 AST로 강제하고, 라운드별 `nvidia-smi --query-compute-apps` 스냅샷을 `outputs/gates/`에 남겨 프로세스가 2개가 아님을 증빙한다.
- **라운드 중 글로벌 평가셋 접근 0.** 코드에 경로를 두지 않는다.
- 진입에서 `guard_no_cloud_logging()` + **`assert_local_tracking()`** + `guard_no_egress()`, 종료 시 `require_tags()`.

**테스트** `tests/test_accounting_vlm.py`, `tests/test_uni_fed_protocol.py`
- `test_빈_셀은_run_무효다` / `test_steps_ran이_S_k와_다르면_실패`
- `test_실현_epoch등가가_허용오차를_벗어나면_실패` / `test_n_k_declared_빼기_dropped가_n_k_effective와_같다`
- `test_두_칸_총_옵티마이저_스텝이_잔차_9이내로_같다`
- `test_해시컬럼이_셀마다_다르면_실패` / `test_model_builds와_pid쌍이_유일하다`
- `test_adapter_l2_B가_0이면_실패한다` / `test_coverage와_exposure_규칙`
- `test_라운드4에서_죽고_재시작해도_회계가_R곱3셀을_채운다`
- `test_실패_클라이언트가_있으면_라운드가_중단된다` / `test_노드가_덜_샘플링돼도_중단된다`
- `test_fraction_evaluate가_0이라_클라이언트_evaluate가_호출되지_않는다`
- `test_3클라이언트_수신본_sha256이_같다` / `test_라운드0_lora_B_norm이_0이다`
- `test_병합은_최종라운드_후_agg_tag별_1회만` / `test_라운드중_병합_시도는_예외`
- `test_최종_체크포인트가_4bit_base에서_병합되지_않았다`
- `test_latest슬롯에서_재시작해도_LR오프셋과_norm궤적이_이어진다`
- `test_server_vlm_상단에_transformers_import가_없다` (AST)

---

## 4-20. `vlm/postprocess.py` · `vlm/infer_eval.py` · `vlm/pilot_budget.py` — 평가 추론

```python
# postprocess.py — 역변환 1지점. 실패는 계약 #4 어휘로 분류만 하고 고치지 않는다
def classify_and_build(*, image_id, cell, seed, text, stop, geom, cfg, allowed_iso,
                       size_basis_by_iso, raw_ref, latency_ms, gen_tokens,
                       coord_cfg_hash_value) -> RecordOutcome
def normalize_iso(code: str) -> tuple[str, bool]          # alt → 대표. label_map 파생, 분리형과 같은 함수
def bbox_size_px(bbox_px, basis) -> float
def coord_signature(model_boxes, cfg) -> dict[str, float]
def failrate_report(outcomes) -> dict

# infer_eval.py — greedy 1회 · batch 1 · 재시도 분기 없음
def run_eval(*, cfg_path, cell, seed, ckpt_dir, manifest_csv, out_dir, device="cuda") -> Path
```

**핵심 결정**
- **판정 순서를 코드 순서로 고정**: ① `stop == "length"` → `truncated` ② `no_json` ③ `json_decode` ④ `schema_violation` ⑤ `unknown_iso_code`(정규화 후에도 미매칭) ⑥ 역변환 → bbox 판정. **앞 단계에서 실패하면 역변환에 진입하지 않는다**(12a §3).
- `stop == "length"`인데 JSON이 온전하면 `truncated` 오답으로 확정하되 **`truncated_but_parsable`·`stop_token_missing`·`budget_exhausted` 3종 감사 카운터를 따로 남긴다.** "능력 부족"·"예산 부족"·**"정지 토큰 미지정"**을 섞지 않기 위해서다 — 세 번째 원인이 관점 3의 어휘에 없었고 그것이 F20이다. 계약 #4 어휘는 `truncated` 하나로 유지하고 쪼갠 값은 `failrates.json`에만 낸다.
- **ISO 대체 코드 정규화**(구조 비대칭 제거): `label_map.yaml`에 `iso_code_alt: ["2012"]` 같은 유효 대체 코드가 실재한다. 분리형은 검출기 클래스 인덱스 → label_map 경로라 `unknown_iso_code`가 **구조적으로 불가능**한데, 통합형만 alt를 뱉었다고 레코드 전체(verdict·cited_clauses 점수 포함)가 죽으면 **분리형에서 발생할 수 없는 실패가 통합형에만 발생**하고 그 차이가 RQ2에 실린다. 이것은 답을 고치는 것이 아니라 분리형이 이미 하는 사상을 통합형에도 동등하게 적용하는 것이다. `allowed_iso = 대표 ∪ alt`, 정규화는 D 어댑터가 분리형 `cls`→ISO에 쓰는 **같은 함수**, 정규화 건수는 별도 카운터.
- **부분 이탈 박스는 값 그대로 통과시키고 카운터만 올린다.** 클램프는 IoU를 올리는 방향으로만 작동하므로 답을 고쳐 주는 것이다. `snap_to_bounds`는 호출하지 않는다(판정 6). 이탈량 분포를 `failrates.json`에 남겨 D가 스냅 발생률과 대조하게 한다.
- **역변환 정수화 0회**(판정 5). `bbox_px`는 float.
- `bbox_size_px`: `major_axis = max(w,h)`, `equiv_diameter = sqrt(4·w·h/π)`. **통합형과 분리형이 같은 함수를 써야** size 계열 비교가 성립하므로 `detection/export_preds.py`가 import한다(소유 위치는 Q4).
- `coord_signature`: 모델 좌표 max 분포·선언 공간 범위 내 비율·역변환 후 경계 밖 비율. **좌표계 사고는 loss에 안 보이므로 이 세 값이 학습 중 유일한 조기 신호다.**
- `run_eval` 시작 순서: `guard_no_cloud_logging()` → `assert_local_tracking()` → `check_required_tags()`(**추론 전에** 실패하게 한다. 평가셋 12,600장을 끝낸 뒤 `MissingRunMetadata`로 죽으면 GPU-시간이 통째로 날아간다) → `reject_best_checkpoint(ckpt_dir)` → `assert_size_not_overridden` → `assert_eval_runtime(model)` → `require_canary_pass` → 프롬프트 SNAPSHOT 대조 → `assert_generation_prompt` → GenerationConfig bind.
- 출력: `outputs/{cell}/{seed}/generations.jsonl` + `raw_texts.jsonl` + `failrates.json` + `eval_access.jsonl` + `SNAPSHOT.sha256`. **계약 #4 §2-3의 키를 유지한다** — `text`(전문)와 `bbox_px_parsed`. 관점 3이 제안한 `text_ref`/`defects_px` 개명은 D의 `unified.py`가 `rec["text"]`를 읽으면 전량 KeyError, `rec.get("text","")`면 **전 레코드 `no_json`으로 0점이 정상 채점된 것처럼** 보고되는 경로를 만든다. 계약 개정을 코드보다 먼저 확정하되, 기본값은 **키 유지**다. `resized_wh`·`vision_tokens`는 감사용 추가 필드이며 공통 스키마 jsonl로는 넘어가지 않는다(계약 #4는 `additionalProperties: false`).
- **`raw_output_ref`는 위치가 아니라 내용 주소**: `"raw_texts.jsonl:{image_id}"`. 위치 참조는 재개·부분 재실행이 도입되는 순간 조용히 다른 생성문을 가리키고, 스키마 검증은 형식만 보므로 전부 통과한다. D 어댑터가 두 파일의 `image_id` 집합 일치를 assert한다.
- **재개**(F22): 이미지마다 append+flush + `progress.json`{run_id, manifest_sha256, prompt_bundle_sha, gen_effective_sha, coord_cfg_hash, env_sha, last_image_id, n_done}. 종료 시 image_id 정렬 정규화본으로 rewrite하고 `SNAPSHOT.sha256`에 기록해 "재실행 비트 일치"는 정규화 시점에서 보장한다. `eval_access.jsonl`은 **append-only 해시 체인**이고 각 줄에 `{ts, run_id, manifest_sha256, rows_read, reason: initial|resume}`. **재개가 "표현 가능한 사실"이어야 격리 증거가 산다 — 표현할 수 없으면 지우게 된다.** 재개 시 6해시 불일치면 거부하고 전량 재실행을 요구한다.
- **루프에 재생성·재프롬프트 분기가 존재하지 않는다.** 예외가 나도 다시 부르지 않고 `parse_ok=false` 레코드로 계상한다 — 이미지를 빠뜨리면 오답보다 낙관적으로 잡힌다.
- latency는 `torch.cuda.synchronize()`로 감싼 전처리+생성+후처리 구간을 batch=1로 잰다. 모델 로드 제외.
- 이 진입점이 `split == "eval"`을 읽는 **유일한 곳**이고 **`limit` 인자가 없다**(F17).

**추론 OOM에는 사다리가 없다.** 12_spec_C §5-1 사다리는 배치↓ → grad accum↑ → gradient checkpointing인데 배치는 이미 1로 고정돼 있고 accum·ckpt는 학습 전용, `max_pixels` 하향은 검토 결정 C로 금지다. 따라서 추론 OOM은 **G7-B1(최악 배치 프로브)에서 착수 전에 잡아야 하며**, 발생 시 legal한 조정 방향은 `size.longest_edge` 재동결(단 **어느 통합형 칸도 돌기 전에만**) 또는 표본 축소뿐이다. 배치 확대는 디코딩 고정 항목 변경이므로 금지.

### `pilot_budget.py`

```python
def run_pilot(*, split: Literal["val"], cell, seed, ckpt_dir, n: int) -> Path
```
생성자에서 `split == "eval"`이면 예외. `max_new_tokens`·`truncated_rate`·`eos_stop_rate`·`parse_rate`·`images/s`·`p50/p99 latency`·`peak VRAM`·`예상 총 wall clock`을 산출하고 G9 게이트 레코드에 넣는다.

**테스트** `tests/test_vlm_postprocess.py`, `tests/test_vlm_infer_eval.py`, `tests/test_vlm_no_retry.py`

| 테스트 | 검증 대상 |
|---|---|
| `test_판정순서_truncated_우선` | ① 우선 |
| `test_stop_length인데_JSON이_온전하면_truncated_오답_and_stop_token_missing_카운트` | 원인 3종 분리 |
| `test_alt_iso코드가_대표코드로_정규화된다` | 구조 비대칭 제거 |
| `test_정규화_건수가_별도_카운터로_남는다` | |
| `test_부분이탈_박스는_값_무수정_통과_and_out_of_bounds_카운트` | 보정 0회 |
| `test_교집합0_퇴화는_bbox_invalid` | |
| `test_vlm전체에_snap_to_bounds_호출_0건` | AST |
| `test_bbox_size_px_두_basis_수식` | |
| `test_coord_signature_3종_스칼라` | 좌표계 사고의 유일한 조기 신호 |
| `test_failrate_report_합계_무결성` | |
| `test_generations_jsonl에_text와_bbox_px_parsed_키가_있다` | 계약 준수 |
| `test_generations_jsonl에_bbox_2d나_model계열_키가_없다` | 모델 좌표 저장 금지 |
| `test_raw_output_ref가_image_id_기반이다` | 위치 참조 금지 |
| `test_run_eval에_limit_인자가_없다` | AST |
| `test_run_pilot이_split_eval을_거부한다` | |
| `test_eval_access_jsonl이_append_only_해시체인이다` | |
| `test_재개시_여섯_해시가_다르면_거부한다` | |
| `test_generate_호출이_샘플당_1회이고_except_while_밖에_있다` | AST |
| `test_retry_resample_regenerate_심볼_부재` | AST |
| `test_vlm_어느_모듈도_최상단에서_transformers_torch를_import하지_않는다` | 현 venv에서 760건 스위트 유지 |

---

## 4-21. 게이트 실행체 — `vlm/canary_shapes.py` · `vlm/canary.py` · `vlm/gate.py` · `vlm/selfcheck.py`

```python
# canary_shapes.py
def build_specs(*, seed, min_pixels, max_pixels, n=19) -> tuple[ShapeSpec,...]
def confusion_iou(spec, geom, a: CoordCfg, b: CoordCfg) -> float
def assert_separable(specs, geoms, cfgs, *, max_confusion_iou=0.5, min_images=8) -> None
def probe_sha256() -> str          # 프로브 프롬프트 + 도형 스펙 → gate_key 입력

# canary.py
def run_canary1(*, model, processor, cfg, facts, prompt, decode, specs, tag: Literal["1a","1b"]) -> CanaryResult
def run_canary2(*, manifest_csv, ...) -> CanaryResult    # 생성자에서 split!="eval" 강제

# gate.py
def gate_key(**kwargs) -> str
def require_canary_pass(*, cfg, key, dir=Path("outputs/gates/canary1")) -> dict
def require_gate(name, *, key=None) -> dict

# selfcheck.py
def selfcheck_pairs(pairs_jsonl, *, cfg, geom_of) -> SelfcheckReport
def write_train_index(report, path) -> Path
```

`IoU`는 `evaluation.metrics.localization.iou`를 import한다 — 카나리아와 채점기가 다른 IoU 정의를 쓰는 상태를 만들지 않는다.

**테스트** `tests/test_vlm_canary_gate.py`, `tests/test_vlm_selfcheck.py`
- `test_축퇴집합(1280x720_큰박스만)을_기각한다`
- `test_하단_소형도형_집합을_통과시킨다` — 실측 혼동 IoU 0.156 회귀 고정
- `test_합성_모델출력으로_argmax가_정답규약을_고른다`
- `test_margin_015_미달시_보류` / `test_파싱률_09_미달시_실패`
- `test_게이트레코드_부재_passed_false_내용해시불일치_키불일치에서_전부_예외`
- `test_프로브를_고치면_키가_바뀌어_이전기록이_무효가_된다`
- `test_attempt_seq_3부터는_총괄승인_필드가_없으면_거부한다`
- `test_canary2_로더가_split_eval_행을_거부한다`
- `test_1a와_1b의_argmax가_갈리면_에스컬레이션_플래그가_선다`
- `test_selfcheck_기준이_IoU가_아니라_roundtrip_budget_px다` — 소형 박스 케이스
- `test_규약을_어긋나게_준_페어에서_위반이_전량_검출된다`
- `test_train_index가_생성되고_sha256이_봉인된다`
- `test_클라이언트별_drop_rate_상한과_편차_게이트`

---

## 4-22. 부록 산출물 — `fl/agg_error_probe.py` · `fl/dw_project.py`

**1단(필수)**: 마일스톤 페이로드로 `ε_l`·`ε'_l` 층별 실측. GPU 0, 평가 데이터 무접촉. `fro_gap`과 `cross_cov_fro`를 **별도 경로로 계산해 일치 여부를 CSV 열에 남긴다** — 논문 대수 주장이 코드에서 성립함을 산출물 자체가 증명한다.

**2단(권장)**: 최종 라운드에서만 집계 규칙을 갱신량 평균으로 바꾼 대안 어댑터.

**dense SVD 금지**: `ΔW = Σ_k p_k B_k A_k`의 랭크는 정의상 `K·r = 48` 이하다. mlp 층 하나가 9,216×2,560 fp32 = 94MB이고 248층이면 2~4시간이다.
```
B_cat = [√p_1·B_1 | … | √p_K·B_K]   (d_out × 48)
A_cat = [√p_1·A_1 ; … ; √p_K·A_K]   (48 × d_in)
Q_B,R_B = qr(B_cat);  Q_A,R_A = qr(A_cat.T)
U,S,Vt = svd(R_B @ R_A.T)            # 48×48
B' = Q_B @ U[:, :r] @ sqrt(S[:r]);   A' = sqrt(S[:r]) @ Vt[:r] @ Q_A.T
```
`ΔW = B_cat·A_cat`이 **정확히** 성립하므로 결과는 dense SVD와 수학적으로 동일한 최적 rank-r 근사이고 층당 메모리는 MB, 시간은 밀리초다.

**채점은 val loss(teacher forcing)만.** 글로벌 평가셋으로 두 집계 규칙을 비교하면 평가셋에 선택압이 생긴다. **본문 표는 언제나 `agg=matrix_avg`**이며, 평가셋 행이 필요하면 메인 런 시작 **전에** 총괄 사전 등록.

**테스트** `tests/test_dw_project.py`
- `test_QR코어_SVD가_dense_SVD와_bit수준_근사_일치` (작은 픽스처)
- `test_클라이언트_1개면_투영이_원_어댑터와_같은_ΔW를_준다`
- `test_dense_ΔW를_전량_적재하지_않는다`
- `test_agg_tag가_주표_필터에서_배제된다`

---

# 제5부 — 절대 규칙 최종 스윕

| 절대 규칙 | 이행 지점 | 큰 실패 |
|---|---|---|
| **조기 종료 금지** (불변조건 3-1) | `vlm/sft_loop.py`가 `Trainer`·`SFTTrainer`·`EarlyStoppingCallback`·`load_best_model_at_end`·`metric_for_best_model`·`should_training_stop`·`save_total_limit`를 **import도 참조도 하지 않는다**(소스 스캔). 유일한 종료는 `StepBudget`. NaN·inf 손실은 스킵이 아니라 run 실패. `clip_grad_norm_(error_if_nonfinite=True)`로 "조용히 0으로 눌린 스텝"까지 실패로 승격 | ✅ |
| **best 금지 · last 채점** (3-2) | 산출 디렉터리 `last/` 고정. 루프 안에 val 지표 접근 코드가 없다(val loss는 호출자가 라운드 사이에 부르는 별도 함수 + 별도 프로세스라 루프가 구조적으로 val로 분기할 수 없다). `reject_best_checkpoint`를 병합·채점·**저장 시점**에 호출. `assert_final_only(round_idx, rounds, agg_tag)` | ✅ |
| **학습량 등가 R×E=N** | `plan_steps` 단일 공식(`floor(x+0.5)`, `fl/seeding.py` 소유), 분모 `n_k_effective`. 증인 4종. 회계 규칙 ②③④′④″④‴⑨⑭. 단일 가상 순열로 표본 노출까지 등가. 잔차 ≤9 실측 보고 | ✅ |
| **재시도·재프롬프트 금지, greedy 1회** | 추론 루프에 재생성 분기가 존재하지 않는다(AST). `assert_greedy`에 완화 스위치 없음. 예외가 나도 다시 부르지 않고 `parse_ok=false`로 계상. `batch_size=1` 고정 | ✅ |
| **스키마 위반은 오답 + 실패율 별도 보고** | `classify_and_build`가 계약 #4 어휘로만 분류, `failrates.json` + MLflow 메트릭. 어휘 부분집합 테스트 | ✅ |
| **5칸 공통 고정 8종 → 9종** | 제3부 표. `FIXED_UNI` 검사 대상이 "쓰일 값"(판정 19). `CellFingerprint` 확장(판정 20). 실현 산출물 해시(판정 21) | ✅ |
| **min/max_pixels 프로파일 override 금지 (Q15)** | `base.yaml` 단독 소유 + 프로파일 diff 화이트리스트(판정 18) + `encode_inputs` 단일 호출 경로 + 매 호출 면적 assert. OOM 사다리에서 제거 | ✅ |
| **외부 클라우드 로깅 금지 (2-3)** | `guard_no_cloud_logging()` + **`assert_local_tracking()` 신설**(`MLFLOW_`·`DATABRICKS_` 접두사 추가, `mlflow.get_tracking_uri()`가 `sqlite:///` + 저장소 내부 경로인지 실측 확인). Trainer 미사용이라 `report_to` 자동 연동 경로 없음. **`HF_HUB_OFFLINE=1` 강제**(`save_embedding_layers="auto"`의 Hub 조회 차단) | ✅ |
| **평가 자산 격리 (1-4)** | `split=="eval"`·`eval_subset` 유입 시 즉시 실패. `run_eval`이 eval을 읽는 유일 진입점이고 `limit` 인자 없음. `pilot_budget`은 val 전용. `eval_access.jsonl` append-only 해시 체인. 게이트 프로브는 합성 픽스처만. 전 셀 `consumed_image_ids` ∩ eval == ∅ | ✅ |
| **매니페스트 단일 진실 (1-2)** | A 승인 로더 단일 경로, `pandas.read_csv` 직접 호출 금지, 페어 값은 대조용 | ✅ |
| **스냅샷 고정·재생성 금지 (1-6)** | 프롬프트 `PROMPT_FROZEN` 마커, `train_index` sha256 봉인, corpus·페어·색인 해시 | ✅ |
| **라벨 하드코딩 금지 (1-8)** | 프롬프트는 `label_map.yaml`에서 생성. ISO 정규화는 label_map 파생 함수 | ✅ |
| **공통 스키마 · 단일 채점기 (3-7)** | `vlm/target_format.py`를 D가 import(이중 추출기 차단). `generations.jsonl` 계약 키 유지 | ✅ |
| **bbox는 원본 픽셀 저장 (3-8)** | 모델 좌표는 dataclass 안에만. 디스크 쓰기 API 없음. 역변환 정수화 0회 | ✅ |

**잔여 위험 1건**: 게이트 레코드는 파일 시스템에 있고 내용 해시로 자기 검증하지만 **의도적 위조를 막지는 못한다.** 방어는 MLflow run id 동반 기록 + `attempt_seq` + gate_key 재계산이며, 최종 방어선은 D의 골든 픽스처 독립 재실행·D1 분포 검사·시각 카나리다.

---

# 제6부 — 구현 순서

원칙: **D가 기다리는 것 → 게이트가 요구하는 것 → 학습 → 연합 → 평가 → 부록.** 각 Phase는 그 앞 Phase의 테스트가 전부 통과해야 시작한다.

## Phase 0 — 계약 해소 (0.5일, 코드 없음)

| 할 일 | 상대 |
|---|---|
| `pyproject.toml` 정확 핀 + `uv lock` 커밋 → **G0** | C 단독 |
| `configs/gpu/*.yaml`에서 `max_seq_len_qa`·`max_seq_len_reason` 삭제 → `base.yaml` | A |
| `defect_instance_id ↔ ann_id` 사상 규칙 확정 요청 | B |
| 스펙 §2-1(TRL) · Q7(셔플) · 결정 F(n_k) 개정 요청 | 총괄 |
| `generations.jsonl` 키 계약 재확인(변경 없음 통보) | D |

## Phase 1 — 리프 3종 (1일) · **D 언블록**

| 순서 | 경로 | 대기자 |
|---|---|---|
| 1 | `vlm/target_format.py` + `tests/test_vlm_target_format.py` | **D의 `unified.py` 어댑터** |
| 2 | `fl/seeding.py` + `detection/round_runner.py` 재수출 + `tests/test_fl_seeding.py` | E(총 스텝 수 재계산), D |
| 3 | `vlm/loss_norm.py` + `tests/test_vlm_loss_norm.py`(더미 모델 부분) | 분리형 판정부(정규화 주입) |

`vlm/coords.py`는 **손대지 않는다**. 착수 전·후 `python vlm/coords_fixtures/run_fixtures.py` 12/12 통과와 `coords.py` sha256 불변 확인.

## Phase 2 — 설정과 접합 (2일)

| 순서 | 경로 | 산출 게이트 |
|---|---|---|
| 4 | `configs/base.yaml`(A와 공동) · `configs/quant_uni.yaml` · `configs/lora_uni.yaml` | |
| 5 | `vlm/fixed.py` + `tests/test_vlm_fixed_uni.py` | |
| 6 | `scripts/vlm_processor_probe.py` → `vlm/coord_runtime.py` + `tests/test_vlm_coord_runtime.py` | **G3** |
| 7 | `scripts/pick_longest_edge.py` → `size.longest_edge` 동결 | **G3-P6** |
| 8 | `vlm/prompts/*` + `scripts/build_unified_prompt.py` + `vlm/prompt_store.py` + `tests/test_vlm_prompts.py` | |

## Phase 3 — 모델 계층과 착수 게이트 (3일)

| 순서 | 경로 | 산출 게이트 |
|---|---|---|
| 9 | `scripts/gate_structure_probe.py` (meta device) | **G1** |
| 10 | `vlm/model_io.py` + `tests/test_vlm_model_io.py` | |
| 11 | `scripts/dump_target_modules.py` → `configs/target_modules_uni.json` + `tests/test_target_modules.py` | **G4** |
| 12 | `fl/exchange_audit.py` + `scripts/gate_exchange_audit.py --stage structure\|runtime` + `tests/test_exchange_audit.py` | **G2** |
| 13 | `vlm/generation.py` + `tests/test_vlm_generation.py` | |
| 14 | `vlm/canary_shapes.py` · `vlm/canary.py` · `vlm/gate.py` · `scripts/vlm_canary1.py` + `tests/test_vlm_canary_gate.py` | **G5 (1a → 1b)** |

> **G2와 G5를 통과하지 못하면 Phase 4 이후 코드는 실행되지 않는다.** 작성은 병행 가능하되 학습 스텝은 0이다.

## Phase 4 — 데이터와 자기채점 (2일)

| 순서 | 경로 | 산출 게이트 |
|---|---|---|
| 15 | `vlm/collate.py` + `tests/test_vlm_collator.py` | |
| 16 | `vlm/pair_dataset.py` + `tests/test_vlm_dataset.py` | |
| 17 | `vlm/selfcheck.py` + `scripts/vlm_target_selfcheck.py` + `tests/test_vlm_selfcheck.py` → `train_index` 봉인 | **G6** |
| 18 | `vlm/corpus_dataset.py` (배경지식 D3-(b)/(c)) | |

## Phase 5 — 학습 루프와 예산 (2일)

| 순서 | 경로 | 산출 게이트 |
|---|---|---|
| 19 | `vlm/sft_loop.py` + `tests/test_vlm_loop_budget.py` · `test_vlm_lr_schedule.py` | **G8** |
| 20 | `scripts/vram_probe.py` · `scripts/step_time_probe.py` | **G7** |
| 21 | `vlm/round_runner.py` + `tests/test_vlm_round_equivalence.py` | |
| 22 | `vlm/train_cell.py` → **배경지식 SFT 2단 실행 → 병합본 동결 → G5-1b 재실행** | |
| 23 | **통합·중앙 본학습** | |

## Phase 6 — 연합 (3일)

| 순서 | 경로 |
|---|---|
| 24 | `fl/adapter_io.py` + `tests/test_adapter_io.py` |
| 25 | `fl/adapter_aggregate.py` + `tests/test_adapter_aggregate.py` |
| 26 | `fl/accounting_vlm.py` + `tests/test_accounting_vlm.py` · `test_accounting_parity.py` |
| 27 | `fl/comms.py` + `tests/test_comms_ledger.py` |
| 28 | `fl/checkpoint.py` · `fl/client_vlm.py` · `fl/server_vlm.py` + `tests/test_uni_fed_protocol.py` |
| 29 | **통합·연합 본학습** (R=6) |

## Phase 7 — 평가 (2일 + 추론 시간)

| 순서 | 경로 | 산출 게이트 |
|---|---|---|
| 30 | `vlm/postprocess.py` + `tests/test_vlm_postprocess.py` |  |
| 31 | `vlm/pilot_budget.py` → val 스모크 | **G9** |
| 32 | `vlm/infer_eval.py` + `tests/test_vlm_infer_eval.py` · `test_vlm_no_retry.py` | |
| 33 | `tests/test_vlm_oracle_injection.py` — GT를 가짜 생성문으로 만들어 종단 통과, IoU 1.0 확인. **역변환 이중 적용·미적용을 잡는 유일한 종단 검사** | |
| 34 | **통합 2칸 eval 추론 1회** | |

## Phase 8 — 부록·표 (1일)

| 순서 | 경로 |
|---|---|
| 35 | `fl/agg_error_probe.py` (1단) |
| 36 | `fl/dw_project.py` + `tests/test_dw_project.py` (2단) |
| 37 | `scripts/comms_table.py` + `tests/test_comms_ledger.py` |

---

# 제7부 — 열린 질문과 안전 기본값

| # | 질문 | 안전 기본값 (승인 전 이대로 진행) | 판단 주체 |
|---|---|---|---|
| **Q1** | `CoordCfg`에 `size_shortest_edge`·`size_longest_edge`·`max_seq_len` 필드를 추가할 것인가 | **추가하지 않는다.** 대신 `preproc_sha256`을 `REQUIRED_TAGS` + `check_cells_identical` 비교 필드에 넣고, `base.yaml` sha256을 `gate_key`에 봉인한다. 같은 방어를 `coords.py` 수정 없이 얻는다 | **총괄** — 수정하면 골든 픽스처 12/12·`SHA256SUMS`·5칸 `coord_cfg_hash`가 동시에 깨지고 전 게이트 재봉인이 따라붙는다 |
| **Q2** | TRL 미채택(스펙 §2-1 문면 개정) | 직접 루프로 진행. `run_steps` 공개 표면 프레임워크 중립 유지 | **총괄** |
| **Q3** | `defect_instance_id ↔ ann_id = f"{image_id}#{i}"` 사상 규칙 | **확정 전에는 assert로 막고 진행하지 않는다.** 타깃 생성 불가 | **B·C 공동 → 총괄 확정** |
| **Q4** | `bbox_size_px` 소유 위치 | `vlm/postprocess.py` 임시 소유, `detection/export_preds.py`가 import. 중복 구현 금지 | **총괄** — 정상 소유처는 A의 `data/convert/geometry.py`이거나 신설 공용 리프 |
| **Q5** | 16bit 그레이스케일 RT 원본의 8bit 변환 규칙 | `vlm/coord_runtime.load_image()`에 명시 함수 + mode 화이트리스트. **분리형과 동일 규칙 선언이 필수** | **A·C 공동** — 갈리면 RQ2에 전처리 비대칭이 교란으로 들어간다 |
| **Q6** | `n_k` 이원화(결정 F 개정) | `n_k_effective`를 `S_k`·FedAvg 가중 분모로. 두 값 모두 회계 기록 | **총괄** |
| **Q7** | 표본 노출: 단일 가상 순열(Q7 개정) | 전역 커서 슬라이스로 진행 | **총괄** |
| **Q8** | FedAvg 가중을 `T_k`(총 총괄 토큰)로 전환 | `max(τ_k)/min(τ_k) ≤ 1.02`면 `n_k_effective` 유지. 초과 시에만 전환 제안 | **총괄** (파일럿 결과 첨부) |
| **Q9** | `merger`(`model.visual.merger.linear_fc1/2`) 어댑터 포함 여부 | **제외**(현행 비전 동결 결정 따름). 통합 2칸 동일 적용이라 칸 간 공정성은 유지되나 "통합형 grounding이 약하다"의 귀속이 흐려진다 | **총괄** — 포함 시 어댑터 파라미터가 늘고 공통 고정 2번이 바뀐다 |
| **Q10** | `linear_attn.in_proj_a`/`in_proj_b`(SSM 게이팅·Δ 프로젝션, 각 995,328 파라미터) 어댑터 부착 | **포함**(언어부 전 층 대칭). 저랭크 갱신 평균의 SSM 안정성에 선행 사례가 없다 | 파일럿에서 제외 변형 1회 비교 후 동결 권고 |
| **Q11** | 48GB(Linux) 이관 | **G7-B3~B5 결과로 착수 전 결정.** 이관 시 통합 2칸을 **동시에** 옮기고 둘 다 재실행 | **총괄** (GPU-일 실측 첨부) |
| **Q12** | Windows에서 `triton-windows` + `fla-core`로 fast path를 켤 수 있는가 | G7-B5에서 실측. 켜지면 패키지·버전을 `env_fingerprint`(공통 고정 9번)에 편입. 못 켜면 폴백 확정 사실과 총 GPU 시간을 논문 §8에 싣는다 | C 실측 후 보고 |
| **Q13** | Flower API(레거시 vs Message) | G0에서 실측 확정. 어느 쪽이든 전략이 실패·참여 수를 직접 막는다 | C 실측 후 레지스트리 #2·#10 갱신 |
| **Q14** | `unknown_iso_code`를 레코드 kill로 둘지 결함 단위로 둘지 | **레코드 kill + 결함 단위 감사 카운트 병행.** 두 방식의 지표 차이를 파일럿에서 실측 보고 | **D 확인 → 총괄** (계약 #4 §2-5와 §3-4가 서로 다른 단위를 전제한다) |
| **Q15** | 부록 2단(dW)을 글로벌 평가셋으로 채점할 것인가 | **하지 않는다.** val loss만. 본문 표는 언제나 `agg=matrix_avg` | **총괄 사전 등록 필요** (메인 런 시작 전) |
| **Q16** | `size.longest_edge` 최종값 | **1,048,576**(≈4,096 패치 = 1,024 비전 토큰) 사전 등록. `max_seq_len` 2048 유지 → 텍스트 예산 1,024 확보. 8/25 실데이터 확정 후 `pick_longest_edge.py`로 재확인하되 **어느 통합형 칸도 돌기 전에만** 변경 허용 | C 실측 → 총괄 동결 |
| **Q17** | 평가 정밀도 | **bf16 병합본 단일 경로**(4bit 평가 금지). `configs/gpu/*.yaml`의 `vlm` 블록을 `train:`/`eval:`로 쪼개고 `eval:`에는 양자화 키를 두지 않는다 | A와 공동 |
| **Q18** | 통합형 시드 | **1시드**(스펙 §1-5 확정). bnb 4bit·SDPA 폴백·recompute가 비트 결정론이 아니므로 이 한계가 특히 크다. 논문에 "시드 재현이지 비트 재현이 아니다"를 명시 | 확정 |

---

# 제8부 — 숨은 기본값 레지스트리 신규 등재 요청

기존 #1~#6에 이어 **#7~#18**을 등재한다. 전 건이 같은 형태다 — 선언과 실제 동작이 다르고 그 불일치가 지표에 흔적을 남기지 않는다.

| # | 프레임워크 · 기본값 | 깨뜨리는 선언 | 처방 |
|---|---|---|---|
| 7 | `Qwen3_5ForConditionalGeneration.accepts_loss_kwargs = False` + `forward`가 `**kwargs`를 `loss_function`에 전달하지 않음 → 멀티모달 VL 클래스는 `num_items_in_batch`를 **구조적으로 무시**하고 마이크로배치 토큰 평균만 반환 | 통합형·판정부 손실 정규화 동일(RQ2 통제) | `labels=None` forward + `vlm/loss_norm.py` 단일 공식. 분리형에도 같은 함수 주입. `effective_train_config`에 `loss_reduction`·`denominator_rule`·`model_class`·`accepts_loss_kwargs` 4필드 |
| 8 | `BitsAndBytesConfig.llm_int8_skip_modules`를 **명시하는 순간** transformers 기본 보호(lm_head·tied·output embeddings)가 통째로 폐기 (`quantizers/base.py` L233-247, `quantizer_bnb_4bit.py` L129-131이 `add_default_skips`를 안 넘김) | 5칸 공통 고정 1번 "같은 sha256에서 출발" | skip 목록을 `get_keys_to_not_convert(meta) ∪ {"visual"}`로 **합성**. 로드 후 `Linear4bit` 집합 sha256 + `lm_head` 비양자화 + tie 유지 assert |
| 9 | `BitsAndBytesConfig` 실제 인자명은 `bnb_4bit_*`. `quant_type=`·`use_double_quant=`·`compute_dtype=`는 **예외도 경고도 없이 무시**되고 fp4 / fp32 compute / double-quant 없음으로 돈다 | "QLoRA 4bit(NF4) + bf16 연산" | `configs/quant_uni.yaml` 명시 + `quantization_config.to_dict()` 키 집합 **완전 일치** 대조(교집합 아님) + 실체 3종 assert |
| 10 | `from_pretrained`의 `dtype` 기본값이 v5에서 `"auto"`(v4는 fp32) → 비양자화 2.4GB의 정밀도를 리포 `config.json`에 위임 | 5칸 공통 고정 1번 | `QLoraSpec.dtype='bfloat16'` 명시 + `declared_dtype` vs dtype 히스토그램 대조. `base_ckpt_sha256`에 `config.json` 포함 |
| 11 | PEFT `target_modules` 접미사 매칭 + 표준 Qwen 레시피 → **linear_attention 24개 층 전부 누락**(부착 128/248, 파라미터 35% 실종, 에러 없음) | 5칸 공통 고정 2번 | 접미사 앵커 목록 + 부착 모듈 수 248 assert + `realized_adapter_sha256` |
| 12 | `named_parameters(remove_duplicate=True)`(기본)가 tied weight를 dedup → `state_dict − named_parameters`가 `lm_head.weight`를 buffer로 **오탐** | buffer 감사의 정확성 | `remove_duplicate=False`로 뺄셈. tied 별칭은 `TIED_ALIAS` 별도 컬럼 |
| 13 | bnb `Linear4bit._save_to_state_dict`가 quant state를 `register_buffer` 없이 state_dict에 직접 주입 → `state_dict ∩ named_buffers` 정의로는 **구조적으로 보이지 않는다** | "전수 열거로 확인했다" | P/B/X 3집합 분류, X 전 원소 분류 강제, `base_quant_digest`를 packed weight + X∩QUANT_ARTIFACT로 재정의 |
| 14 | `use_kernel_func_from_hub_with_fallback`이 `try/except`로 **로그도 경고도 없이** 커널을 교체. `self.norm`은 flash-linear-attention 설치 여부에 따라 **클래스 자체가 바뀐다**. `_attn_implementation`은 32층 중 8층만 지배 | 5칸 공통 고정 "구조 내 동일" | `env_fingerprint`를 공통 고정 9번째 항목으로 승격 + 칸 간 대조 필드 |
| 15 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`가 Windows에서 **무동작**(`CUDAAllocatorConfig.h:35-45`가 무조건 false 반환) | OOM 사다리 칸 수 | 사다리에서 제거. `garbage_collection_threshold`·`max_split_size_mb` + 길이 버킷팅으로 대체 |
| 16 | `get_peft_model_state_dict(save_embedding_layers="auto")`가 판정 과정에서 **HF Hub로 네트워크 요청**(`save_and_load.py:433-449`). 승격 조건은 (a) target이 embed/lm_head 매칭 (b) vocab resize 2종 | 보안 레드라인 2절 · 통신량 서사 | `False` 명시 + 단일 진입점 + `HF_HUB_OFFLINE=1` 강제 + `guard_no_egress()` |
| 17 | `set_peft_model_state_dict`가 내부 `load_state_dict(strict=False)`를 쓰고 `_IncompatibleKeys`를 **반환값으로만** 알린다 → 키 접두사가 어긋나면 496개 키가 전부 `unexpected_keys`로 흘러가고 조용히 성공 | FedAvg가 무관한 3벌을 평균 | 반환값 assert + 주입 후 왕복 bit 검증. `low_cpu_mem_usage=False` |
| 18 | `torch.nn.utils.clip_grad_norm_`의 `error_if_nonfinite` 기본 False → inf 그래디언트에서 클립 계수 0이 되어 **그 스텝이 조용히 무효화되는데 세 증인이 전부 정상 증가** | 학습량 등가 | `True` + `grad_norm_max`·`clip_hit_count` 회계 컬럼 |
| 19 | Flower Message API `FedAvg`에 `accept_failures` 파라미터가 **없고** `aggregate_train`이 에러 응답을 버린 뒤 남은 것으로 집계. `fraction_evaluate` 기본 1.0 | R×E=N | 전략이 직접 검사(판정 14). #2·#10 갱신 |
| 20 | `Trainer.__init__`의 `set_seed` + generator 없는 `RandomSampler` → fresh Trainer가 **매번 같은 순열**을 낸다 | R×E=N(표본 수준) | 단일 가상 순열 + 전역 커서(판정 8) |
| 21 | `smart_resize`가 `round(x/factor)*factor`이고 파이썬 `round`는 banker's rounding → `ceil` 식 대비 약 5% 과대 추정 | `size.longest_edge` 영구 동결값 | 결정 자체를 코드로(`pick_longest_edge.py`) |
| 22 | `model.merged_adapters`는 `BaseTunerLayer` 클래스 속성이라 `PeftModel`에서 **AttributeError** | 병합 금지 가드 | `get_model_status().merged_adapters` |

---

# 제9부 — 실측 수치 부록 (게이트에서 갱신)

## 9-1. 모델 실물 (`Qwen/Qwen3.5-4B`, 총 4,539,265,536 파라미터)

| 항목 | 값 |
|---|---|
| `vocab_size` / `hidden_size` / `out_hidden_size` | 248,320 / 2,560 / 2,560 |
| `tie_word_embeddings` | `true` |
| `layer_types` | `linear_attention` 24 : `full_attention` 8 (`full_attention_interval=4`) |
| `named_buffers()` | 3개, 전부 `persistent=False`, `rope_type="default"`(불변) |
| tied alias | `lm_head.weight` 1건 |
| mlp / linear_attn / embed / visual / self_attn | 2264.9M / 1011.4M / 635.7M / 333.5M / 293.6M |
| `generation_config.json` | **없음** |
| `mtp_num_hidden_layers` | 선언 1, transformers 5.14.1에서 실제 모듈 생성 0건 |

## 9-2. 어댑터·통신량 (핀 버전에서 재생성 필수)

| 항목 | 값 |
|---|---|
| 부착 모듈 | 248 |
| 어댑터 파라미터 (r=16) | 32,464,896 |
| `tensor_nbytes` (fp32) | 129,859,584 B |
| `wire_nbytes` (flwr 직렬화) | 129,923,072 B (496 텐서 × 128 B 헤더 = 63,488 B, 0.049%) |
| 대조: `all-linear` | 38,993,920 (visual 98개 포함 → 비전 동결 위반) |
| 대조: `modules_to_save=[embed,lm_head]` 오발동 | 635,699,200 파라미터 = **2.54 GB** fp32 |

## 9-3. 16GB 메모리 예산 (micro=1, RTX 5060 Ti sm_120)

| 항목 | S=1300 | S=2048 |
|---|---|---|
| 가용 (`mem_get_info` idle free) | 14.72 GB | 14.72 GB |
| embed_tokens bf16 (양자화 불가) | 1.19 | 1.19 |
| lm_head bf16 (기본 보호, tied) | 1.19 | 1.19 |
| NF4 언어부 (3.57B) | 1.84 | 1.84 |
| 비전 타워 bf16 (**skip 명시**) | 0.67 | 0.67 |
| LoRA fp32 + grad + AdamW 2모멘트 | 0.52 | 0.52 |
| lm_head+CE+backward (나이브) | 4.22 | 6.64 |
| lm_head+CE+backward (**총괄 200토큰 한정**) | ~0.65 | ~0.65 |
| 선형어텐션 폴백 그래프 | ~1.3 | 1.97 |
| 단편화 | 1~2 | 1~2 |
| **합계 (총괄 한정 적용)** | **~8.5** | **~9.3** |

## 9-4. 스텝 예산

| 칸 | 계산 | 스텝 |
|---|---|---|
| 통합·중앙 | `3 × n_eff / 32` | ≈ 4,312 |
| 통합·연합 | `R=6 × Σ_k S_k` | ≈ 4,313 |
| 반올림 잔차 상한 | `R·K/2` | 9 |
| 배경지식 SFT (QA 2만 + 추론 1만, 3 epoch) | | ≈ 2,812 |

## 9-5. 커널 성능 (폴백 확정 시)

| 항목 | 값 |
|---|---|
| `full_attention` fwd+bwd (bf16, ckpt on, B=1, S=1300) | 0.0611 s/층 |
| `linear_attention` 동일 조건 | **0.1509 s/층** (2.47배) |
| 32층 (8 full + 24 linear) | 4.11 s/마이크로배치 |
| accum=32 | 132 s/옵티마이저 스텝 |
| 통합 3종 합계 (bf16 하한) | **~15 GPU-일** (NF4 dequant·비전·CE·로더 포함 시 19~21일) |

**이 수치가 G7의 판정 입력이다.** 100 step 파일럿을 기다리지 말고 지금 GPU-일 재추정에 넣는다.

---

## 마지막 한 줄

이 명세에서 **삭제하면 안 되는 것 세 가지**는 다음과 같다.

1. **G2(교환 폐포)** — 이것 없이는 "어댑터만 교환한다"는 논문 문장을 쓸 수 없다. 통과할 수밖에 없는 게이트로 그 문장을 쓰는 것이 게이트가 없는 것보다 나쁘다.
2. **G5(카나리아-1)의 `gate_key`에 프로브 스펙 포함** — 이것 없이는 통과할 때까지 프로브를 고쳐 돌린 결과가 "실측"으로 논문에 실린다.
3. **회계 규칙 ④‴(통합·중앙 총 스텝 == Σ R·S_k)** — 각 칸이 자기 계획대로 돌았는지만 검사하는 회계는 항진명제다. 두 칸이 같은 양을 돌았는지를 숫자로 증명하는 규칙이 R×E=N의 유일한 실체다.