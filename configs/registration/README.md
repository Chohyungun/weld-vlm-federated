# 등록 파일 — 통합형 본실험의 등록 블록

이 폴더에는 **값을 보기 전에 정한 것**을 담은 등록 파일과 그 영수증을 둔다.
파일의 꼴과 읽기는 `evaluation/prereg_unified.py` 가 정한다. 이 문서는 그 규칙을 옮겨 적지 않고 가리킨다.

## 무엇을 두나

| 목적(`generation.purpose`) | 등록 파일 | 영수증 |
|---|---|---|
| `main` — 본실험 | **여기** | **여기** — 등록이 본줄기에 병합된 다음 커밋으로 적는다 |
| `frame_diag` — 착수 전 프레임 진단 | **여기** — 진단 전에 본줄기에 먼저 커밋한다 | **여기** — 같은 규칙 |
| `rehearsal` — 배관 리허설 | 두지 않는다. 리허설 루트의 `registration/` 에 둔다 | 같음(로컬) |

## 이름

`{purpose}-{YYYYMMDD}-{n}.json` 과 그 영수증 `{purpose}-{YYYYMMDD}-{n}.receipt.json`.
`n` 은 같은 날 같은 목적의 등록 순번(1부터)이다. **한 번 커밋한 등록 파일은 고치지 않는다** — 고치면 새 등록이고 새 이름이다.

## 등록 파일 옆에 두는 것 — 본실험 · 진단

등록 파일과 **같은 커밋에** 같은 줄기 이름(`{purpose}-{YYYYMMDD}-{n}`)으로 둔다. 영수증의 `main_commit` 이 그 커밋을 가리키고, 읽는 쪽은
작업 트리가 아니라 **그 커밋에서**(`git show <main_commit>:<경로>`) 바이트를 읽어 등록의 칸과 맞댄다. 등록 파일처럼 한 번 커밋하면 고치지 않는다.

| 파일 | 이름 | 바이트 | 맞대는 등록 칸 | 목적 | 읽는 쪽 |
|---|---|---|---|---|---|
| **생성 목록** | `{purpose}-{YYYYMMDD}-{n}.generation.list` | `evaluation.eval_list.canonical_bytes` 가 낸 바이트 — `validate_eval_list` 가 받는 꼴(id 마다 한 줄 · LF · 끝 LF 하나) | `generation.eval_list_file_sha256` · `generation.eval_list_set_sha256` | `main` | 본실험 export — **목록 경로를 따로 받는 인자가 없다**(리허설 미니스펙 2판 반영판 §1-4). 본채점 진입점은 `--generation-list` 로 받은 파일이 같은 두 해시인지 본다 |
| **에코 목록** | `{purpose}-{YYYYMMDD}-{n}.echo.list` | 같은 꼴 | `generation.echo_list_file_sha256` · `generation.echo_list_set_sha256` | `main` | 본실험의 에코 export(본 export 전에 한 번) · 본채점 진입점(`--echo-list`) |
| **학습 설정 원문** | `{purpose}-{YYYYMMDD}-{n}.train_config.json` | 등록 단계가 만든 정규 JSON **문자열 그대로** — `json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` 를 UTF-8 로. **끝에 LF 를 붙이지 않는다** — 파일의 sha256 이 곧 등록의 칸이다 | `generation.train_config_sha256` | `main` · `frame_diag` | 에코 export — 에코 곁 파일의 `train_config` 는 학습 산출물이 아니라 **이 원문**을 싣는다(2판 반영판 §2-5). 모델 묶음은 어댑터 meta 의 원문을 싣고 같은 해시여야 한다 |

- 진단(`frame_diag`)의 생성 · 에코 목록은 진단 루트의 `lists/` 에 둔다(진단 흐름 — 2판 반영판 §3-1). 등록 칸의 두 해시가 그 파일을 묶는다.
- 리허설은 이 폴더에 아무것도 두지 않는다 — 리허설 루트의 `registration/` · `lists/` 다.
- 채점 쪽의 모집단 목록(`population.eval_list_file_sha256`)은 생성 목록과 같거나 그 부분집합이다. 다르면 같은 꼴로 `{…}.population.list` 를 둔다.

## 등록 파일의 꼴

- `prereg_unified.dump_registration` 이 낸 바이트 그대로다 — 키 정렬 · UTF-8 · BOM 없음 · 한 줄 + LF.
- `prereg_unified.load_registration` 은 **이 꼴만** 받는다. 읽은 객체를 다시 적은 바이트가 파일 바이트와 같아야 하고,
  모르는 키 · 빠진 키 · 다른 등록 판은 멈춘다. 손으로 고친 파일(들여쓰기 · 키 순서 · 줄끝)은 읽히지 않는다.
- 빈 칸은 JSON `null` 이다. 어떤 칸이 비어 있어도 되는지는 목적마다 다르다(`missing` · `invalid`).

## 영수증의 키

두 지문(`generation_sha256` · `scoring_sha256`) · 오프셋 있는 ISO 시각(`registered_at`) · 등록이 병합된 커밋(`main_commit`) ·
종류(`kind` — 등록의 목적과 같다) · **이 폴더 안 등록 파일의 저장소 상대 경로**(`registration_path`) · **그 파일 바이트의 sha256** · 장비(선택).

## 읽는 법

진입점은 등록을 작업 트리에서 읽지 않는다. **영수증의 `main_commit` 에서 `registration_path` 의 바이트**를 읽어
(`git show <main_commit>:<registration_path>`) 파일 해시를 영수증과 맞대고, `prereg_unified.registration_for_receipt` 로 등록 객체를 만든다.
그래야 영수증이 가리키는 등록과 지금 쓰는 등록이 같은 것이다.

## 이 폴더의 다른 설정과의 관계

`configs/base.yaml` 과 계획 파일(`rehearsal_uni.yaml` · `frame_diag_uni.yaml`)은 등록 파일을 읽지 않고, 등록 파일도 그것들을 병합하지 않는다.
등록 값이 설정 값과 같아야 하는 자리는 내보내기와 본채점이 실측값 대조로 본다.
