# Weld Lab — 로컬 연구 관제실

현재 `prep_v3_imgonly_20261008T2335`의 실제 자료만 연결한 **읽기 전용** 웹앱입니다. Python 표준 라이브러리와 HTML/CSS/JS만 사용하며 별도 설치가 없습니다. Sites 스킬의 local-only 흐름을 따랐고 외부 등록·동기화·게시를 하지 않았습니다.

## 실행

프로젝트 루트에서:

```powershell
.venv/Scripts/python.exe -B -X utf8 dashboard/server.py
```

브라우저에서 <http://127.0.0.1:8765/>를 엽니다. 포트가 사용 중이면 `--port 8766`을 지정합니다. 종료는 서버 터미널에서 `Ctrl+C`입니다. 기본과 실제 바인딩 모두 `127.0.0.1`로 고정되어 있습니다. 외부 포트 공개나 터널 연결을 하지 마세요.

검증:

```powershell
.venv/Scripts/python.exe -B -X utf8 dashboard/test_dashboard.py
```

테스트는 잠깐 임의 loopback 포트에서 HTTP를 확인한 뒤 종료합니다. GPU 작업을 실행하지 않습니다.

## 표시 범위와 근거

- 작업 진행: 준비 → 생성 → 자동 판정 → 사람 검수 → 릴리스 → 학습. 자동 판정 통과와 품질 승인을 구분합니다.
- 데이터: 고정 명부 `RUN.json`, `tasks.jsonl`의 49,845 **작업행**(고유 이미지 장수 아님), train/val, C1–C3, 결함 분포, source QA 0.
- 샘플: `server_return/20261009T0452_budget_stop/`의 고정 회수 tar를 추출 없이 읽습니다. 생성 779 / 판정 89 샤드가 들어 있습니다. 시작 시 RUN/tasks/manifest/tar와 사용되는 각 샤드의 SHA-256 및 run identity를 확인합니다.
- 영상: 고정 task ID에 대응하는 train/val 입력 영상만, project `data/` 내부 경로·확장자·파일 SHA를 검증한 뒤 제공합니다. 실제 입력은 `data/interim/tiles_v1_histmatch/`의 전처리 타일입니다. 원본 raw 영상이나 모델 예측 상자가 아니며, 박스는 고정 주석입니다. eval/test 이미지, 임의 경로, 외부 URL은 제공하지 않습니다.
- 결과: 실제 저장된 샤드 처리 시간·토큰 수·메모리 및 회수된 판정 분포. 현재 run의 학습은 미실행이며 과거 다른 실험 또는 테스트 fixture의 지표를 섞지 않습니다.
- 기록: 원장의 제한된 이벤트 필드만 표시합니다. 명령, SSH 주소, 계정, 키, raw 시스템 로그는 API에서 노출하지 않습니다.

최신 서버의 샤드 수가 늘어도 샘플/처리시간/판정분포는 **검증된 10월 9일 회수 자료** 기준입니다. 새로운 회수 묶음을 자동으로 신뢰하거나 교체하지 않습니다.

## 선택적 라이브 스냅샷 계약

운영자가 읽기 전용 SSH 관측 후 `dashboard/runtime/live_status.json`을 UTF-8 JSON으로 원자 교체하면 페이지가 30초마다 반영합니다. 웹앱 자체는 SSH나 GPU 명령을 실행하지 않습니다.

```json
{
  "run_id": "prep_v3_imgonly_20261008T2335",
  "identity": "06ac84c187639a06e82e1f0542abe337431f6cc091a1dd9d229dc1f71b4b621e",
  "observed_at_utc": "<actual ISO 8601 UTC timestamp>",
  "phase": "<observed phase>",
  "supervisor_running": true,
  "generated_shards": 779,
  "generated_rows": 49845,
  "judged_shards": 89,
  "judged_rows": 5696,
  "pending_judge_rows": 44149,
  "cost_estimate_won": null,
  "cap_won": 100000,
  "hard_stop_at_utc": "<operator deadline>",
  "gpu": {"utilization_pct": null, "memory_used_mb": null},
  "events": [],
  "source": "server-readback"
}
```

위 JSON은 계약 예시이며 실제 관측 파일을 만들지 않습니다. 120초 이상 경과하면 STALE로 표시하고 현재 GPU 실행 중이라고 주장하지 않습니다. 관측 실패 시 기존 성공 스냅샷을 유지하되 `live_status.error.json`에 `{"active":true}`를 기록하면 즉시 연결 오류 상태로 표시합니다. 복구되면 `{"active":false}`로 교체합니다.

현재 미해결 품질 보류: 결함행의 치수·두께 부재 한계 문장 누락 검토, 무결함 짧은 문장 예외에 대한 Phi 과탈락 가능성. 사람 검수·권리·정식 릴리스·실험 등록·조기 좌표 점검은 이 웹앱에서 승인하거나 변경할 수 없습니다.
