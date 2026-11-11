# 코드 안내

이 저장소에는 현재 포스터의 Qwen 본실험과 그 이전의 사전실험 코드가 함께 있다. 결과를 이해하려면 [연구 소개](../README.md)를, 구현을 따라가려면 아래 표를 먼저 읽는다.

## 본실험 경로

| 역할 | 주요 파일 |
|---|---|
| 학습량과 실행 조건 검증 | `vlm/budget.py`, `vlm/registration.py` |
| VLM 미세조정 | `vlm/pilot_vlm.py` |
| 참여자 학습·연합 집계 | `fl/client_vlm.py`, `fl/uni_fed.py` |
| 생성 전 학습·평가 범위 대조 | `vlm/export_preflight.py` |
| 모델 출력 생성 | `scripts/export_uni.py`, `vlm/export_model.py` |
| 정식 평가 입력 검증 | `evaluation/strict_main.py` |
| 검출 성능과 표본 불확실성 | `evaluation/sample_bootstrap.py` |
| 설명문·조항 평가 | `evaluation/strict_narrative.py`, `evaluation/clause_axis.py` |
| 메타데이터 기반 baseline | `evaluation/box_gate_baseline.py` |

`detection/`과 `rag/`는 분리형 검출기·검색 기반 판정부의 사전실험 코드다. 그 결과를 현재 Qwen 본실험의 같은 조건 대조군으로 합치지 않는다.

## 환경과 검증

Python 3.11을 사용한다. 의존성은 `pyproject.toml`과 `uv.lock`에 기록되어 있다. Torch 설치 설정은 CUDA 12.8 저장소를 사용하므로 환경에 맞는 장치·드라이버를 먼저 확인해야 한다.

아래 검증은 합성 입력으로 학습 비율, 등록 설정과 생성 전 검사 규칙이 정해진 대로 동작하는지 확인한다. 모델 학습이나 실제 영상 생성 실행 명령이 아니다.

```powershell
python -m pytest -q -m "not private_assets" tests/test_export_main_eval.py tests/test_budget_fraction.py tests/test_main_registration.py
```

위 세 파일에는 비공개 토큰표·학습 자료·평가 벡터가 필요한 통합 검증 10개도 있다. `private_assets` 표시는 그 검증을 공개본 검증 명령에서만 제외한다. 자료를 갖춘 연구 환경에서는 표시된 검증도 실행해야 하며 원래의 검증 조건과 실패 판정은 유지한다. 학습 비율이 `1.0`이면 학습 행 선택 시드가 바뀌는지 보는 시험 하나는 적용 대상이 아니다.

그 밖의 전체 시험도 비공개 자산·개발 이력 또는 실행 자원을 요구할 수 있다. 위 명령의 통과를 전체 연구 재현이나 GPU 실행 검증으로 해석하지 않는다.

## 실행 설정을 고를 때

- 작업용 `configs/train_registration.yaml`은 학습 비율 `1.0`의 후속 준비 설정이다.
- README에 보고한 결과는 `main-20261102-1`의 약 1/4 학습 조건이다. 현재 작업용 설정과 혼용하지 않는다.
- 재현에는 실행 당시 설정 사본, 동결 분할 목록, 초기·최종 어댑터, 생성 응답 및 채점 산출이 필요하다. 공개 설정의 파일 해시는 동일성 확인용이며 해당 비공개 파일의 배포를 뜻하지 않는다.
- 학습·생성·채점 실행기는 필요한 자산과 조건을 먼저 검사한다. 누락된 자료를 임의로 대체해 기존 실행 이름을 재사용하지 않는다.

## 과거 실행을 추적할 때

주석의 과거 명세 번호, 코드에 저장된 등록 경로·검증 지문은 당시 실행을 추적하는 정보다. 공개 문서와 주석의 정리는 과거 실행의 등록값을 갱신하는 절차가 아니다. 원래 실행을 재현할 때는 당시 코드와 등록·입력 자산을 함께 확보해야 한다. 공개본에 없는 운영 기록을 요구하는 검증은 임의로 우회하지 않는다.

대시보드는 2026년 10월 코퍼스 준비 기록을 읽는 도구이며 현재 포스터 결과를 표시하지 않는다. [평가 코드 안내](../evaluation/README.md)에서도 현재 Qwen 경로와 이전 YOLO 절차를 구분한다.

상세 조건은 [README 재현 범위](../README.md#재현-범위), 자료 이용 범위는 [관리 원칙](개발규약.md)을 따른다.
