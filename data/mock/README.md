# `data/mock/` — 합성 매니페스트 스냅샷

두 스냅샷은 실제 데이터셋에서 뽑은 표본이 아니다. 로더·채점기·학습 래퍼를 시험하려고
계약 스키마에 맞춰 만든 합성 자료이며, 실이미지는 포함하지 않는다.

| 스냅샷 | 매니페스트 | 주석 | 위치 라벨 |
|---|---|---|---|
| `mock_aihub_v1` | 1,000행 | 616행 | 있음(폴리곤) |
| `mock_riawelc_v1` | 300행 | 259행 | 없음(분류 전용) |

## 생성기와 입력

생성기는 `scripts/make_mock_manifest.py`다. `configs/mock_profile.yaml`의 재질별 장수·
클래스 가중치·해상도·분할 설정과 고정 시드를 사용하며 실데이터를 읽지 않는다.
결함 이름과 ISO 6520-1 코드는 `configs/label_map.yaml`에서 가져온다.

생성 시각은 상수로 고정돼 있다. 같은 생성기와 의존성 환경에서 설정·사상표·시드·판정 모드가
같으면 같은 바이트를 만든다. 이는 현재 입력으로 기존 동결본을 그대로 복원한다는 뜻은 아니다.

2026-09-21 확인에서 기본 판정 모드로 만든 `mock_aihub_v1`은 동결본과 네 파일 모두 같았다.
`mock_riawelc_v1`은 사상표의 원본 라벨 표기가 바뀌어 동결본과 다르다.
동결본의 `CR`·`PO`·`LP` 대신 `Difetto1`·`Difetto2`·`Difetto4`가 기록되며,
`manifest.csv`의 `src_labels_raw`와 `annotations.csv`의 `src_label_raw`, 그리고 이를 잠그는
`SNAPSHOT.sha256`이 달라진다. 행 수·순서·분할·해시 열·정규화 결함 유형·ISO 코드는 같다.

두 동결본과 위 확인에서 만든 기본 생성분은 불변식 IV1~IV11을 통과했다.
IV12는 실파일이 없어 검사하지 않았다. 각 스냅샷의 `data_capabilities.yaml`에는
`is_mock: true`가 기록돼 있다.

## 별도 생성 예제

기본 출력 경로는 이미 동결된 `data/mock/`다. **출력 경로 없이 생성기를 실행하지 않는다.**
아래 예제는 저장소 루트의 PowerShell에서 실행하며, 새 출력 폴더인지 먼저 확인한다.
기존 스냅샷을 재현해 덮어쓰는 명령이 아니다. `absolute` 예제도 별도 자료를 만든다.

```powershell
$mockExampleRoot = Join-Path (Get-Location) '_workspace/mock_examples'
if (Test-Path -LiteralPath $mockExampleRoot) {
    throw '이미 있는 출력 폴더입니다. 새 경로를 지정하십시오.'
}
uv run python scripts/make_mock_manifest.py --out-root "$mockExampleRoot/clause_only"
uv run python scripts/make_mock_manifest.py --profile mock_aihub_v1 --verdict-mode absolute --out-root "$mockExampleRoot/absolute"
```

## 합성 값 확인

- `sha256`는 이미지 파일의 해시가 아니라 `rel_path` 문자열의 sha256다.
- `phash_hex`는 지각 해시가 아니라 `"phash:" + group_id`의 sha256다.
- `rel_path`는 생성기가 만든 이름(`.../w#####_f#.png`)이고 그 경로에 실제 파일은 없다.
- 결함 폴리곤은 난수로 그린 볼록 다각형이다. bbox·면적·축 길이는 그 다각형에서 계산한다.

## 공개 검사와 보존

`image_id`는 계약 스키마의 `<출처>:<경로>` 형식이다. 출처 접두의 형식만 실물과 같고 값은
합성이다. 공개 검수기의 식별자 규칙에 걸릴 수 있으므로, 검사에서 제외할 때는 합성 여부를
확인한 근거와 제외 범위를 함께 기록한다.

65번의 합성 판정은 `data/mock/mock_aihub_v1/`과 `data/mock/mock_riawelc_v1/`에 보존된
두 동결 스냅샷의 구성 파일에 한정한다. 검사 제외는 공개 승인이나 검사 통과를 뜻하지 않는다.
경로 이름이 `mock`이거나 기존 제외 경로 아래에 있다는 이유로 새 자료나 변경된 자료에
이 판정을 자동 적용하지 않는다. 그런 자료는 별도로 검토해야 한다.

두 디렉터리는 동결 스냅샷이다. `SNAPSHOT.sha256`와 digest가 고정돼 있고 여러 시험이
참조하므로 재생성해 덮어쓰거나 파일을 늘리고 줄이지 않는다. 이 README는 스냅샷 디렉터리
밖에 있어 digest에 포함되지 않는다.
