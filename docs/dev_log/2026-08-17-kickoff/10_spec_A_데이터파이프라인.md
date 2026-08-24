# 데이터 파이프라인 설계

작성 2026-08-17
상태 **Phase B (스펙)**. 구현 코드는 §9 환경 부트스트랩 1건을 제외하고 아직 쓰지 않았다.

읽은 것: `docs/개발규약.md`, `00_연구브리프.md`, `docs/구현설계.md` 2장, `docs/실험설계.md` 3장,
`docs/데이터셋_현황.md`, `docs/의사결정로그.md`.

---

# 0. 이 스펙의 한 줄 요약

**어느 데이터셋이 도착하든(AI허브 승인/미승인) 하류 4개 트랙의 코드가 바뀌지 않도록,
데이터의 "능력 차이"를 컬럼 값과 하나의 기계 생성 플래그 파일로 흡수한다.**

8/25 시한에 데이터가 뒤집히는 것은 리스크가 아니라 **전제**로 놓았다. 그래서 이 스펙의
설계 축은 성능이 아니라 *분기 흡수*다.

---

# 1. 계약 #1: `configs/label_map.yaml`

가장 먼저 얼리는 계약이다.

## 1-1. 설계 원칙: 계층 4단 분리

"판정 수준(불량/정상)과 결함 유형이 섞이지 않게" 한다는 요구를 다음으로 구현한다.

> **"정상"은 결함 유형이 아니다.** `defect_types`에 정상 항목을 두지 않는다.
> 정상은 이미지 수준 속성(`has_defect=false`)이고, 결함 인스턴스 행이 0개인 상태로 표현된다.

정상을 5번째 클래스로 넣는 순간 (a) 검출 프레임에서 "박스 없음"과 "정상 박스"가 이중 표현이
되고, (b) ISO 6520-1에 대응 코드가 없는 항목이 온톨로지에 섞이며, (c) Macro-F1 분모가
칸마다 달라질 여지가 생긴다.

| 계층 | 이름 | 내용 | 성격 |
|---|---|---|---|
| L1 | `verdict_levels` | `normal` / `defective` | 이미지 수준 판정. 코드 없음 |
| L2 | `defect_types` | 결함 유형 온톨로지 (ISO 6520-1 코드) | **단일 진실.** 다른 계층은 이 키를 참조만 |
| L3 | `sources.<dataset>` | 데이터셋 원본 라벨 문자열 → L2 키 | 데이터셋이 늘어도 L2는 불변 |
| L4 | `eval_spaces` | 평가 시나리오별 라벨 공간 축소 | 낯선 데이터 평가용 |

## 1-2. 스키마 확정안

```yaml
version: 1                       # 변경 시 반드시 증가. manifest 에 기록되어 대조된다.
iso_standard: "ISO 6520-1"

verdict_levels: [normal, defective]

defect_types:                    # key = 코드 전역에서 쓰는 정규 이름 (snake_case, ASCII)
  crack:
    iso_code: "100"              # 100번대 대표값
    iso_code_alt: []
    name_ko: "균열"
    name_en: "Crack"
    train_class_id: 0            # 검출 학습용 정수 ID. **L2에서만 부여**한다.
  porosity:
    iso_code: "2011"
    iso_code_alt: ["2012"]       # 원본이 구형/균일분포를 구분하지 않으므로 대표 2011
    name_ko: "기공"
    name_en: "Porosity"
    train_class_id: 1
  lack_of_fusion:
    iso_code: "401"
    name_ko: "융합불량"
    name_en: "Lack of fusion"
    train_class_id: 2
  slag_inclusion:
    iso_code: "301"
    name_ko: "슬래그혼입"
    name_en: "Slag inclusion"
    train_class_id: 3
  lack_of_penetration:
    iso_code: "402"
    name_ko: "용입불량"
    name_en: "Lack of penetration"
    train_class_id: 4            # RIAWELC 에만 존재. 주 실험(AI허브 RT)에는 나타나지 않는다.

sources:
  aihub71761:
    label_field: "TODO"          # probe 결과로 확정 (§4-1). 추측으로 채우지 않는다.
    normal_labels: ["정상"]       # 이 값이면 has_defect=false, 결함 행 0개
    mapping:
      "균열": crack
      "기공": porosity
      "융합불량": lack_of_fusion
      "슬래그혼입": slag_inclusion
    unmapped_policy: fail        # 사상표에 없는 라벨이 나오면 즉시 실패. 조용히 버리지 않는다.
  riawelc:
    label_field: "dirname"       # 폴더명이 클래스
    normal_labels: ["ND"]
    mapping:
      "CR": crack
      "PO": porosity
      "LP": lack_of_penetration
    unmapped_policy: fail

eval_spaces:
  main_rt:                       # 주 실험 (AI허브 RT)
    defect_types: [crack, porosity, lack_of_fusion, slag_inclusion]
  unseen_riawelc_common3:        # 낯선 데이터 평가. 실험설계 8-4의 "공통 3클래스"
    defect_types: [crack, porosity]     # + normal (L1). 합쳐서 3
    excluded: [lack_of_fusion, slag_inclusion, lack_of_penetration]
    reason: "AI허브 RT ∩ RIAWELC 의 교집합만 채점. 나머지는 한쪽에 존재하지 않는다."
```

## 1-3. 소비 규약 (B·C·D 가 지켜야 할 것)

1. **라벨 문자열을 코드에 쓰지 않는다.** `"기공"`, `"porosity"`, `"2011"` 어느 형태든 금지.
   전부 이 파일을 로드해서 얻는다. 위반은 재작업 사유다(`docs/개발규약.md` 불변조건 1-8).
2. **`train_class_id`는 L2에서만 부여된다.** 학습 담당이 YOLO `data.yaml`을 만들 때 자기 순서를
   새로 매기지 않는다. `eval_spaces`로 축소할 때도 원래 ID를 유지하고, 축소된 공간에서만
   쓰는 조밀 ID가 필요하면 `dense_id = eval_space 내 정렬 순서`로 별도 생성하고 매핑을 로깅한다.
3. **`unmapped_policy: fail`.** 사상 실패를 경고로 넘기면 라벨이 조용히 사라진다.

---

# 2. 계약 #2: 매니페스트 스키마

소유 A · 소비 C·D.

## 2-0. 정규화 결정: 파일 2개

한 이미지에 결함이 여러 개다. 결함 기하(bbox·장축·등가직경)는 **결함 인스턴스 단위**인데
분할(split·client)과 중복 묶음은 **이미지 단위**다. 한 파일에 넣으면 이미지 수준 컬럼이
행마다 반복되고, 반복된 값이 어긋나는 순간 "매니페스트가 단일 진실"이 깨진다.

따라서 **2파일 + 1FK**로 간다. 둘은 항상 함께 생성되고 **하나의 `SNAPSHOT.sha256`에 함께
잠긴다.** 따로 갱신하는 것을 금지한다.

| 파일 | 그레인 | 행 수(RT 70k 기준 추정) | 잠금 |
|---|---|---|---|
| `data/processed/<snapshot>/manifest.csv` | 이미지 1장 = 1행 | 62,998 | 읽기 전용 |
| `data/processed/<snapshot>/annotations.csv` | 결함 인스턴스 1개 = 1행 | 6~10만 | 읽기 전용 |

> 최초 요구사항은 bbox·장축·등가직경을 "manifest 컬럼"으로 열거했다. 이를 2파일로 나눈 것이
> 이 스펙의 유일한 계약 형태 변경이며 검토를 거쳐 확정됐다.
> 조건 2가지가 붙었고 아래에 반영했다.
>
> 1. 두 파일을 **하나의 `SNAPSHOT.sha256`** 에 함께 잠근다 → `data/manifest_io.py`의
>    `write_snapshot()` / `verify_snapshot()`. 목록이 다르면 검증기가 거부한다.
> 2. **조인은 A가 제공하는 로더 함수로만 한다** → §2-5. 로더가 계약의 일부다.

`<snapshot>` = `{source}_{yymmdd}_{sha256 앞 8자}`. 예: `aihub71761_260825_3f9a1c04`.

## 2-1. `manifest.csv` 컬럼 전수 명세

생성 주체: `S` = split 단계, `I` = ingest 어댑터, `V` = convert, `D` = dedup.

| # | 컬럼 | 타입 | 필수 | 생성 | 설명 · 예시값 |
|---|---|---|---|---|---|
| 1 | `image_id` | str | ✔ | I | **PK.** `{source}:{rel_path}` 정규화 문자열. `aihub71761:RT/ST/0012/img_004411.png` |
| 2 | `source` | enum | ✔ | I | `aihub71761` \| `riawelc`. label_map `sources` 키와 일치 |
| 3 | `rel_path` | str | ✔ | I | `data/raw/` 기준 상대경로, POSIX 슬래시. **절대경로를 저장하지 않는다**(이식성·로컬 전용 설정 파일 규칙) |
| 4 | `sha256` | str(64) | ✔ | I | 이미지 파일 바이트 해시. 원본 불변 검증용 |
| 5 | `width_px` | int | ✔ | I | 1280 |
| 6 | `height_px` | int | ✔ | I | 720 |
| 7 | `modality` | enum | ✔ | I | `RT` \| `VT`. 주 실험은 RT 만 |
| 8 | `material` | enum | ✔ | I | `ST` \| `AL` \| `UNK`. RIAWELC 은 `UNK` |
| 9 | `has_defect` | bool | ✔ | V | L1 판정 수준. false = annotations 행 0개 |
| 10 | `n_defects` | int | ✔ | V | 0 이상. `has_defect == (n_defects > 0)` 불변식 |
| 11 | `defect_types` | str | ✔ | V | L2 키를 정렬 후 `;` 조인. `crack;porosity`. 정상은 빈 문자열 |
| 12 | `iso_codes` | str | ✔ | V | 11을 코드로. `100;2011`. 정상은 빈 문자열 |
| 13 | `src_labels_raw` | str | ✔ | I | 원본 라벨 문자열 정렬 조인. **원본 보존용**, 학습에 쓰지 않는다 |
| 14 | `label_type` | enum | ✔ | I | `polygon` \| `bbox` \| `classification`. 원본 라벨의 성격 |
| 15 | `has_localization` | bool | ✔ | I | 위치 정보 유무. RIAWELC = false. **하류 null 판별의 1차 스위치** |
| 16 | `phash_hex` | str(64) | ✔ | D | pHash 16×16 → **256bit hex 64자.** §6-2 에서 hash_size=16 으로 확정했으므로 64자다(8×8·16자가 아니다) |
| 17 | `group_id` | str | ✔ | D | 중복 묶음 ID. `g000123`. 단독 이미지도 자기 자신만의 묶음을 갖는다 |
| 18 | `group_size` | int | ✔ | D | 묶음 크기. 1 = 중복 없음 |
| 19 | `strata_key` | str | ✔ | D | 층화 키. §6에서 정의. 재현성을 위해 **계산 결과를 컬럼으로 박아둔다** |
| 20 | `split` | enum | ✔ | S | `eval` \| `train` \| `val`. eval = 글로벌 평가셋 20%(선분리) |
| 21 | `client` | enum? | . | S | `C1` \| `C2` \| `C3`. `split==eval` 이면 **null** |
| 22 | `eval_subset` | enum? | . | S | `judgment_2000` \| null. 판정문·조항 채점 표본(구현설계 4-5) 표시 |
| 23 | `thickness_mm` | float? | . | I | 원본 메타데이터에 있을 때만. 없으면 null |
| 24 | `thickness_source` | enum | ✔ | I | `metadata` \| `assumed` \| `none` |
| 25 | `px_per_mm` | float? | . | I | 픽셀-mm 스케일. 없으면 null |
| 26 | `scale_source` | enum | ✔ | I | `metadata` \| `assumed` \| `none` |
| 27 | `quality_level` | enum? | . | I | ISO 5817 품질수준 `B`\|`C`\|`D`. 원본에 없으면 null → `assumed` 경로 |
| 28 | `ingest_version` | str | ✔ | I | 어댑터 버전. `aihub-v1.0` |
| 29 | `label_map_version` | int | ✔ | V | 사상표 버전. 불일치 시 하류가 즉시 중단 |
| 30 | `notes` | str? | . | * | 자유 서술. 파이프라인 로직이 이 값을 읽지 않는다 |

## 2-2. `annotations.csv` 컬럼 전수 명세

| # | 컬럼 | 타입 | 필수 | 생성 | 설명 |
|---|---|---|---|---|---|
| 1 | `ann_id` | str | ✔ | I | **PK.** `{image_id}#{seq}`, seq는 원본 JSON 등장 순서 0-base |
| 2 | `image_id` | str | ✔ | I | **FK → manifest.image_id** |
| 3 | `src_label_raw` | str | ✔ | I | 원본 라벨 문자열 |
| 4 | `defect_type` | str | ✔ | V | L2 키 |
| 5 | `iso_code` | str | ✔ | V | L2 코드 |
| 6 | `polygon_json` | str | . | I | **원본 폴리곤 좌표 그대로.** `[[x,y],...]` JSON. 덮어쓰지 않는다 |
| 7 | `bbox_x1_px` | int | . | V | 원본 이미지 픽셀 좌표계 (좌상 기준) |
| 8 | `bbox_y1_px` | int | . | V | |
| 9 | `bbox_x2_px` | int | . | V | |
| 10 | `bbox_y2_px` | int | . | V | |
| 11 | `area_px` | float | . | V | 폴리곤 면적 (shapely) |
| 12 | `major_axis_px` | float | . | V | `cv2.minAreaRect` 긴 변 |
| 13 | `minor_axis_px` | float | . | V | 짧은 변 (허용치 표가 폭을 요구할 때 대비) |
| 14 | `equiv_diameter_px` | float | . | V | √(4·area/π) |
| 15 | `major_axis_mm` | float? | . | V | `major_axis_px / px_per_mm`. 스케일 없으면 null |
| 16 | `equiv_diameter_mm` | float? | . | V | 동일 |
| 17 | `geom_valid` | bool | ✔ | I | 좌표 이상 통과 여부 |
| 18 | `geom_flags` | str | ✔ | I | `` \| `negative_coord;out_of_bounds;self_intersect;too_few_points;zero_area` |

**파생 규칙.**
- 원본은 `polygon_json` 하나에만 있고 **어떤 단계도 이 컬럼을 수정하지 않는다.**
- 픽셀 파생은 `_px` 접미, mm 환산은 `_mm` 접미로 **컬럼을 새로 쌓는다.** 제자리 갱신 금지.
- mm 컬럼은 `px_per_mm`이 있을 때만 채워진다. 없으면 null이고, 가정값으로 채우지 않는다
  (가정값은 §4-2의 `verdict_mode: conditional` 경로에서 채점 시점에 적용한다. 데이터에 굽지 않는다).

## 2-3. null 의미 규약: 하류(C·D)가 결측을 구분하는 방법

이 계약의 핵심 요구다. **결측에는 세 종류가 있고 절대 같이 취급하면 안 된다.**

| 종류 | 뜻 | 판별 | C·D 가 해야 할 일 |
|---|---|---|---|
| **N1 구조적 부재** | 데이터셋이 위치 라벨을 주지 않음 (RIAWELC) | `has_localization == false` | 검출 학습은 분류 경로로 강등. **mAP·BBox-IoU를 산출하지 않는다.** 결과표에 `N/A(no_localization)` 문자열을 넣는다 |
| **N2 정상 이미지** | 결함이 실제로 없음 | `has_defect == false` (annotations 행 0개) | 검출: "박스 없음" 정답. 채점: 정답 집합이 공집합. **N1과 달리 지표에 정상적으로 기여한다** |
| **N3 파생 불가** | 스케일·두께가 없어 mm 환산 불가 | `px_per_mm is null` / `_mm` 컬럼 null | 합부를 절대 기준으로 산출하지 않는다. §4-2 `verdict_mode` 를 따른다 |

**금지 사항 (검수 대상)**
- 결측을 `0`으로 채우고 평균에 넣지 않는다. N1을 0으로 채우면 mAP가 인위적으로 낮아져
  다섯 칸 비교가 통째로 무의미해진다.
- `pandas.read_csv` 기본 동작이 빈 문자열을 `NaN`으로 바꾸므로, `defect_types`·`iso_codes`처럼
  **빈 문자열이 유의미한 값(정상)** 인 컬럼은 반드시 `keep_default_na=False`로 읽는다.
  → 이 로더를 A가 `data/manifest_io.py`로 제공하고 C·D는 직접 `read_csv` 하지 않는다.
- N1/N2/N3을 `if pd.isna(x)` 하나로 뭉뚱그리지 않는다. 위 표의 판별 컬럼을 쓴다.

## 2-4. 불변식 (매니페스트 검증기가 매번 확인)

```
IV1  image_id 유일, annotations.image_id ⊆ manifest.image_id
IV2  has_defect == (n_defects > 0) == (해당 image_id 의 annotations 행 수 > 0)
IV3  defect_types 는 label_map L2 키의 정렬·중복제거 조인과 정확히 일치
IV4  has_localization == false  ⇒  해당 이미지의 모든 bbox_*·*_px·polygon_json 이 null
IV5  px_per_mm is null          ⇒  해당 이미지의 모든 *_mm 이 null
IV6  split == 'eval'            ⇔  client is null
IV7  같은 group_id 의 모든 행은 split 이 동일하다        ← 누수 방지의 핵심
IV8  같은 group_id 의 모든 행은 client 가 동일하다
IV9  eval_subset 이 non-null 인 행은 split == 'eval'
IV10 bbox 좌표는 0 ≤ x1 < x2 ≤ width_px, 0 ≤ y1 < y2 ≤ height_px
IV11 label_map_version 이 configs/label_map.yaml 의 version 과 일치
IV12 manifest 의 sha256 컬럼이 실제 파일 해시와 일치 (표본 1% 무작위 재계산)
```

IV7·IV8은 **누수 여부를 직접 판정하는 불변식**이다. 실패하면 실험 결과 전체가 무효다.

구현: `data/invariants.py::check_invariants()`. 첫 위반에서 멈추지 않고 전부 모아 반환한다.
IV12는 `raw_root` 를 넘길 때만 동작한다(실파일이 필요하다).

## 2-5. 로더 계약

**계약 #2는 컬럼 표가 아니라 "컬럼 표 + 이 로더"다.** C·D가 각자 `pd.read_csv` 와 `merge` 를
쓰면 조인 규칙과 결측 해석이 트랙마다 갈라진다. 아래가 승인된 유일한 접근 경로다.

| API | 역할 | 하류가 직접 하면 안 되는 것 |
|---|---|---|
| `load_snapshot(root, verify=True)` | 세 파일 + 해시 검증을 한 번에 | `read_csv` 직접 호출. 빈 문자열이 NaN 이 되어 "정상"과 "결측"이 섞인다 |
| `join_defects(snapshot)` | manifest × annotations LEFT JOIN, **정상 이미지도 1행 유지**, `row_kind` 부여 | 자체 `merge`. 정상 이미지를 떨어뜨리거나 N1/N2를 혼동한다 |
| `write_snapshot(root, m, a, caps)` | 바이트 정규 CSV + **하나의 SNAPSHOT.sha256** | 파일 개별 저장·개별 잠금 |
| `verify_snapshot(root)` | 잠금 확인. 목록이 계약과 다르면 거부 | 해시 검증 생략 |
| `Snapshot.can_score(metric)` | 이 스냅샷으로 그 지표를 낼 수 있는가 | 전역 플래그를 직접 파싱해 조건문 흩뿌리기 |
| `defect_free_images` / `localizable` / `split_view` | 결측·분할 판별의 단일 창구 | `split == "eval"` 류 문자열 비교를 코드 곳곳에 |

**`MetricStatus`** enum 을 함께 제공한다. 산출 불가 지표는 `0` 이 아니라
`MetricStatus.NO_LOCALIZATION`("N/A(no_localization)")을 결과표에 넣는다.

로더 자체의 계약성은 `tests/test_manifest_io.py`가 지킨다. 컬럼 순서 변경 거부, 빈 문자열
보존, 부분 잠금 거부, 개조 탐지, 바이트 왕복 동일성.

---

# 3. ingest 어댑터: 공통 인터페이스

"8/25에 어느 쪽으로 확정되든 코드가 안 바뀌는 것"이 목표이고, 이 절이 그 답이다.

## 3-1. 인터페이스

```python
# data/ingest/base.py
class IngestAdapter(Protocol):
    source: str            # label_map.sources 키와 동일
    version: str           # manifest.ingest_version 에 기록

    def discover(self, raw_root: Path) -> Iterator[RawItem]:
        """data/raw/<source>/ 를 순회하며 (이미지 경로, 원본 라벨 레코드) 를 낸다.
        원본을 읽기만 한다. 쓰기·이동·삭제 금지."""

    def parse(self, item: RawItem) -> ImageRecord:
        """pydantic 검증을 통과한 이미지 1장 + 결함 인스턴스 리스트를 낸다.
        검증 실패는 예외가 아니라 ImageRecord.reject_reason 으로 표시하고 계수한다."""

    def capabilities(self, records: Sequence[ImageRecord]) -> Capabilities:
        """전수 스캔 결과로 이 데이터셋이 무엇을 할 수 있는지 보고한다. §4-2."""
```

`ImageRecord`(pydantic)가 §2-1·§2-2 컬럼과 1:1 대응한다. **어댑터는 manifest 컬럼을 직접
쓰지 않고 `ImageRecord`만 만든다.** 컬럼 직렬화는 공통 writer 한 곳에서만 일어난다 
어댑터가 늘어도 스키마가 갈라지지 않게 하는 장치다.

## 3-2. 두 어댑터의 차이 = 값의 차이일 뿐

| 항목 | `AiHub71761Adapter` | `RiawelcAdapter` |
|---|---|---|
| 원본 | 이미지 + 이미지당 폴리곤 JSON | 클래스별 폴더 안의 227×227 PNG |
| `label_type` | `polygon` | `classification` |
| `has_localization` | `true` | `false` |
| annotations 행 | 결함 개수만큼 (기하 컬럼 채움) | 결함이 있으면 **1행**, 기하 컬럼 전부 null |
| `modality` | 원본 메타 (`RT`/`VT`) | `RT` 고정 |
| `material` | 원본 메타 (`ST`/`AL`) | `UNK` (원본에 재질 정보 없음) |
| `thickness_mm` / `px_per_mm` | probe 결과에 따름 (§4-1) | `none` 고정 |
| `phash` | 원본 이미지 | 227×227 패치(문서 기재 224는 오기). **§6의 임계 재확정 필요** |
| 묶음 키 | probe 결과에 따름 (필름 ID 등) | **파일명 접두사 = 모원본.** 24,407장은 독립 표본이 아니라 **479개 모원본**의 격자 타일이다. 어댑터가 `group_key` 로 채우고 dedup 이 E2 엣지로 받는다 |
| 저자 제공 분할 | 없음 | `training/validation/testing` 이 원본에 있으나 **분할 정보로 쓰지 않는다** (아래) |

RIAWELC에서도 결함 이미지에 annotations 행을 1행 만드는 이유: `defect_type`·`iso_code`는
존재하므로 Class-Jaccard·Macro-F1 경로가 그대로 성립한다. 기하만 null이다.
즉 **N1(위치 없음)과 N2(결함 없음)가 데이터 구조에서 확실히 갈린다.**

### 저자 제공 분할을 쓰지 않는다: RIAWELC 은 쓰면 누수율 100%다

RIAWELC(`training/validation/testing`)과 LoHi-WELD(kfold)에는 저자가 나눠 둔 분할이 들어
있다. **분할 정보로 해석하지 않는다.**

> **실측(전수 sha256).** RIAWELC `testing` 2,443장은 **전량 `training` 의 바이트
> 동일 복제본**이다. 대조 2,443/2,443 일치. 저자 분할을 그대로 쓰면 **테스트셋 전체가
> 학습셋 안에 있다. 누수율 100%다.** 원칙 문제가 아니라 이 데이터셋에서 실제로 그렇다.
>
> 재확인(2026-08-23, 표본 40쌍): `training` 의 같은 이름 파일이 40/40 존재하고
> sha256 40/40 일치, pHash 도 40/40 일치했다. 이 연구는 글로벌 평가셋을 회사별 분할보다 **먼저**
직접 선분리하고(불변조건 1-3), 그 순서가 다섯 칸이 같은 기준으로 채점되는 근거다. 저자 분할을
섞으면 칸마다 다른 기준이 된다. 어댑터에서 이 디렉터리는 **경로의 일부**일 뿐이고,
`split` 컬럼은 §6-6~§6-8 의 분할 단계만 채운다. 검증: `test_author_split_dirs_never_become_split_values`.

### 타일 데이터셋의 묶음: 파일명이 pHash보다 정확하다

RIAWELC 24,407장은 **479개 모원본**을 격자로 자른 타일이다
(`bam5_Img2_A80_S5_[3][10].png` → 모원본 `bam5_Img2_A80_S5`, 타일 `[3][10]`).
같은 모원본의 타일이 학습과 평가로 갈리면 누수다(불변조건 1-5).

**pHash로는 못 잡는다.** 비겹침 타일은 서로 다른 화면이라 지각 해시가 가깝지 않다
(§6-10의 원리적 한계). 반면 파일명 접두사는 모원본을 **정확히** 지시한다. 그래서 이 데이터셋의
실질 누수 방어선은 pHash가 아니라 E2 엣지이며, pHash는 보조로만 쓴다.

> **접두사 정규식 주의.** ` - Copia` 접미가 붙은 파일이 30건 있다. 이 접미를 흡수하지 않는
> 정규식을 쓰면 접두사가 **479개가 아니라 509개**로 세어지고, 그 30건이 각각 단독 묶음이 되어
> 원래 모원본과 갈린다. 막으려던 누수가 그대로 발생한다. 타일 블록 뒤를 `.*`로 흡수해야 한다.
> (전수 실측 2026-08-21: 엄격·느슨 정규식 모두 479, 미매칭 0)

**논문 서술 사항.** 24,407은 **파일 수**이고 고유 이미지는 **21,964**다. 게다가 그 21,964도
독립 표본이 아니라 **모원본 479개**의 격자 타일이다. 낯선 데이터 평가에서 표본 수를 쓸 때
세 숫자를 구분해 밝힌다. 파일 24,407 / 고유 21,964 / 모원본 479. 유효 표본 수가 파일 수와
다르다는 사실을 감추면 일반화 주장이 과장된다.

### 바이트 동일 복제본은 ingest 에서 걷어낸다

묶음(E1 엣지)이 이미 **누수**는 막는다. 같은 바이트는 반드시 한 묶음이라 학습·평가로
갈리지 않는다. 실측으로 확인했다(2026-08-23): 복제 쌍 25쌍을 pHash 임계 0으로 투입해도
**25/25가 한 묶음**이 됐고 E1·E2·E3 세 엣지가 모두 발동했다.

그런데 **묶음은 개수를 고치지 않는다.** 복제본을 남기면 ① 표본 수가 부풀어 논문 규모
서술이 틀리고 ② 복제된 이미지만 학습에서 2회 노출되어 클래스 분포가 기울며
③ `R × E = N` 등가 계산의 N 이 실제 고유 표본과 어긋난다.

→ **누수 방어(묶음)와 별개로 ingest 에서 한 번 걷어낸다.**
`data/ingest/base.py::drop_exact_duplicates()` 가 sha256 기준으로 하나만 남기고,
남기는 쪽은 `rel_path` 사전순 첫 번째다(입력 순서 비의존). 제외 건수를 러너가 보고한다.
묶음의 E1 엣지는 그대로 두어 **이중 방어**로 남긴다.

### 저자 분할 파생본은 매니페스트에 넣지 않는다

LoHi-WELD 는 원본과 파생본이 한 배포본에 섞여 있다. 실측 디렉터리 집계(2026-08-23):

| 구역 | 파일 수 | 성격 |
|---|---|---|
| `high_resolution_welds` | 1,022 | **원본** |
| `low_resolution_welds` | 2,000 | **원본** |
| `kfold_high/*`, `kfold_low/*` | 12,690 | 위 3,022장을 5-fold 로 **재배치한 파생본** |
| 계 | 15,712 | |

`kfold_*` 를 그대로 넣으면 같은 이미지가 4~5회 중복 등재되고 저자 fold 가 섞여 들어온다.
**원본 구역만 ingest 하고 파생본 구역은 어댑터가 경로 규칙으로 배제한다.**
일반 규칙: **한 배포본 안에 원본과 재배치본이 함께 있으면 원본만 넣는다.** 파생본을 넣지
않는다는 사실과 배제한 파일 수를 등록부 `layout.exclude_globs` 와 처리 기록에 남긴다.

## 3-3. 백업 경로 전환 시 실제로 바뀌는 것

| 대상 | 바뀌는가 | 비고 |
|---|---|---|
| manifest·annotations 스키마 | **아니오** | 컬럼 값만 다름 |
| 분할 절차(§6) | **아니오** | `material`이 전부 `UNK`이라 층화 키에서 재질 축이 자동 소거 |
| C3 정의 | **예** | 자연 재질 분할이 불가 → Dirichlet 3-way (실험설계 3-6) |
| 검출 학습 | **예** | 분류 과제로 강등. `has_localization`이 스위치 |
| 평가 지표 | **예** | mAP·BBox-IoU 제외 (비교 항목 5→4). D가 `N/A` 처리 |

C3 정의 변경은 코드가 아니라 **`configs/split.yaml`의 클라이언트 정의 블록**으로 흡수한다.
분할 함수는 "클라이언트 정의 리스트"를 인자로 받는 순수함수이므로 시그니처가 안 바뀐다.

---

# 4. 두께·픽셀 스케일 부재 분기 (**최대 리스크**)

## 4-1. 프로그램 전수 확인: 설명서를 믿지 않는다

`data/ingest/probe_metadata.py`

```
1) 라벨 JSON 전 건을 순회하며 **키 경로 히스토그램**을 만든다.
   {"meta.thickness": 68231, "meta.pixelSpacing": 0, ...}   ← 표본이 아니라 전수
2) 후보 키를 정규식으로 넓게 긁는다: thick|두께|plate|pixel|spacing|mm|scale|resolution|dpi
   설명서에 없는 이름으로 들어있을 수 있으므로 이름을 가정하지 않는다.
3) 값의 타입·단위·범위 sanity:
   - thickness 후보: 수치인가, 1~100 범위인가, 단위 문자열이 섞여 있는가("12mm" vs 12)
   - scale 후보: 0 초과인가, 이미지 해상도와 곱해 물리 크기가 합리적인가
4) 결측 패턴: 전체 결측 / 일부 결측(비율) / 특정 재질·폴더에만 존재. 셋을 구분해 보고
5) 산출: data/interim/<source>/metadata_probe.json  (원본은 건드리지 않는다)
```

**부분 보유가 가장 위험하다.** 일부만 실측 두께로 절대 판정하고 나머지를 조건부로 하면
다섯 칸이 서로 다른 채점 기준을 갖게 된다. 그래서 판정은 전부-아니면-전무로 간다(§4-2 임계).

## 4-2. 런타임 플래그: `configs/data_capabilities.yaml`

**사람이 손으로 쓰지 않는다.** ingest 단계가 probe 결과로 **기계 생성**하고, 스냅샷 해시에
포함시켜 잠근다. C·D는 이 파일 하나만 읽으면 자기가 무엇을 산출할 수 있는지 안다.

```yaml
# 자동 생성: 손으로 편집 금지. 생성: data/ingest/emit_capabilities.py
generated_at: "2026-08-25T10:00:00+09:00"
snapshot_id: "aihub71761_260825_3f9a1c04"
source: aihub71761

counts:
  images_total: 70000
  with_thickness: 0
  with_pixel_scale: 0
  with_quality_level: 0

capabilities:
  localization: true          # bbox·mAP·BBox-IoU 산출 가능한가
  thickness_mm: false         # 실측 두께가 있는가
  pixel_scale: false          # 픽셀→mm 환산이 되는가
  size_mm: false              # = thickness_mm AND pixel_scale
  verdict_mode: clause_only   # absolute | conditional | clause_only

assumptions:                  # verdict_mode 가 conditional 일 때만 유효
  thickness_mm: null
  px_per_mm: null
  quality_level: null
  rationale: null
```

**`verdict_mode` 3단계와 판정 결정 규칙**

| 값 | 조건 | 판정 과업 | 채점 |
|---|---|---|---|
| `absolute` | `pixel_scale` 및 `thickness_mm` 보유율 ≥ 0.95 | 실측 mm 로 합부 | 판정 정합성 정상 산출 |
| `conditional` | 미보유이나 **가정값을 명시적으로 채택** | 가정 두께·가정 스케일 하 조건부 합부 | 가정값 기준으로 산출. 한계 절에 명시 |
| `clause_only` | 미보유 + 가정값 미채택 | **조항 검색 + 기준 서술까지.** 합부 미산출 | 판정 정합성 대신 `N/A`, 인용 일치율만 |

- 보유율 임계 **0.95**. 그 미만은 반올림해서 true 로 올리지 않는다(부분 보유 위험).
- **안전 기본값은 `clause_only`.** probe 가 돌기 전, 또는 판단이 안 서면 `clause_only`다.
  가장 좁은 과업이고 여기서 나온 결과는 다른 모드에서도 유효하다.
- `absolute` ↔ `conditional` ↔ `clause_only` 전환은 **확정 전 재검토 사항**으로 둔다. 이 값 하나가
  판정 정합성 지표의 정의를 바꾸기 때문이다.

**노출 위치 요약 (C·D 가 읽는 곳)**

| 층위 | 위치 | 용도 |
|---|---|---|
| 전역 | `configs/data_capabilities.yaml` → `capabilities.*` | "이 지표를 산출할 수 있는가" |
| 행 수준 | `manifest.has_localization`, `px_per_mm`, `thickness_source`, `scale_source` | "이 행에 값이 있는가" |

두 층위가 어긋나면(예: 전역 `localization: true`인데 특정 행이 `false`) **검증기가 실패시킨다.**
전역 플래그는 행 수준의 집계여야 한다.

---

# 5. 폴리곤 → bbox·크기 산출

`data/convert/geometry.py`. 전부 순수함수.

| 산출 | 방법 | 라이브러리 | 컬럼 |
|---|---|---|---|
| bbox | 폴리곤 x·y의 min/max, `floor(min)`/`ceil(max)` 후 이미지 경계로 clip | numpy | `bbox_*_px` |
| 면적 | 폴리곤 면적 | shapely `Polygon.area` | `area_px` |
| 장축 | 최소 외접 사각형의 긴 변 | `cv2.minAreaRect` → `max(w,h)` | `major_axis_px` |
| 단축 | 같은 사각형의 짧은 변 | `min(w,h)` | `minor_axis_px` |
| 등가직경 | √(4·area/π) | numpy | `equiv_diameter_px` |
| mm 환산 | `px / px_per_mm` | . | `*_mm`, 스케일 없으면 null |

**규칙**
- 입력 폴리곤은 절대 수정하지 않는다. `polygon_json`에 원본 그대로 저장.
- 좌표 이상 처리: 음수·이미지 밖은 **버리지 않고** clip 한 뒤 `geom_flags`에 사유를 남긴다.
  단 clip 후 `area_px == 0` 이거나 점이 3개 미만이면 `geom_valid=false`로 표시하고
  **학습·채점에서 제외**한다. 제외 건수를 로그와 스펙 보고에 남긴다(구현설계 2-1).
- 자기교차 폴리곤은 `shapely.make_valid` 로 보정한 뒤 면적을 계산하고 `self_intersect` 플래그.
  보정 결과가 MultiPolygon 이면 **최대 면적 조각만** 채택하고 플래그에 남긴다.
- bbox 는 정수 픽셀. 학습용 좌표 변환(YOLO 정규화 / VLM 네이티브 좌표계)은 **여기서 하지 않는다.**
  학습 단계가 `to_train_coords()` / `to_orig_coords()` 한 쌍으로 처리한다(`docs/개발규약.md` 모델정책 8).

---

# 6. 중복 묶음 + 분할 (함정 구간 #8)

> 이 절은 다관점 병렬 검증(설계 3관점 → 관점별 적대 검증 → 종합)으로 도출했다.
> 실행 기록: 워크플로 `track-a-phash-split-design`.

설계 3안(누수 관점 / 층화 통계 관점 / 운영·재현성 관점)을 각각 독립 생성한 뒤 **관점별 적대
검토**를 붙였고, 적대 검토에서 무너진 항목을 폐기·수정해 하나로 종합했다. 세 관점이 공통으로
채택했던 두 가지(64-bit pHash, 비둘기집 다중 인덱스)가 **적대 검증에서 전부 깨졌다**. 단독
진행이었으면 그대로 갔을 항목이다. 이것이 이 구간을 함정으로 지정한 값이다.

아래 파라미터의 정본은 `configs/base.yaml`이며, 여기 적힌 값은 그 초기 기입값이다.

## 6-0. 쟁점별 채택 결정 요약

| # | 쟁점 | 채택 | 기각 | 근거 |
|---|---|---|---|---|
| 1 | 해시 크기 | **256-bit (hash_size=16)** 처음부터 | 64-bit 시작 + 실패 시 승격 | 승격 분기는 인덱스 재설계 없이는 위음성 상당수 미탐. 가장 밟힐 확률이 높은 분기에서 누수 방어가 꺼진다 → 분기 자체를 제거. 균일 배경 RT에서 64-bit 판별력 부족은 3안 공통 인정 |
| 2 | 후보쌍 생성 | **비트팩 전수 popcount(브루트포스)** | 비둘기집 다중 인덱스(MIH) 3안 공통 채택이었음 | 치명 결함 4건이 전부 인덱스 회수 보장에서 발생. 2.45×10⁹쌍은 numpy 비트팩으로 수 분~수십 분. 순환 논증·가짜 골짜기 문제가 근원에서 소멸 |
| 3 | 전처리 | 비트깊이 정규화 → **CLAHE 적용** | 그레이스케일 단독 | 16-bit 함정을 명시적 정규화로 방어. CLAHE는 노출 드리프트 프레임을 병합 방향으로, 서로 다른 용접부를 분리 방향으로 민다 |
| 4 | 임계 결정 규칙 | **관측 미병합 0 우선** (t\* 상향 반복) | 쌍-정밀도 ≥0.95인 최대 t | 손실 비대칭(6-1). 정밀도 규칙은 t 상향을 허용해 조용한 누수를 만든다 |
| 5 | 거대 성분 처리 | **성분 절단 금지** + NCC AND 엣지 재검증(G2)만 | complete-linkage 재분할 | 성분 절단은 임계 이내 쌍을 분할 경계에 남기는 것이 그래프 이론적으로 확정 |
| 6 | 교차 재질 근접쌍 | 전역 계산 + **해소 필수 게이트** | 재질 파티션 분리 / 로그만 | 파티션 분리는 탐지 자체를 막고, 로그만 남기면 탐지하고도 통과한다 |
| 7 | 층화 알고리즘 | **SGKF + 묶음 대표 라벨을 구성원 전 이미지에 복제** | 커스텀 다중라벨 그리디 | 그리디의 유일한 이점(오차 제어)이 granularity 한계에서 무력화 |
| 8 | 최희소 클래스 기준 | **재질 내 빈도, manifest 런타임 유도, 동률은 ISO 코드 오름차순** | 전역 빈도 / 공지 수치 하드코딩 | 전역 빈도는 AL에서 순서가 정반대. AL 융합불량 329 vs 균열 332 는 살얼음이라 실측 유도가 필수 |
| 9 | 층화 허용 밴드 | **granularity 인지형** ±max(2%p, 층내 최대 묶음/층 크기), 위반 시 **기록·보고** | 고정 ±2%p + 하드 실패 | 소수 층의 ±2%p 는 묶음 원자 크기보다 좁아 정상 데이터에서도 원리적으로 실패한다. 하드 정지는 8/25 시한을 태운다 |
| 10 | Dirichlet 파라미터화 | **농도 벡터 (1/3, 1/6) 를 그대로 명기** | "α=0.5" 축약 표기만 남기기 | 축약 표기는 재현 불가. `α·p`인지 `α·n·p`인지에 따라 이질성 강도가 달라진다 |
| 11 | 극단 실현 방어 | **실현 총량비 수용 밴드 + 결정론적 재추첨 수열** | 무검사 / 예산 캡 / 클래스 0장 재추첨 | 농도 <1 은 쌍봉이라 클래스당 상당 비율이 극단 배정된다. 캡·0장 재추첨은 분포를 추가 절단해 α 주장에서 더 멀어진다 |
| 12 | val 분할 | **신설: 묶음 단위·시드 고정** | (3안 모두 미명세였음) | manifest 스키마가 요구하는데 스펙 공백이었다. 임의 분할은 val 곡선 오염 + 비결정 |
| 13 | 재현 표면 | numpy·sklearn·Pillow·imagehash·opencv·pandas·Python **전부 lock** + 캐시 키에 버전 포함 + manifest 바이트 규격 | Pillow·imagehash만 고정 | numpy Generator 스트림 비보장(NEP 19), Pillow 리샘플링 변동, win32 CRLF |

## 6-1. 설계 공리: 손실 비대칭

- **과분리(under-merge)는 누수다.** 평가셋 중복률만큼 지표가 상향 오염되고, 오염이 **칸별로
  비대칭**이다(중앙집중이 학습 풀 전체를 봐 이득 최대 → 회복률 분모 과대 → 회복률 하향,
  RQ3 편익은 상향). 비교 주장 자체가 무효가 된다.
- **과병합(over-merge)은 누수가 아니다.** 묶음이 통째로 한쪽에 가므로 지표 편향이 없고,
  층화 오차·배정 granularity 저하만 남는다.
- 따라서 **모든 이지선다에서 병합 쪽을 택한다.** 부작용은 성분 크기 가드(6-5)와 granularity
  인지형 허용 밴드(6-6)가 통제한다.
- **연결 성분은 절단하지 않는다.** 유일한 예외는 G2 엣지 재검증(6-5). 절단이 아니라 엣지별
  판정 기준의 강화다.

## 6-2. 전처리·해시

| 항목 | 확정값 | configs 키 |
|---|---|---|
| 비트깊이 전수 스캔 | ingest 에서 전 이미지 mode·비트깊이·min/max 를 프로그램으로 확인(설명서 불신). 리포트 커밋 | . |
| 비트깊이 정규화 | 16-bit 존재 시 이미지별 percentile clip [0.5%, 99.5%] → uint8 선형 스케일. 8-bit 면 항등. 규칙 ID를 캐시 키에 포함 | `dedup.preprocess.bit_depth_clip: [0.5, 99.5]` |
| 그레이스케일 | 정규화된 uint8 에서 `L` 변환 | . |
| 대비 증폭 | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` | `dedup.preprocess.clahe` |
| 기하 변형 | **금지** (레터박스·크롭·EXIF transpose 없음). dedup 전처리는 학습 전처리와 완전 분리 | . |
| 해시 | `imagehash.phash(hash_size=16, highfreq_factor=4)` = 256-bit | `dedup.phash` |
| 캐시 | `data/interim/dedup/phash.parquet`, 키 = **image sha256(경로 아님)**. `phash_config_hash` = sha256(hash_size, highfreq_factor, 전처리 규칙 ID, Pillow·imagehash·opencv·numpy 버전 문자열). 불일치 시 무효화. 중단-재개와 환경 재구축이 겹쳐도 혼합 커널 오염이 불가능 | . |
| 버전 고정 | `uv.lock` 에 고정 (실측: numpy 2.2.6 / opencv 5.0.0 / imagehash 4.3.2 / Pillow 12.3.0 / sklearn 1.9.0 / pandas 2.3.3). `np.bitwise_count` 때문에 **numpy ≥ 2.0 이 하드 요구사항**이라 pyproject 하한을 올려뒀다. 고정 픽스처 이미지의 해시 기록값 대조 테스트로 드리프트 감지 | . |

## 6-3. 쌍 거리 계산: 전수 popcount (MIH 폐기)

- 해시를 4×uint64 로 비트팩, 이미지를 **sha256 오름차순 정렬** 후 타일(1024×4096) 단위
  XOR + `np.bitwise_count` 로 전수 계산. 2.45×10⁹쌍, 예산 수 분~수십 분.
- 근사·인덱스가 없으므로 **회수율 100%가 자명**하고, 임계를 어느 값으로 바꿔도 재인덱싱이 없다.
- 산출: ① 거리 구간 **정확 전수 히스토그램**(ST / AL / 교차 3벌, 로그 스케일),
  ② 거리 빈별 표본쌍 최대 2,000개(정렬-타일 스캔 순서 선착순. 결정론, RNG 불요),
  ③ 먼 거리는 카운트만.
- `d ≤ threshold_cap(96)` 쌍 총수가 `dedup.pair_budget(5×10⁷)` 초과 시 **조기 퍼콜레이션
  경보**로 기록(6-5 사다리로).
- 묶음 생성은 t\* 확정 후 2차 전수 패스에서 `d ≤ t*` 쌍을 union-find 로 스트리밍한다
  (쌍을 저장하지 않으며, 분할은 순서 불변).

## 6-4. 임계 확정 절차 (표본 100쌍 눈 확인)

출발점 **t0 = 32/256** (관례 8/64 와 동일 비율 12.5%), 상한 **threshold_cap = 96**
(37.5%).

> **상한은 운용점이 아니다.** 96 은 4단계의 상향 반복이 막히지 않도록 둔 천장이고,
> 실제로 쓰는 값은 눈 확인으로 정하는 t\* 다. 초기값 48 은 픽스처 실측에서 **너무 낮았다** 
> 과분리(누수)가 0 이 되는 지점이 t=74 였고 48 에서는 같은 용접부 쌍 30건이 갈려 있었다.
> 48 까지 과병합이 0 이라 상한을 올리는 대가도 없었다. 실측 기록은 `21_dedup_실측_A.md`.
>
> **절대 t\* 는 실데이터 도착 후 이 절차로 다시 도출해야 한다.** 픽스처의 프레임 간 차이는
> 합성한 것이라 실제 연속 촬영과 거리 분포가 다를 수 있다. 74 도 96 도 t\* 가 아니다.

1. **골짜기 탐색** 6-3의 정확 히스토그램(전수·무편향, 인덱스 산물이 아님)에서 ST/AL 별도로
   same-weld 모드와 different-weld 모드 사이 골짜기를 확인. 이중모드 미형성은 그 자체를
   과병합 경보로 기록.
2. **표본 100쌍** (`rng = default_rng(seed_split+1)`, 빈별 저장 표본에서 추출)
. 병합 경계 [t0−6, t0] 40쌍 / 미병합 경계 (t0, t0+12] 40쌍 / 양성 대조 d≤8 5쌍 /
   음성 대조 d≥56 5쌍 / 최대 성분 내부 10쌍. contact-sheet PNG + 판정 시트 CSV 생성.
3. **판정**. 박상은 1차, 조현건 독립 2차, 불일치는 합의. 시트를 커밋(통과율 근거).
4. **결정 규칙 (미병합 0 우선)**. 미병합 경계 표본에서 same-weld 쌍이 나오면 t\* 를 그 최대
   거리로 **상향**하고, 새 경계에서 40쌍을 (이미 계산된 정확 거리에서) 재추출해 반복.
   종료 조건: 경계 표본 40쌍에 same-weld 0, 또는 t\* = 96 도달. t\* 는 단조 증가·상한 유계라
   종료가 보장되고, 판정이 같으면 결과도 같다(결정론). t\*=96 에서도 same-weld 가 나오면
   pHash 부적합 → 총괄 에스컬레이션.
5. 확정한 (t\*, 절차 결과, 판정 시트 경로)를 `dedup.threshold_final` 과 의사결정로그에 기록.
6. **정직 조항 (스펙에 명문화)**. t\* 확정은 "관측 범위 내 무누수"이지 무누수의 증명이 아니다.
   표본 100쌍은 낮은 실제율의 잔존 과분리를 놓칠 수 있고, **pHash 는 평행이동 불변이 아니라
   이동 촬영 부분 겹침을 원리적으로 잡지 못한다.** 실질 방어선은 6-5의 E2 메타데이터 엣지이며,
   부재 시 이 한계를 논문 한계 절에 쓴다.

## 6-5. 묶음 생성 (union-find + 가드)

**엣지 3종** (전부 union, 우선순위 없음)

| 엣지 | 정의 | 가드 취급 |
|---|---|---|
| E1 | 파일 sha256 완전 일치 | 확실. 재검증 제외 |
| E2 | AI허브 JSON 의 필름/용접부/촬영연번 ID 동일 (채택 게이트 통과 시) | 크기 가드 **면제**, 별도 회계 |
| E3 | 정확 해밍 거리 ≤ t\* | G1 / G2 대상 |

- **E2 채택 게이트** (승인 직후). ingest 가 관련 필드를 전수 스캔 → 존재 시 ID별 이미지 수
  분포 리포트 + **표본 20 ID 눈확인**(ID 내 이미지가 정말 같은 용접부인가). 카세트·배치 단위
  ID 로 밝혀지면 기각. 통과 시 채택. **이동 촬영 누수의 실질 방어선이다.**
- **교차 재질**. 재질이 다른 이미지는 어떤 엣지로도 union 하지 않는다. 그런데 교차 재질에서
  E1 일치나 E3 임계 이내 쌍이 나오면 **해소 필수 게이트**에 적재하고 manifest 잠금 전 반드시
  해소한다. 로그만 남기고 통과 금지.
- **union-find**. 경로 압축 + rank. 처리 순서는 sha256 오름차순 고정. 최종 분할이 엣지 순서
  불변임을 테스트로 증명한다.
- **`group_id` = `"grp_" + min(구성원 image_sha256)[:16]`**. 내용에서만 유도되어 순회·삽입·
  병렬 순서와 무관하다. 단독 이미지도 같은 규칙을 적용하므로 **`group_id` 는 전 행 non-null**
  이고 하류 C·D 에 null 분기가 필요 없다. ID↔구성원 단사성을 테스트로 검증한다.
- **가드 G1 (경보, 정지 아님)**. E3 유래 성분에 한해 최대 성분 > max(200장, 0.3%·N) 이거나
  1위 > 3×2위 이면 경보. 크기 ≥20 성분은 내부 최대 거리(지름)를 리포트. E2 유래 성분은
  면제·별도 보고(심 단위 정당 묶음은 클 수 있다).
- **가드 G2 (경보 성분 한정, 결정론)**. 성분 내부 **E3 엣지만** `pHash ≤ t* AND NCC ≥ 0.85`
  (CLAHE 후 128×128)로 재선별한 뒤 재-union. E1·E2 엣지는 보존. 절단이 아니라 판정 기준의
  강화다. 수작업 엣지 편집 금지, 발동 이력 로그.
- **해소 사다리** (정지 상태로 시한을 태우지 않는다). G1 경보 → G2 재검증 + 성분 내부 표본
  눈확인 → ① "진짜 같은 용접부" → 묶음 유지, granularity 밴드가 흡수 ② "서로 다른 용접부
  혼입"(퍼콜레이션 확정) → 총괄 에스컬레이션, **안전 기본값: 성분 원자성 유지 + 전량 학습 풀
  배정(평가셋 금지), ST→C1 / AL→C3 고정**, 건수·사유를 논문에 보고. 평가셋 무결성(비가역)을
  지키면서 파이프라인은 진행된다.
- 묶음 내 재질 단일성 assert. 산출물 `data/interim/dedup/groups.parquet` + `stage_meta.json`.

## 6-6. 층화 키와 글로벌 평가셋 20% 선분리

- **층화 라벨 단일화(2단 축약)**. 이미지 라벨 집합 = 결함 클래스 집합(없으면 {정상}).
  묶음 대표 클래스 = 구성원 라벨 합집합 중 **재질 내 이미지 빈도 최희소** 클래스.
  빈도 순서는 공지 수치가 아니라 **manifest 실측에서 런타임 유도**하고, 동률은 ISO 코드
  오름차순, 유도된 순서를 configs 에 스냅샷한다.
  (참고 예상 순서: ST 슬래그<균열<융합불량<정상<기공 / AL 융합불량<균열<슬래그<기공<정상.
  AL 의 융합불량 329 대 균열 332 는 3장 차이라 하드코딩하면 실측에서 뒤집힌다.)
- **단일 정의**. SGKF 의 `y` = **묶음 대표 클래스를 구성원 전 이미지에 복제**한
  (재질 × 대표클래스) 10층 라벨. 허용 검사도 같은 정의를 참조한다. 다른 정의의 y 사용 금지.
  → 이 값이 §2-1의 `strata_key` 컬럼에 그대로 기록된다.
- **실행**. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed_split+2)`,
  `groups=group_id`, **fold 0 = 글로벌 평가셋**. 평가셋 행은 `split=eval`, `client=null`.
- **검사 2단 + 의미론**

| 검사 | 기준 | 위반 시 |
|---|---|---|
| 묶음 straddle | 어떤 `group_id` 도 eval 과 나머지에 걸치지 않음 | **하드 실패** (버그다) |
| 층별(축약 라벨) 평가 비율 | 20% ± max(2%p, 층내 최대 묶음 크기 / 층 이미지 수) | 기록 후 진행 |
| 클래스별 이미지 수준 비율 (축약 전 **진짜** 클래스) | 동일 공식의 밴드 | 기록 후 진행 |
| 소수 층 최소량 | 층별 평가셋 ≥ 50장 | 리포트 지표 (판단 재료) |

밴드가 granularity 에 스케일링되므로 정당한 대형 묶음(연속 촬영 30프레임)이 하드 정지를
유발하지 않는다. 이것은 "검증"이 아니라 **"검사 + 보고"** 임을 문구로 명시한다.

## 6-7. 학습 풀 클라이언트 분할

- **C3 = AL 학습 풀 전량** (자연 분할, 함수 미경유, 무작위성 없음).
- **C1:C2 = ST 묶음 Dirichlet 배정.** 순수함수:

```python
def dirichlet_partition(
    group_ids: tuple[str, ...],          # 사전순 정렬 (호출 규약). ST 학습 풀 묶음만
    group_labels: np.ndarray,            # (G,) 묶음 대표 클래스 6-6 과 동일 정의
    group_sizes: np.ndarray,             # (G,) 묶음 이미지 수
    concentration: tuple[float, float],  # (1/3, 1/6) = α·p,  α=0.5, p=(2/3, 1/3)
    seed: int,
) -> np.ndarray:                         # (G,) ∈ {0(C1), 1(C2)}
    ...
```

  I/O·전역 상태 없음, `rng = np.random.default_rng(seed)` 만 사용.
  알고리즘: 클래스 k 를 **재질 내 빈도 오름차순**으로 순회 → `p_k ~ rng.dirichlet(concentration)`
  → 클래스 k 묶음을 `group_id` 정렬 후 rng 셔플 → 누적 **이미지 수**(묶음 수가 아니다)가
  `p_k[0] × (클래스 k 총 이미지)` 에 도달할 때까지 C1 에 순차 충전, 나머지 C2.
  예산 캡·빈패킹·클래스 0장 재추첨은 **전부 기각**(실현 분포를 명목 α 에서 추가 왜곡시킨다).
- **수용 밴드 + 결정론 재추첨**. 실현 C1 지분 = C1/(C1+C2) 이미지 수 ∈ **[0.60, 0.73]**
  (목표 2/3). 불통과 시 `seed_d, seed_d+1, …` 수열로 재추첨해 **최초 통과를 채택**하고,
  채택 시드·시도 횟수를 manifest 메타와 논문에 기록. 농도 벡터·수용 밴드·절단 규칙을
  configs 와 논문에 그대로 명기한다. **"α=0.5" 축약 표기 단독 사용 금지.**
- 클라이언트 특정 클래스 0장은 **허용**(의도된 non-IID). 실현 분포는 **원본 클래스 기준**
  (축약 라벨이 아니다) 재질×클래스×클라이언트 히트맵으로 보고하고, RQ3 해석 시 클라이언트별
  분해를 필수로 한다.
- **평가셋 선분리가 먼저임을 함수 합성으로 강제**. 클라이언트 분할 함수는 평가셋을 제외한
  묶음 목록만 입력받는다. 코드 구조상 순서 역전이 불가능하다.

## 6-8. 클라이언트 내 train/val 분할

- 각 클라이언트 내부에서 6-6 과 같은 기계: `StratifiedGroupKFold(n_splits=10, random_state=seed_split+4)`,
  `y` = 복제 대표 라벨, `groups=group_id`, **fold 0 = val (10%)**.
- val 은 곡선 로깅 전용(조기 종료 금지 정책)이므로 허용 밴드는 리포트 전용. **묶음 straddle 만
  하드 실패.**

## 6-9. 순서·잠금·재현성

- **순서 고정** ① 묶음 생성 → ② 평가셋 20% 선분리 → ③ C3 자연 분할 + C1:C2 Dirichlet →
  ③′ 클라이언트 내 val → ④ manifest 확정·sha256·읽기전용 전환·커밋 → ⑤ 히트맵(원본 클래스 기준).
- **시드 표**. 근원 `split.seed = 20260825` (configs/base.yaml 단일 출처).
  `+1` 눈확인 표본 / `+2` SGKF eval / `+3` Dirichlet 수열 시작 / `+4` 클라이언트 내 val.
  전역 `np.random` 상태 사용 금지.
- **manifest 바이트 규격 (win32 함정)**. UTF-8(BOM 없음), 개행 `\n` 고정, 컬럼 순서 고정,
  행 정렬 `sha256` 오름차순, 소수 2자리 고정 포맷, 결측은 빈 문자열. **동일 config 재실행 시
  파일 sha256 이 바이트 단위로 동일**함을 테스트로 강제한다.
- **멱등·재개**. 스테이지별 `stage_meta.json`(입력 해시, config 해시. 라이브러리 버전 포함)
  일치 시 스킵. pHash 는 sha256 키 이미지 단위 캐시라 중단-재개가 안전하다.
- 모든 목표량·빈도 순서·검사 기준은 **manifest 실측에서 런타임 유도**한다. 공지 수치
  하드코딩 금지. 승인 후 실측이 달라도 코드가 안 바뀌고, 계획 재작성 여부만 다시 판단한다.

## 6-10. RIAWELC 백업 경로

- 어댑터가 동일 스키마로 수렴하므로 dedup·split 코드 **무수정 완주**가 목표다. 재질 축이
  상수가 되어 층화가 클래스 단독으로 자동 퇴화한다(코드 변경 없음).
- **t\*·전처리를 이식하지 않는다.** 227×227 패치는 중복 구조가 달라 6-4 절차를 전체 재수행.
- pHash 만으로는 부족하다. 같은 원판(필름)에서 잘린 **비겹침 패치는 pHash 가 원리적으로 못
  잡는다.** → **조사 완료(2026-08-21). 파일명 접두사가 모원본을 정확히 지시한다**(479개).
  E2 엣지로 투입하며 이것이 이 데이터셋의 실질 방어선이다. §3-2 참조.

## 6-11. 단위 테스트 목록 (dedup·split)

| # | 테스트 | 검증 내용 |
|---|---|---|
| 1 | `test_phash_fixture_stable` | 고정 픽스처 해시 = 기록값 (라이브러리 버전 드리프트 감지) |
| 2 | `test_bitdepth_normalize` | 16-bit percentile clip 정규화 결과 고정 |
| 3 | `test_pairwise_tile_vs_reference` | 타일 popcount 거리 = 참조 전수 구현 (n=500) 완전 일치 |
| 4 | `test_pair_sampling_deterministic` | 빈별 표본쌍·눈확인 추출이 실행 간 동일 |
| 5 | `test_unionfind_order_invariance` | 엣지 순서를 셔플해도 분할 동일 |
| 6 | `test_group_id_content_derived` | 순열 불변 + 단독 이미지 규칙 + ID↔구성원 단사 |
| 7 | `test_cross_material_gate` | 교차 재질 근접쌍이 병합되지 않고 해소 게이트에 적재된다 |
| 8 | `test_material_purity` | 혼합 재질 묶음이 즉시 오류 |
| 9 | `test_component_guard_alarm` | 합성 체인에서 G1 발동 (E3 유래만, E2 면제) |
| 10 | `test_ncc_reverify` | G2 가 E3 만 재선별·E1/E2 보존·결정론 |
| 11 | `test_stratum_key_rarest_per_material` | 재질 내 최희소·ISO 코드 타이브레이크·런타임 유도(수량 교란 픽스처) |
| 12 | `test_stratum_label_replicated` | SGKF `y` 가 묶음 대표 라벨의 구성원 복제와 일치 (단일 정의) |
| 13 | `test_no_group_straddles_split` | eval/C1/C2/C3/val 어디에도 묶음이 걸치지 않는다 |
| 14 | `test_eval_before_client` | 클라이언트 분할 입력에 평가셋 묶음이 없다 (합성으로 강제) |
| 15 | `test_tolerance_semantics` | 밴드 위반은 리포트 산출, straddle 만 하드 실패 |
| 16 | `test_dirichlet_pure_deterministic` | 동일 입력·시드 2회 동일, 시드 변경 시 상이, 부수효과 없음 |
| 17 | `test_dirichlet_disjoint_complete_atomic` | 서로소·전체 커버·묶음 원자성·이미지 수 기준 회계 |
| 18 | `test_dirichlet_prior_convergence` | 농도 ×10⁶ 스케일에서 C1:C2 → 2:1 수렴 |
| 19 | `test_dirichlet_skew` | 확정 농도의 클라이언트 간 클래스 분포 L1 이 고농도 대비 유의하게 크다 |
| 20 | `test_dirichlet_redraw_deterministic` | 밴드 불통과 시 seed+1 수열 최초 통과 채택·시드 기록 |
| 21 | `test_manifest_byte_reproduce` | 규격(LF·UTF-8·정렬·포맷) 하 2회 실행 sha256 동일 |
| 22 | `test_resume_after_interrupt` | 중단-재개 결과 = 무중단 결과 (캐시 무효화 포함) |
| 23 | `test_phash_cache_invalidation` | 라이브러리 버전 문자열 변경 시 캐시 무효 |
| 24 | `test_riawelc_adapter_convergence` | 동일 스키마 수렴·bbox 공란·파이프라인 무수정 완주 |

## 6-12. 이 절에서 나온 열린 질문

§8 표에 Q10~Q15 로 합류시켰다.

---

# 7. 단위 테스트 목록

`tests/` 아래. 통과 조건은 전부 green 이다.

## 7-1. label_map (`tests/test_label_map.py`)

| 테스트 | 검증 대상 |
|---|---|
| `test_l2_keys_unique_and_ascii` | L2 키가 유일하고 snake_case ASCII |
| `test_train_class_id_dense_and_unique` | `train_class_id`가 0..N-1 조밀·유일 |
| `test_normal_not_in_defect_types` | 정상이 L2에 없다 (계층 분리 회귀 방지) |
| `test_source_mapping_targets_exist` | L3 사상의 우변이 전부 L2 키 |
| `test_eval_space_subset_of_l2` | L4 라벨 공간이 L2 부분집합 |
| `test_no_hardcoded_label_strings` | 저장소 전체 grep. 코드에 한글 라벨/ISO 코드 리터럴 없음 |
| `test_iso_code_alt_no_collision` | 대표 코드와 별칭이 다른 유형과 충돌하지 않음 |

## 7-2. geometry (`tests/test_geometry.py`)

| 테스트 | 검증 대상 |
|---|---|
| `test_bbox_from_axis_aligned_rect` | 축정렬 사각형 폴리곤 → bbox 정확 일치 |
| `test_bbox_clip_to_image_bounds` | 음수·초과 좌표가 clip 되고 플래그가 남는다 |
| `test_major_axis_rotated_rect` | 45° 회전 사각형의 장축이 대각이 아니라 긴 변 |
| `test_equiv_diameter_circle` | 반지름 r 원 근사 폴리곤 → 등가직경 ≈ 2r (허용오차 1%) |
| `test_selfintersect_bowtie` | 나비넥타이 폴리곤 → make_valid, 최대 조각 채택, 플래그 |
| `test_degenerate_rejected` | 점 2개·면적 0 → `geom_valid=false`, 예외 아님 |
| `test_mm_null_when_no_scale` | `px_per_mm=None` → 모든 `_mm` null (0 아님) |
| `test_polygon_json_not_mutated` | 입력 폴리곤 객체가 변형되지 않음 |

## 7-3. ingest / capabilities (`tests/test_ingest.py`)

| 테스트 | 검증 대상 |
|---|---|
| `test_both_adapters_emit_same_columns` | 두 어댑터 산출 DataFrame 의 컬럼 집합·dtype 동일 |
| `test_riawelc_localization_false` | RIAWELC 행 전부 `has_localization=false` + 기하 null |
| `test_unmapped_label_fails_loudly` | 사상표에 없는 라벨 → 예외. 조용히 스킵하지 않음 |
| `test_capabilities_matches_row_level` | 전역 플래그가 행 수준 집계와 일치 |
| `test_verdict_mode_default_clause_only` | probe 정보 없음 → `clause_only` |
| `test_verdict_mode_threshold_095` | 보유율 0.94 → false, 0.96 → true |
| `test_raw_dir_untouched` | ingest 전후 `data/raw/` 파일 해시·mtime 불변 |

## 7-4. manifest 검증기 (`tests/test_manifest_invariants.py`)

IV1~IV12 각각에 대해 **통과 케이스 1 + 위반 케이스 1** = 24 테스트.
`test_IV7_group_split_leak_detected` 와 `test_IV8_group_client_leak_detected` 는
고의로 누수를 만든 픽스처로 검증기가 실제로 잡는지 확인한다.

## 7-5. dedup / split

§6에서 확정한 목록을 따른다.

## 7-6. 로더 (`tests/test_manifest_io.py`)

| 테스트 | 검증 대상 |
|---|---|
| `test_empty_string_not_nan` | 정상 이미지의 `defect_types` 가 `""` 로 읽힌다 (NaN 아님) |
| `test_dtypes_stable_roundtrip` | write → read 왕복 후 dtype·값 동일 |
| `test_readonly_after_freeze` | 잠금 후 쓰기 시도가 실패 |

---

# 8. 열린 질문

> **2026-08-17 검토에서 15건 전부 처리됐다.** Q1·Q8 은 조건부 승인(아래 표 비고),
> 나머지 13건은 기본값 승인. 표는 기록으로 남긴다.

| # | 질문 | **안전 기본값 (답 없으면 이대로 간다)** | 누가 답하나 | 언제 |
|---|---|---|---|---|
| **Q1** | 계약 #2를 `manifest.csv` 단일 파일이 아니라 `manifest.csv` + `annotations.csv` 2파일로 나눠도 되는가 | **승인(조건 2).** 하나의 SNAPSHOT.sha256 에 함께 잠금 + 조인은 §2-5 로더로만 → 둘 다 구현·테스트 완료 | | ✅ 닫힘 |
| **Q2** | `verdict_mode`가 `conditional`일 때 가정 두께·스케일·품질수준 값을 무엇으로 할 것인가 | **`clause_only` 로 시작한다.** 가정값을 정하는 것은 B(허용치 CSV) 확정 이후에만 의미가 있다 | | 8/25 이후 |
| **Q3** | 학습 풀 안의 `val` 비율. 조기 종료를 안 쓰므로 val은 로깅 전용이다 | **클라이언트별 학습 풀의 10%, 묶음 단위·층화.** 학습에 개입하지 않는다(불변조건 3-1) | | 확정 시 |
| **Q4** | `eval_subset=judgment_2000` 표본을 언제 뽑는가 | **분할 확정과 동시에**(§6 ④ 직전) 뽑아 manifest 에 박는다. 나중에 뽑으면 채점 표본이 실행마다 달라진다 | | 확정 시 |
| **Q5** | RIAWELC의 `LP`(용입불량 402)는 AI허브 RT에 없다. 주 실험 라벨 공간에 미리 넣어둘 것인가 | **L2에는 정의하되 `eval_spaces.main_rt`에서 제외.** 온톨로지는 데이터셋과 독립이어야 한다 | | 확정 |
| **Q6** | VT 74,019장을 ingest 대상에 포함할 것인가 | **manifest 에는 넣되 `modality=VT`로 표시하고 주 실험 분할에서 제외.** 확장·낯선 데이터용으로 남긴다. 지금 버리면 나중에 재수집 비용 | | 8/25 |
| **Q7** | 스냅샷 잠금을 OS 읽기 전용 속성으로 걸 것인가, 해시 검증으로만 할 것인가 | **해시 검증 + git 추적.** Windows 읽기 전용 속성은 worktree/정션 환경에서 도구마다 다르게 동작해 오히려 사고를 낸다. 잠금 = "쓰기 시 검증기가 실패" | | 확정 시 |
| **Q8** | 하류 작업이 manifest 확정을 기다려야 하는가 | **아니다. 완료.** `data/mock/mock_aihub_v1`(1,000행) + `mock_riawelc_v1`(300행) 커밋. §11 참조 | | ✅ 닫힘 |
| **Q9** | `.venv` 를 worktree 5개가 각각 가질 것인가 | **각자 갖는다.** `uv sync` 로 lock 에서 재현되고 디스크 여유(226GB)가 충분하다. 공유는 트랙 간 의존성 오염 위험 | | 확정 |
| **Q10** | AI허브 JSON 에 필름/용접부 ID 필드가 존재하는가, granularity 는 심(seam) 단위인가 | 20 ID 눈확인 통과 시 **E2 채택.** 부재·기각 시 pHash 단독으로 가고 "이동 촬영 부분 겹침 누수 잔존 가능"을 논문 한계 절에 명시 | | 데이터 확보 직후 |
| **Q11** | 퍼콜레이션 확정 성분(서로 다른 용접부 혼입, G2 로도 미해소) 처리 | **성분 원자성 유지 + 전량 학습 풀 배정**(평가셋 금지), ST→C1 / AL→C3 고정, 건수·사유 논문 보고 | | 발생 시 |
| **Q12** | 혼합 재질 묶음·교차 재질 근접쌍의 해소 방법 | **관련 이미지 전부 격리(실험 제외) + 건수 보고.** 재질 정정은 원본 불변 원칙에 따라 manifest 의 별도 override 컬럼으로만 | | 잠금 전 필수 |
| **Q13** | 클라이언트 특정 클래스 0장을 허용할 것인가 | **허용**(의도된 non-IID). 재추첨 없음, 히트맵·클라이언트별 분해 보고 필수 | | 확정 시 |
| **Q14** | CLAHE 를 유지할 것인가 | **켠다.** 히스토그램 이중모드 형성 실패 시에만 CLAHE 끈 버전 1회 비교 후 총괄 에스컬레이션 | | 6-4 실행 시 |
| **Q15** | C1 지분 수용 밴드 [0.60, 0.73] | **기본값 그대로.** 변경 시 configs 와 논문을 동시 갱신 | | 확정 시 |

---

# 9. 환경 부트스트랩 (이 단계에서 유일하게 구현한 것)

## 9-1. 결과: **성공.** sm_120 커널 실측 통과.

`uv run python scripts/check_env.py` 실측 출력 (2026-08-17, `E:\Fedvlm_for_welding_wt_A`):

```
python            : 3.11.13 (AMD64)
torch             : 2.11.0+cu128
torch.version.cuda: 12.8
cuda available    : True
device            : NVIDIA GeForce RTX 5060 Ti
compute capability: 12.0 (sm_120)
total memory      : 15.9 GiB
arch list         : ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
fp32 matmul       : ok, max abs err = 1.984e-04
bf16 autocast     : ok, dtype = torch.bfloat16
peak allocated    : 80 MiB
PASS
```

확인된 것: (a) `arch_list`에 `sm_120`이 실제로 들어 있다. PTX JIT 대체가 아니라 네이티브
커널이다. (b) fp32 matmul 과 bf16 autocast 둘 다 실행됐다. **설명서가 아니라 커널 실행으로
확인했다.**

## 9-2. 산출물

| 파일 | 내용 |
|---|---|
| `pyproject.toml` | Python `==3.11.*` 고정. `[tool.uv.sources]`로 torch·torchvision 을 `pytorch-cu128` 인덱스에 **명시적으로 못박음**(`explicit = true`) |
| `uv.lock` | 전체 잠금. 정본 |
| `requirements.txt` | pip 전용 환경 대비 동결본. 헤더에 `--extra-index-url .../cu128` 을 넣어둠 (uv export 재실행 시 지워지므로 정본은 lock) |
| `scripts/check_env.py` | 재현 가능한 환경 검증. 새 환경·GPU 전환 시 제일 먼저 돌린다 |
| `configs/gpu/16gb.yaml`, `configs/gpu/48gb.yaml` | 프로파일 2벌 **뼈대만.** 두 파일의 키 집합이 동일해야 한다는 제약을 주석으로 명시. 값은 파일럿 후 |

## 9-3. 함정 구간 #5 에 대한 보고

시행착오를 예상했으나 **첫 시도에 통과**했다. 이유는 torch 2.11 시점에 cu128 휠이
sm_120 을 기본 arch list 에 포함하기 때문이다(2025년 초 5000 시리즈 출시 직후의
"cu126 휠에서 sm_120 미포함" 문제는 해소된 상태다). 병렬 시도가 필요 없었으므로
다관점 병렬 검증 예산은 §6(pHash·분할)에 몰아 썼다.

**다만 남은 위험이 있다.** 아래는 아직 검증하지 않았고 학습 착수 전에 먼저 확인해야 한다.

| 패키지 | 위험 | 확인 방법 |
|---|---|---|
| `bitsandbytes` | 4bit 양자화 커널의 sm_120 빌드 포함 여부. QLoRA 전체가 여기 걸린다 | 실제 4bit 로드 + 1 step backward |
| `flash-attn` | Blackwell 사전 빌드 휠이 없을 수 있음 | 없으면 `sdpa` 로 대체 (성능만 손해, 결과 동일) |
| `vllm` | cu128 / sm_120 지원 버전 확인 필요 | 합성 생성 착수 전 |
| `ultralytics` | torch 2.11 API 호환 | 소형 학습 1 epoch 스모크 |

이들은 `[project.optional-dependencies]`의 `vlm`/`fl`/`corpus`/`rag` 그룹으로 분리해
두었으므로, **하나가 깨져도 기본 환경과 데이터 파이프라인 작업은 영향받지 않는다.**

---

# 10. 구현 현황

| # | 작업 | 상태 | 산출물 |
|---|---|---|---|
| 1 | `configs/label_map.yaml` 작성 + 로더 | ✅ | `configs/label_map.yaml`, `data/label_map.py` |
| 2 | 계약 #2 로더·조인·잠금 (조건 1·2) | ✅ | `data/manifest_io.py` |
| 3 | 불변식 검증기 IV1~IV12 | ✅ | `data/invariants.py` |
| 4 | 폴리곤 → bbox·크기 | ✅ | `data/convert/geometry.py` |
| 5 | Dirichlet 순수함수 + 수용 밴드 재추첨 | ✅ | `data/split/dirichlet.py` |
| 6 | **mock 스냅샷 2벌 → C·D 언블록** | ✅ | `data/mock/`, `scripts/make_mock_manifest.py` |
| 7 | 테스트 | ✅ 82 passed | `tests/` 5벌 |
| 8 | `probe_metadata.py` 전수 확인 | ⬜ | AI허브 도착 후 (`label_field` 도 이때 채움) |
| 9 | ingest 어댑터 2벌 (RIAWELC 먼저) | ⬜ | 착수 가능 |
| 10 | dedup 전수 popcount + 임계 확정(§6-4) | ⬜ | 실이미지 필요 |
| 11 | 실데이터 분할 → 잠금 → 분포 히트맵 | ⬜ | 8/25 이후 |

---

# 11. mock 스냅샷: C·D 인계 사항

`uv run python scripts/make_mock_manifest.py` 로 재생성된다. **재실행 시 바이트가 동일**하다
(digest 대조 확인). 실이미지는 없다. `sha256` 은 `rel_path` 에서 만든 값이고
`notes` 컬럼과 `data_capabilities.yaml`의 `is_mock: true` 로 표시된다.

| 스냅샷 | 행 | 결함 인스턴스 | localization | verdict_mode | 용도 |
|---|---|---|---|---|---|
| `data/mock/mock_aihub_v1` | 1,000 | 616 | **true** | `clause_only` | 주 실험 경로. bbox·mAP 코드 개발 |
| `data/mock/mock_riawelc_v1` | 300 | 259 | **false** | `clause_only` | **N1 결측 경로.** mAP 를 `N/A` 로 처리하는 분기 개발 |

`mock_aihub_v1` 실측: split `train 720 / eval 200 / val 80`, client `C1 446 / C2 233 / C3 121`,
Dirichlet 농도 (1/3, 1/6) · 채택 시드 20260828 · 시도 1회 · 실현 C1 지분 0.657
(수용 밴드 [0.60, 0.73] 통과).

**두께·스케일이 있는 경로**를 개발해야 하면 다음으로 변형 스냅샷을 만든다(커밋하지 않는다):

```
uv run python scripts/make_mock_manifest.py --profile mock_aihub_v1 \
    --verdict-mode absolute --out-root <임시경로>
```

**C·D 가 지켜야 할 것**

1. `pd.read_csv` 를 직접 부르지 말고 `load_snapshot()` 을 쓴다. 정상 이미지의
   `defect_types` 가 `""` 인데 직접 읽으면 `NaN` 이 되어 N2 와 N3 가 섞인다.
2. 조인은 `join_defects()` 만 쓴다 (조건 2).
3. 산출 불가 지표는 `0` 이 아니라 `MetricStatus` 값을 넣는다. `mock_riawelc_v1` 로
   자기 코드가 그렇게 동작하는지 확인할 수 있다.
4. mock 의 절대 수치로 성능을 판단하지 않는다. 스키마·분기 검증 전용이다.
