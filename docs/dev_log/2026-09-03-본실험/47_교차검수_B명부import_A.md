# 47. 트랙 B 명부 import 전환(598cd47) 교차 검수 — 트랙 A (2026-09-16)

지시: 총괄 pane(09-16 저녁). 대상 `git diff 5e38f2e...598cd47 -- corpus tests`
(`corpus/generate/frozen_out.py` +62/−… · `corpus/validate/verify_backup.py` +27 · `tests/corpus/test_frozen_out.py` +91 ·
`test_release_and_backup.py` +5 · 33번 §12). `57736e3`(45번)은 문서라 사실 확인만(§6). `wt_B` 는 git 객체로만 읽었다 —
체크아웃·정션·삭제 없음. 실행은 `git archive 598cd47` 스크래치 사본(정션·git 없음)에서 wt_A venv 로, 실물 대조는
사본의 모듈에 뿌리만 wt_A(정션 트리)로 주어 **읽기 전용**으로 했다. 본체 `.git/CTO_GATE_OPEN` 이 없을 때만 pytest 를 띄웠고
`-m "not resource_heavy" -p no:cacheprovider --basetemp=<스크래치> -B` 를 썼다.

## 0. 판정

| 등급 | 건수 | |
|---|---|---|
| **Critical** | **0** | |
| **Important** | **0** | |
| Minor | 2 | 일관성 (§7). 머지를 막지 않는다 |

**머지 가부: 가.** 게이트 SHA 는 **`598cd47`**. 명부 값은 바뀌지 않았고(A 파일 무변경), 동결 데이터·계약·임계값도 그대로다.

| 볼 것 | 판정 | 핵심 근거 |
|---|---|---|
| 명부가 B 쪽에 복제되지 않고 같은 객체인가 | **같은 객체** | `frozen_out.EXPECTED_SEALED is frozen_guard.EXPECTED_SEALED` · `verify_backup.EXPECTED_SEALED is …` 둘 다 True. 두 모듈 AST 에 `EXPECTED_SEALED` 재정의 없음(B 시험이 단언). 변이 N1(재정의)에서 실패 (§1, §5) |
| `restored` = `expected`, `lost` 만 제외 | **정확** | `PRESENT_STATUSES = {expected, restored}` · `ABSENT_STATUSES = {lost}`, 합집합 = `SEALED_STATUSES`. `verify_contract` 는 이 두 집합으로만 분기하고 어휘 밖은 `ValueError`. 변이 N2·N3·N4 에서 실패 (§2, §5) |
| `manifest_v1` 이 백업 대상에 실제로 드는가 | **든다** | 실물 뿌리(wt_A)로 `sealed_dirs()` → 7곳: `manifest_v1` + `cycle_pilot`·`v2` + `pairs_pilot_v1` + 복원 3곳. `pairs_pilot_v2`(lost) 제외. `BACKED_UP_STATUSES is PRESENT_STATUSES` (§3) |
| 뿌리 있고 봉인본 없음 = 실패 / 뿌리 없는 clone = 경고 | **규칙대로** | 가짜 뿌리: `restored` 자리 비움 → `missing_contract`(FAILING). 뿌리 없음 → `absent_tree`(실패 아님). `restored` 차 있음 → `ok` + `evidence`. `lost` 비어 있음 → `lost_recorded`(실패 아님) (§4) |
| B 변이 5종 재현 | **5/5 재현** | 각 변이가 해당 시험을 무너뜨린다 (§5) |

## 1. 명부 단일화

- `frozen_out.py`: 자체 `EXPECTED_SEALED` 딕셔너리(7항목)를 지우고 `from data.frozen_guard import EXPECTED_SEALED, SEALED_STATUSES`.
  `__all__` 에 `EXPECTED_SEALED` 를 그대로 두어 모듈 속성으로 다시 내보낸다 — `FO.EXPECTED_SEALED` 를 읽던 시험·`monkeypatch` 가
  그대로 돈다(내 실측: `check_registry()` 기본 인자가 A 명부를 읽는다).
- `verify_backup.py`: `from data.frozen_guard import EXPECTED_SEALED` 로 직접 읽고, `frozen_out` 에서는 `PRESENT_STATUSES` 만 가져온다.
- 명부 값 갱신은 A 파일에서 총괄 판정으로 한다는 주석이 두 모듈에 있다.

## 2. `verify_contract` — 상태 분기

```python
PRESENT_STATUSES = frozenset({"expected", "restored"});  ABSENT_STATUSES = frozenset({"lost"})
if status is not None and status not in PRESENT_STATUSES | ABSENT_STATUSES: raise ValueError(...)
... no_contract: status in ABSENT → lost_recorded ; else → missing_contract | absent_tree
... contract 있음: status in ABSENT → lost_but_present ; else → _verify_present 결과(ok/broken/incomplete_tree)
if status == "restored": out["evidence"] = expectation.get("evidence")
```

`restored` 가 `expected` 와 같은 경로를 타고, 근거 등급이 결과와 CLI 출력("복원 근거: …")에 옮겨진다. 어휘 밖 상태를
`expected` 로 읽어 넘기지 않는 거부는 명부 오타를 잡는 좋은 방어다 — A 가 `SEALED_STATUSES` 에 단어를 더하면
`test_상태_어휘를_빠짐없이_판정한다` 가 B 를 멈춰 세운다(그 단어가 "있어야 하는지"를 B 가 정해야 한다). 계약이 명확하다.

## 3. 백업 대상 — 실물 뿌리로 확인

`verify_backup.sealed_dirs(root=<wt_A>)` (읽기 전용):

```
corpus/generate/cycle_pilot · corpus/generate/cycle_pilot_v2
data/interim/manifest_v1                                        ← F 39번 I-1, 62f660b
data/processed/pairs_pilot_v1
data/processed/aihub71761_rt_v1_pilot3000 · …_crop_only · …_scale_control   ← restored (9595cbe)
```

`pairs_pilot_v2`(lost) 는 없다. 33번 §12 의 "24개 → 44개 파일, 약 54.6 MB" 는 이 7곳에서 나온 수치다(총괄 수용).

## 4. 실물 대조 — 대조기를 wt_A 뿌리로

`check_registry()`(뿌리 = wt_A, 정션 트리): `ok` 5(`manifest_v1`·`pairs_pilot_v1`·복원 3곳, 복원 3곳은 근거 등급 동반) ·
`incomplete_tree` 2(`cycle_pilot`·`v2` — 추적 밖 구성원이 이 트리에 없음, 기존) · `lost_recorded` 1(`pairs_pilot_v2`) · **실패 0.**
33번 §12 의 "봉인처 8곳 · 깨짐 0 · 소실 기록 1" 과 같다(B 는 wt_B 에서 구성원 0개 미대조, 여기는 추적 밖 구성원이 없는 트리라 2곳이 `incomplete_tree`).

규칙 확인(가짜 뿌리, `tmp`): 위 표. 특히 `restored` 자리가 비면 `missing_contract` 로 **실패**한다 — 45번 I-1 이 짚은
"복원 자리 하나가 사라져도 스위트가 초록" 구멍을 `test_실물_트리에서_명부_전체_대조가_실패_0이다` 가 닫는다.

## 5. 변이 시험 (스크래치 사본, B 5종 재현)

시험 2파일(`test_frozen_out.py`·`test_release_and_backup.py`), 자산 없는 사본이라 자산 전제 8건 skip.

| 변이 | 무엇을 바꿨나 | 결과 |
|---|---|---|
| N0 기준 | 없음 | 62 passed · 8 skipped |
| N1 | `frozen_out` 이 `EXPECTED_SEALED = dict(EXPECTED_SEALED)` 로 재정의(복제) | **1 failed** — `test_명부는_data_frozen_guard_의_단일_명부다` |
| N2 | `PRESENT_STATUSES` 에서 `restored` 제거 | **5 failed** — 어휘 완비·restored 판정·실물 전체 대조·백업 목록·소실 알림 |
| N3 | 어휘 밖 거부(`ValueError`) 제거 | **1 failed** — `test_어휘_밖_상태는_거부한다` |
| N4 | `evidence` 옮김 제거 | **1 failed** — `test_restored_는_expected_와_같게_판정하고_근거를_옮긴다` |
| N5 | `BACKED_UP_STATUSES` 를 별도 `frozenset` 객체로 | **1 failed** — `test_백업_목록은_봉인처_명부에서_파생된다`(`is` 단언) |

B 가 33번 §12 에 적은 다섯 변이와 같은 것이고 결과도 같다.

## 6. 45번(`57736e3`) — 사실 확인과 A 의 수용

B 의 9595cbe 검수. "pilot3000 계열 3곳만 lost → restored, evidence·record 있음, 나머지 5항목 diff 없음", "근거 등급이 실물과
일치", "원 경로 복사 00:58 → 상태 갱신 20:15" — 전부 A 기록과 같다. 판정 Important 1(I-1)·Minor 2 를 이 커밋과 함께 A 쪽에서 처리했다(별도 커밋, §8):

- **I-1 수용** — `test_restored_자리에_실물과_계약서가_있다` 의 skip 조건을 "세 자리 중 하나라도 없으면" 에서 **저장 뿌리
  `data/processed` 유무**로 바꿨다. 뿌리가 있는데 자리가 비면 실패다(B 의 실물 전체 대조 시험과 같은 규칙).
- **m-2 수용** — 전환기 전용이던 `test_B_명부와_어긋나지_않는다` 를 지웠다. 동일성·재정의 없음은 B 의 `test_명부는_data_frozen_guard_의_단일_명부다` 가 단언한다.
- **m-1 수용** — 두 팔 `evidence` 에 76번 구성 수치(N-crop 1,666 / N-crop 1,057·N-tile 606·N-band 3, 정상 10.2% / 43.0%)를 적었다.

## 7. Minor 2건 (598cd47)

| # | 내용 | 처리 |
|---|---|---|
| m-1 | `verify_backup.main()` 의 소실 안내가 `e.get("status") == "lost"` 를 그대로 센다 — 대조기는 `ABSENT_STATUSES` 로 판정하므로 같은 집합을 쓰는 편이 일관된다 | B 다음 커밋, 선택 |
| m-2 | `frozen_out.__all__` 에 `PRESENT_STATUSES`·`FAILING` 은 올렸는데 `ABSENT_STATUSES` 는 없다(시험은 속성으로 접근해 동작엔 영향 없음) | B 다음 커밋, 선택 |

## 8. A 쪽 후속 커밋

`data/frozen_guard.py`(두 팔 evidence 수치) · `tests/test_frozen_guard.py`(skip 조건, 전환기 시험 삭제). 검증은 §9.

## 9. 재현 명령 (전부 읽기 전용)

```
git diff 5e38f2e...598cd47 -- corpus tests
git archive 598cd47 | tar -x -C <스크래치>/598cd47_src
# 사본의 모듈을 wt_A 뿌리로: sealed_dirs(root=<wt_A>) · FO._REPO = <wt_A>; check_registry()
```

## 10. 하지 않은 것

- `wt_B` 체크아웃·수정 없음. `verify_backup plan` 실행 없음(드라이브 계획은 검수 범위 밖).
- `_regen_check/` 정리 없음(총괄 지시 대기).
