# 30. `data/processed` 봉인 자산 소실 경위 — 트랙 B

지시: `dispatch_B_processed_소실_경위.md` (2026-09-13 총괄, 최우선)
제약 이행: **아무것도 복원·재생성하지 않았다** · 남은 봉인 자산 미접촉 · `wt/B` 커밋만

---

## 0. 결론

**총괄 가설이 맞다. 내가 지웠다.** 09-11 15:43:56 에 내가 실행한
`git worktree remove --force` 두 번이 임시 트리에 붙은 `data/processed` 정션을 따라가
원본 디렉터리를 비웠다. 다른 원인은 없다.

세 가지를 덧붙인다. 순서대로 중요도가 높다.

| # | | |
|---|---|---|
| 1 | **손실 범위가 보고된 것보다 넓다** | 사라진 봉인 디렉터리는 2개가 아니라 **5개**다. `pairs_pilot_v1`·`v2` 외에 `aihub71761_rt_v1_pilot3000`·`_crop_only`·`_scale_control` 이 같이 없어졌다. 이쪽은 A·C·D·E 의 스크립트 10여 곳이 읽는다 |
| 2 | **`pairs_pilot_v1` 실물이 살아 있다** | 내 스크래치에 09-08 리허설 사본이 있고 4파일 전부 해시가 맞는다. 계약서까지 온전하다. **다만 `%TEMP%` 안이라 언제든 청소될 수 있다** |
| 3 | **같은 사고가 지금도 가능하다** | 워크트리 8개에 정션 **47개**가 살아 있고, 그 너머에 **33.95 GB**(raw 8.02 + interim 7.26 + outputs 18.67)가 있다. 이번에 날아간 4.29 MB 는 같은 동작의 가장 싼 경우였다 |

---

## 1. 과제 1·2 — 정확한 명령

09-11 F13·F14 패치 검수에서 `c516387` 을 임시 트리 둘에 떠서 비교했다. 원문 그대로 옮긴다.

**① 무패치 트리 `wt_review` 생성** (Bash, 15:33)

```bash
W="$SP/wt_review"
git worktree remove --force "$W" 2>/dev/null; rm -rf "$W"
git worktree add --detach "$W" c516387
```

**② 정션 부착 시도 — 실패** (Bash)

```bash
mkdir -p data && cmd //c "mklink /J \"$(cygpath -w "$W/data/processed")\" \"<저장소 루트>\\data\\processed\""
```

→ `파일 이름, 디렉터리 이름 또는 볼륨 레이블 구문이 잘못되었습니다.` 따옴표·경로 변환 문제로
정션이 만들어지지 않았다. 이때 `tests/corpus` 가 1건 실패했고(`test_v1_은_덮어쓸_수_없다`),
**그 실패가 "정션이 없다"는 신호였는데 나는 그것을 환경 잡음으로 읽고 정션을 붙이는 쪽으로 갔다.**

**③ 정션 부착 — 성공** (PowerShell)

```powershell
if (-not (Test-Path "$W\data\processed")) {
  New-Item -ItemType Junction -Path "$W\data\processed" -Target "<저장소 루트>\data\processed"
}
```

**④ 패치 트리 `wt_base` 생성 + 정션** (PowerShell)

```powershell
git worktree add --detach $B c516387
New-Item -ItemType Directory -Force "$B\data"
New-Item -ItemType Junction -Path "$B\data\processed" -Target "<저장소 루트>\data\processed"
```

**⑤ 삭제 — 이것이 원인이다** (Bash, 15:43:56)

```bash
git worktree remove --force "$SP/wt_review" && git worktree remove --force "$SP/wt_base" && git worktree prune
```

두 트리 다 `data/processed` 정션을 달고 있었다. 먼저 지운 `wt_review` 가 원본을 비웠고,
`wt_base` 차례에는 이미 빈 디렉터리였다.

---

## 2. 과제 3 — 그 명령이 정션을 따라갔는가

**따라간다. 추정이 아니라 측정했다.** 지시대로 사고를 재현하지 않았다 — 삭제가 전혀 없는
읽기 전용 관찰로 대신했다. 완전히 격리된 임시 디렉터리에 정션을 만들고 `git status` 와
`git clean -xdn`(**dry-run**)만 돌렸다.

```
LinkType=Junction  Target=…\jprobe1\target

$ git status --porcelain -uall
?? linked/CANARY.txt          ← 정션 너머의 파일을 개별 파일로 열거한다
?? linked/CANARY2.txt

$ git clean -xdn              (dry-run, 삭제 없음)
Would remove linked/

target 파일 수: 2             ← 아무것도 지워지지 않았다
```

핵심은 **`?? linked/CANARY.txt`** 다. git 이 정션을 심볼릭 링크로 봤다면 `?? linked` 한 줄로
끝나고 안으로 들어가지 않는다. 개별 파일을 열거했다는 것은 **git 이 정션을 평범한 디렉터리로
보고 그 안을 걸어 들어갔다**는 뜻이다.

원인은 Git for Windows 의 `lstat` 구현이다. 재파스 포인트를 심볼릭 링크로 분류할 때 태그
`IO_REPARSE_TAG_SYMLINK` 만 본다. 정션의 태그는 `IO_REPARSE_TAG_MOUNT_POINT` 라 그 검사를
빠져나가고 일반 디렉터리가 된다. `git worktree remove --force` 가 쓰는
`remove_dir_recursively()` 도 같은 분류를 쓰므로, 디렉터리라고 판단한 정션 안으로 재귀해
**원본 파일을 하나씩 지운다.** 정션 자체는 트리와 함께 사라진다.

측정 환경: `git version 2.49.0.windows.1`, `core.symlinks=false`.
`core.symlinks` 는 이 문제와 무관하다 — 그 설정은 git 이 링크를 *만들* 때 이야기이고,
여기서 문제인 것은 이미 있는 재파스 포인트를 *읽을* 때의 태그 판정이다.

### 시각 대조

| 시각 | 사건 | 근거 |
|---|---|---|
| 09-11 15:33:10 | `corpus_only.diff` 기록 → 직후 `wt_review` 생성 | 스크래치 파일 mtime |
| 15:35:17 / 15:35:50 | 패치·무패치 트리 `pytest` (정션 부착 후) | `pt_patched` / `pt_base` mtime |
| 15:38:18 / 15:39:37 | 재현 스크립트 작성 | mtime |
| **15:43:56.828** | **`data/processed` 비워짐** | 디렉터리 LastWrite |
| **15:43:57** | 29번 커밋 `e8bec3d` | `git log %cd` |

⑤의 삭제 명령과 커밋은 **같은 Bash 호출 안에서 `&&` 로 이어져 있었다.** 삭제 → 1초 뒤 커밋.
빈틈이 없다.

---

## 3. 과제 4 — 다른 원인 배제

| 후보 | 판정 | 근거 |
|---|---|---|
| 시험 픽스처 | 아니다 | `tests/corpus` 는 `tmp_path` 만 쓴다. `test_v1_은_덮어쓸_수_없다` 는 `main()` 이 가드에서 예외를 내는지만 본다 — 가드는 `mkdir` 전에 던지고 삭제 코드가 없다 |
| `make_pairs_pilot` | 아니다 | 쓰기만 하고 삭제 구문이 없다. 게다가 24번 가드가 봉인본을 겨눈 실행을 진입에서 막는다 |
| `snapshot_cycle_pilot` | 아니다 | `SNAPSHOT`·`EVIDENCE`·`SUMMARY` 세 파일만 쓴다 |
| `verify_backup` · `frozen_out` · `screen_public_release` | 아니다 | 전부 읽기 전용이거나 지정 경로에만 쓴다. 삭제 구문 없음 |
| pytest `--basetemp` 청소 | 아니다 | 대상이 `$SP/pt_base`·`$SP/pt_patched` 다. `data/` 아래가 아니다 |
| 09-08 의 `rm -rf "$SP/drive"`·`"$SP/live"` | 아니다 | 사본(일반 파일)이고 정션이 없었다. 날짜도 09-08 이다 |
| 다른 트랙의 워크트리 삭제 | 아니다 | 같은 시각대에 만들어진 `wt_C_pr`(15:32:24 생성, `data\processed` 정션 있음)는 **아직 살아 있다.** 지워진 적이 없다 |
| ①의 `rm -rf "$W"` | 아니다 | 그 시점에 `$W` 는 존재하지 않았고 정션도 아직 없었다 |

**남는 것은 ⑤ 하나다.**

---

## 4. 손실 범위 정정 — 5개 디렉터리

09-08 에 내가 돌린 `frozen_out` 전수 점검이 `data/processed` 안의 봉인 디렉터리를 **5개**로
세었다. 전부 `[ok]`(계약 구성원 실물 일치)였다.

```
○ data\processed\aihub71761_rt_v1_pilot3000                구성원 4개  [ok]
○ data\processed\aihub71761_rt_v1_pilot3000_crop_only      구성원 4개  [ok]
○ data\processed\aihub71761_rt_v1_pilot3000_scale_control  구성원 4개  [ok]
○ data\processed\pairs_pilot_v1                            구성원 3개  [ok]
○ data\processed\pairs_pilot_v2                            구성원 4개  [ok]
```

지금 `data/processed` 는 파일 0개다. 즉 **19개 계약 구성원 + 계약서 5장이 전부 사라졌다.**
총괄 보고는 `pairs_pilot_*` 만 짚었는데, `aihub71761_rt_v1_pilot3000` 계열 셋이 같이 없어졌고
이쪽 의존이 훨씬 넓다.

**그 셋을 읽는 코드** (내 소관 밖 포함, 보고만 한다):

```
evaluation/params.py:139          PILOT_SNAPSHOT
vlm/pilot_vlm.py:43               PAIRS_PATH (= pairs_pilot_v1/pairs.jsonl)
scripts/pilot_c.py:25,247         SNAPSHOT_DIR · pairs_pilot_v1/counts.json
scripts/pilot_export_det.py:33    SNAPSHOT_DIR
scripts/pilot_export_vlm.py:31    SNAPSHOT_DIR
scripts/make_ablation_arms.py:57  PILOT
scripts/probe/diagnostics3.py:33  · judge_cost_probe.py:41 · p9_uni_diagnosis.py:23
corpus/generate/make_pairs_pilot.py:51        SNAP
corpus/validate/compare_pair_generators.py:28 SNAP
```

`pilot3000` 의 digest 는 61번 3줄에 **`c0a254a8…` 8자만** 남아 있다. 파일별 해시 기록은
어디에도 없다 — 내 09-08 백업 매니페스트는 `pairs_pilot_*` 만 덮었다.

---

## 5. 원래 sha256 은 기록돼 있는가 — **있다. 세 겹이다**

| 기록 | 위치 | 내용 | 내구성 |
|---|---|---|---|
| 백업 매니페스트 | `%TEMP%…/scratchpad/backup_manifest.json` (09-08 21:53) | 9파일 **전체 sha256 + 바이트 수 + 계약 기록 해시** | **취약 — `%TEMP%`** |
| 24번 부록 C | `docs/…/24_동결가드_B.md` (커밋·main) | 9파일 sha256 **앞 16자** + 바이트 수 | 영구 |
| v1 계약서 실물 | `%TEMP%…/scratchpad/live/pairs_v1/SNAPSHOT.sha256` | v1 계약 전문 + `snapshot_digest 63fc6b6e…` | **취약** |
| v2 digest | `81_체크리스트_B.md:330` | `472a5d40ddd2a030…` (전체 64자) | 영구 |

즉 **"무엇이 있었는지"는 안다.** `pairs_pilot_*` 9파일의 이름·크기·해시가 전부 살아 있다.
`pilot3000` 계열 셋은 그렇지 않다 — 8자 digest 하나뿐이다.

### 그리고 `pairs_pilot_v1` 은 실물이 살아 있다

09-08 에 가드 실물 확인을 하면서 봉인본을 임시 경로에 복사해 CLI 를 돌렸다. 그 사본이
지워지지 않고 남았다. **읽기만 해서** 해시를 대조했다.

```
scratchpad/live/pairs_v1/
  pairs.jsonl       2,073,786 B  일치  beb98b2e3d965fa0…
  counts.json             859 B  일치  ace7003c6c5ff197…
  discarded.jsonl          84 B  일치  17882a9019c4a35c…
  SNAPSHOT.sha256         325 B  일치  cb0b1392b8e98247…   (CRLF 보존)
전체 일치: True
```

**바이트 단위로 온전한 v1 한 벌이다.** 계약서의 CRLF 까지 그대로다(24번 §210 이 적은 그 특성).
복원은 하지 않았다 — 지시대로다. **`v2` 는 사본이 없다.** 09-08 백업 리허설에서 25개를
임시 경로에 복사했지만 그 디렉터리는 리허설 끝에 지웠다.

> **가장 급한 것:** 위 두 항목은 `%TEMP%` 안에 있다. Storage Sense 나 임시 파일 청소가 한 번
> 돌면 사라진다. 내 판단으로는 **지금 당장 밖으로 빼야 한다.** 다만 그것이 "복원"에 해당할 수
> 있어 손대지 않았다. 총괄이 직접 하려면 이 한 줄이면 된다 (읽기만 하고 옮기지 않는다):
> ```powershell
> Copy-Item -Recurse "<SCRATCH>\live\pairs_v1" "<SHARE>\<보관처>\pairs_pilot_v1_구조"
> Copy-Item "<SCRATCH>\backup_manifest.json" "<SHARE>\<보관처>\"
> ```

---

## 6. 재생성하면 바이트가 같은가

**`pairs_pilot_v1` — 불가능하다. 같은 바이트가 나올 수 없다.** v1 은 08-31 코드의 산출이고
그 뒤 셋이 바뀌었다. ① 좌표 규약이 `ABS_ORIG` 로 전환됐다(main `47c4dbc`, 09-02). ② v2 부터
`PAIRS_META.json` 이 생겨 산출 파일이 3개에서 4개가 됐다 — **계약 구성원 수 자체가 다르다.**
③ `newline=""` 수정이 08-31 에 들어갔고, v1 계약서만 CRLF 인 것이 그 증거다(24번 §210).
지금 코드는 v1 을 만들 수 없고 v2 모양을 낸다.

**`pairs_pilot_v2` — 산출 자체는 결정적이지만, 지금은 돌릴 수 없다.** 코드는 결정적이다:
출력 순서가 `tv.sort_values("image_id")`, 결함은 `sorted(…, key=ann_id)`, 조항은 `sorted(…)`,
재질은 `sorted({…})`, 쓰기는 전부 `newline=""` 다. 난수·해시 순서 의존이 없다. 09-08 에 내가
`make_pairs_pilot` 을 고친 것은 `--out` 필수화와 가드뿐이라 산출 바이트에 닿지 않는다.
**그런데 입력이 없다.** `SNAP = data/processed/aihub71761_rt_v1_pilot3000` 이 같은 삭제로
함께 사라졌다. 그 스냅샷을 먼저 되살리지 않으면 v2 재생성은 **시작조차 못 한다.**

그리고 그 스냅샷을 되살려도 **바이트 동일을 보증할 수 없다.** `make_pilot_subset.py` 가
`manifest_v1` 에서 3,000장을 다시 뽑아야 하는데, 그 표집이 같은 결과를 낼지는 A 트랙이
확인해야 한다. digest `c0a254a8…` 8자로 대조는 되지만 파일별 해시가 없어 **어디가 달라졌는지**는
못 짚는다.

### 규약 1-6 저촉

1-6 은 "corpus·페어·색인은 sha256 부여 후 **재생성 금지**"다. 지금 상황은 그 조항이 상정한
"재생성하고 싶다"가 아니라 **"원본이 없어졌다"**이므로 조항을 그대로 적용할 수는 없다. 다만
**재생성물을 v1·v2 라는 이름으로 되돌리는 것은 금지에 정면으로 걸린다** — 논문에 실을 해시가
가리키는 실물이 아니게 된다. 세 갈래로 갈린다. **판정은 총괄.**

1. **v1 은 복원한다** (재생성이 아니다). 사본의 해시가 계약과 일치하므로 **동일성이 증명된
   복원**이다. 1-6 과 충돌하지 않는다 — 같은 바이트가 같은 자리로 돌아가는 것뿐이다.
2. **v2 는 `v3` 로 새로 만든다.** 같은 이름으로 되돌리지 않는다. 새 digest 를 부여하고
   의사결정로그에 "v2 는 09-11 사고로 소실, v3 가 후속" 을 남긴다. v2 를 인용한 문서
   (`81_체크리스트_B.md`, 24번 부록 C)는 소실 사실을 주석으로 단다.
3. **v2 를 되살리지 않는다.** D4 페어 축소본은 파일럿 자산이고 본실험에 안 닿는다(§7).
   필요해지는 시점에 v3 를 만든다.

내 의견은 **1 + 3**이다. v1 은 증명 가능한 복원이라 값이 싸고, v2 는 지금 만들 이유가 없다 —
만드는 순간 `pilot3000` 재생성이 선행되고 그것이 또 하나의 "이름은 같은데 바이트는 다른" 자산을
낳는다. 필요할 때 만들면 된다.

---

## 7. 본실험·논문에 닿는가 — **닿지 않는다**

총괄 판단이 맞다. 반박 근거를 찾으려 했지만 없다.

- **`detection/` 과 `fl/` 에 `data/processed` 참조가 0건이다.** 검출 3시드·연합 학습은
  `data/interim` 매니페스트로 돌았고 그쪽은 무사하다(raw 8.02 GB·interim 7.26 GB, 계약 4파일 해시 일치).
- 사라진 것을 읽는 코드는 전부 **파일럿·프로브·어블레이션 준비** 계열이다(§4 목록).
  `evaluation/params.py:139` 는 파일럿 스냅샷 상수이고 본채점 경로가 아니다.
- `pairs_pilot_*` 는 D4 페어 **축소본**이다. 24번이 적었듯 "알려진 결함 421건"의 회계가 v1 에
  붙어 있는데, **그 회계 수치는 `81_체크리스트_B.md` 와 `test_pairs_pilot.py::test_v1_회계가_실측과_맞는다`
  에 숫자로 박혀 있다**(219 + 286 − 84 = 421, 0.2816). 실물이 없어도 주장은 문서·시험에 남는다.
- 다만 그 시험은 지금 **실물을 읽지 않는다**(상수 대조라 여전히 통과한다). 즉 **시험이 초록인
  것이 실물이 있다는 뜻이 아니다.** 이것은 24번 A-2 에서 내가 만든 `verify_contract` 가
  `broken` 으로 잡아 줄 자리인데, `data/processed` 는 디렉터리 자체가 비어 계약서가 없으므로
  `no_contract` 로 조용히 넘어간다. **그 구멍은 내 가드의 결함이다** — §8-4 에 적었다.

---

## 8. 재발 방지 제안 (실행 안 함, 판정은 총괄)

### 8-1. 지금 살아 있는 위험

워크트리 8개에 정션 **47개**가 붙어 있고 그 너머 용량은 이렇다.

| 정션 대상 | 파일 수 | 용량 |
|---|---|---|
| `data/raw` | 20 | **8.02 GB** |
| `data/interim` | 186,984 | **7.26 GB** |
| `outputs` | 373,412 | **18.67 GB** |
| `data/processed` | 0 | 0 (이번에 비워짐) |
| `checkpoints` · `mlruns` | 0 | 0 |

**이번에 날아간 4.29 MB 는 같은 동작의 가장 싼 경우였다.** 어느 트랙이든 자기 워크트리에
`git worktree remove` 나 `git clean -xdf` 를 한 번 돌리면 33.95 GB 가 같은 방식으로 사라진다.
`wt_C_pr` 는 09-11 에 검수용으로 만든 임시 트리인데 **아직 살아 있고 정션 5개가 붙어 있다** —
그것을 정리하려는 순간이 정확히 위험 시점이다.

### 8-2. 규약에 넣을 문장 (제안)

> **정션이 붙은 워크트리는 `git worktree remove` 로 지우지 않는다.** Windows 의 정션은 git 에게
> 평범한 디렉터리로 보이고, 삭제가 그 안으로 들어가 **본체를 지운다.** 순서는 하나뿐이다 —
> **정션을 먼저 떼고, 그 다음 트리를 지운다.**

### 8-3. 안전한 삭제 순서 (격리 환경에서 확인함)

```powershell
# ① 트리 안의 정션을 전부 찾는다
$w = "<워크트리 경로>"
$j = Get-ChildItem -LiteralPath $w -Recurse -Force -Directory -ErrorAction SilentlyContinue |
     Where-Object { $_.LinkType }
$j | ForEach-Object { "{0} -> {1}" -f $_.FullName, ($_.Target -join ',') }

# ② 정션만 뗀다 (본체는 그대로)
$j | ForEach-Object { $_.Delete() }

# ③ 정션이 0개인 것을 확인한 뒤에 트리를 지운다
git worktree remove --force $w
```

②의 `.Delete()` 가 정션만 제거하고 대상을 건드리지 않는 것을 throwaway 디렉터리에서 확인했다
(대상 파일 2개 → 정션 생성 → `.Delete()` → 대상 파일 2개 유지).

### 8-4. 코드로 막을 것 — 두 개

**(가) 트리 삭제 가드.** 위 ①을 스크립트로 만들어 정션이 하나라도 있으면 거부한다. 24번에서
내가 만든 `frozen_out.assert_not_frozen` 과 같은 계열이다 — "지우기 전에 무엇을 지우는지 본다".
소관이 `corpus/` 라 위치는 총괄이 정해야 한다(`scripts/` 가 맞아 보인다).

**(나) `verify_contract` 의 구멍.** 지금 구현은 계약서가 없으면 `no_contract` 로 조용히 넘어간다.
그래서 **디렉터리째 사라진 이번 사고를 시험이 못 잡았다.** 09-11 에 `tests/corpus` 507건이
전부 초록이었는데 그 순간 `data/processed` 는 이미 비어 있었다. 고칠 방향은 "있어야 할 봉인처
목록"을 따로 들고, 그 목록에 있는데 계약서가 없으면 **`missing_contract` 로 실패**시키는 것이다.
목록은 `verify_backup.DEFAULT_DIRS` 가 이미 들고 있다. **내 가드의 결함이고 내가 고쳐야 한다 —
지시가 코드 변경을 막고 있어 제안만 남긴다.**

### 8-5. 백업이 실행됐다면 이 사고는 0 이었다

09-08 에 내가 목록·절차·검증기를 다 만들었고 총괄이 승인했지만 **복사가 실행되지 않았다.**
매니페스트만 `%TEMP%` 에 남았다. 도구를 만든 것과 백업이 존재하는 것은 다르다. **"백업 도구
준비 완료"를 백업으로 세지 않는 절차**가 필요하다 — `verify_backup verify` 가 종료코드 0 을
낸 기록이 있어야 백업으로 인정하는 식이다.

---

## 9. 내 과실

변명하지 않는다. 세 군데서 끊을 수 있었다.

1. **정션이 없어 시험이 깨졌을 때** 그것을 "환경 사유"로 처리하고 정션을 붙였다. 그 시험은
   봉인본을 못 봐서 깨진 것이고, 정션을 붙이는 대신 그 시험 하나를 건너뛰면 끝날 일이었다.
   임시 검수 트리에 33.95 GB 로 가는 통로를 낼 이유가 없었다.
2. **정션을 붙였으면 뗄 책임도 같이 졌어야 했다.** 붙일 때는 `New-Item -ItemType Junction` 을
   의식적으로 썼으면서, 지울 때는 `git worktree remove --force` 를 아무 확인 없이 두 번 돌렸다.
3. **바로 그 주에 내가 "정션이라 물리 사본이 한 곳뿐"이라고 24번에 적어 놓고도** 그 위험을
   내가 실행했다. 24번 A-3 의 그 문장이 이번 사고의 예고였다.

---

## 10. 총괄 판정이 필요한 것

1. **`%TEMP%` 의 v1 사본과 백업 매니페스트를 밖으로 빼는 것** (§5). 가장 급하다. 청소 한 번에
   사라진다. 내가 하지 않은 이유는 "복원 금지" 때문이다 — 한 줄 지시면 즉시 한다.
2. **v1 복원 / v2 처리** (§6). 내 의견은 v1 은 증명 가능한 복원이라 되돌리고, v2 는 지금
   만들지 않는다.
3. **`pilot3000` 계열 셋** (§4). 내 소관 밖이다. A 트랙이 `make_pilot_subset.py` 로 되살릴 수
   있는지, 되살린 것이 `c0a254a8…` 와 맞는지 확인이 필요하다. 여기가 실질 병목이다 — v2 도
   파일럿 스크립트들도 이것 없이는 못 돈다.
4. **정션 47개 처리** (§8-1). 특히 `wt_C_pr` 를 정리할 때 §8-3 순서를 반드시 쓰도록.
5. **`verify_contract` 구멍 수리** (§8-4 나). 내가 고칠 것이나 이번 지시 범위 밖이라 대기한다.

## 11. 이 조사에서 한 일 / 하지 않은 일

**하지 않았다** — 복원, 재생성, 사고 재현, 봉인 자산 수정, 코드 변경, main 머지.
**했다** — 파일시스템·git 이력 읽기, 해시 대조(읽기만), 완전 격리된 임시 디렉터리에서
`git status`·`git clean -xdn`(dry-run)·정션 `.Delete()` 관찰.
남은 봉인 자산(`cycle_pilot`·`cycle_pilot_v2`·`manifest_v1`)은 열지 않았다.
관찰용 임시 디렉터리 `scratchpad/jprobe1` 은 스크래치 안에 있고 프로젝트 경로와 무관하다.

---

# 부록 A. §5 구조·복원 실행 기록 (2026-09-13, 총괄 판정)

총괄이 **§5 구조 범위에 한해** 복원 금지를 해제했다. 지정한 순서 그대로 실행했고 단계마다
멈춰서 확인했다. 총괄이 같은 사본을 독립 대조한 결과와 내 측정이 일치한다.

## A-1. 결과 한 줄

**`pairs_pilot_v1` 은 계약서까지 바이트 동일로 복원됐다. `pairs_pilot_v2` 는 소실 확정이다.**

## A-2. 단계별 기록

**① 증거 보존 — 비임시 위치로 복사** (위험 0, 원본 미삭제)

```
<RESCUE>\
  backup_manifest.json                10,706 B   (2026-09-08 21:53:53)
  pairs_pilot_v1\pairs.jsonl       2,073,786 B   (2026-09-08 19:41:46)
  pairs_pilot_v1\counts.json             859 B
  pairs_pilot_v1\discarded.jsonl          84 B
  pairs_pilot_v1\SNAPSHOT.sha256         325 B
```

`%TEMP%` 원본은 그대로 뒀다(4파일 + 매니페스트 존재 확인). 스크래치의 폴더 이름은 09-08
리허설 때 쓴 `pairs_v1` 이었고, 구조본은 **원 자산 이름 `pairs_pilot_v1`** 으로 담았다.
구조 경로는 저장소·워크트리 12곳 어디에도 속하지 않고 git 이 인식하지 않는다. 로컬 `E:` 라
국외 반출에도 걸리지 않는다.

**② 해시 대조 — 4/4 일치**

| 파일 | 크기 | 기록 sha256 | 판정 |
|---|---|---|---|
| `pairs.jsonl` | 2,073,786 B | `beb98b2e3d965fa014be660344a5499dea2ed2b87784e77769b79f7c2dd97e8d` | 일치 |
| `counts.json` | 859 B | `ace7003c6c5ff197e2411e6a79f834b26d224a87c2a5b716fd4921752c54e401` | 일치 |
| `discarded.jsonl` | 84 B | `17882a9019c4a35cde7472dd5e76f9047ecc749c07803017b3a352e96a565561` | 일치 |
| `SNAPSHOT.sha256` | 325 B | `cb0b1392b8e98247c41316d71e7e7938c330298d72d5016fd3b143cc861b8bbb` | 일치 |

> **지시의 "9/9" 정정 (총괄이 스스로 정정한 것과 같은 결론).** 매니페스트의 `pairs_pilot`
> 항목은 9개이지만 그것은 **v1 4개 + v2 5개 합계**다. 살아남은 사본은 v1 뿐이라 대조 가능한
> 모집단이 4개이고, 그 4개가 전부 맞았다. 즉 **v1 기준 완전 일치**다.

계약서 내부 정합성도 확인했다 — 계약서가 적은 세 구성원 해시가 실물과 맞고,
`snapshot_digest` 를 재계산하니 기록된 `63fc6b6e2d89e6fa5fba077de62afd4db43eb8f5f38ddda096cb9be78363efcc`
와 같다. 줄바꿈은 **CRLF** 로, 24번 §210 이 적은 원본 특성 그대로다.

**계약서 복원 조건 충족.** 총괄이 "매니페스트가 계약서 자체의 해시를 갖고 있는지 먼저
확인하라"고 했다. 갖고 있다 — `sha256 = cb0b1392…`. (`contract_sha256` 이 `None` 인 것은
계약서가 자기 해시를 담을 수 없어서이지 기록이 없다는 뜻이 아니다.) 그러므로 계약서도 복원했다.

**③ 복원 — `data/processed/pairs_pilot_v1/`**

정션이 아닌 본체 경로 `<REPO>/data/processed/pairs_pilot_v1` 에 직접 썼다.
복원 전 `data/processed` 파일 수 0 을 확인하고, 기존 디렉터리가 있으면 중단하도록 걸었다.

복원 후 확인:

```
복원본 4/4 해시 일치 (매니페스트 기준)
계약서 줄바꿈: CRLF 보존됨
wt_B 정션을 통해서도 4파일 보임

$ uv run python -m corpus.generate.frozen_out
○ corpus\generate\cycle_pilot     구성원 8개  [ok]
○ corpus\generate\cycle_pilot_v2  구성원 6개  [ok]
○ data\processed\pairs_pilot_v1   구성원 3개  [ok]     ← 다시 초록
봉인 3곳 · 깨짐 0 · 대조 못 한 구성원 0개
```

**이것은 재생성이 아니라 복원이다.** 같은 바이트가 같은 자리로 돌아갔고 해시가 그것을
증명한다. 규약 1-6 은 "sha256 부여 후 재생성 금지"이지 "동일성이 증명된 복원 금지"가 아니다.
`snapshot_digest 63fc6b6e…` 는 사고 전과 같은 값이므로 논문에 실을 해시가 가리키는 실물이
그대로 있다.

## A-3. `pairs_pilot_v2` — 소실 확정

**소실했다. 사본이 없다. 재생성도 불가능하다.** 세 가지가 각각 독립적으로 이것을 확정한다.

1. **사본 없음.** 09-08 리허설 폴더 `live/pairs_v1` 에는 `SNAPSHOT.sha256`·`counts.json`·
   `discarded.jsonl`·`pairs.jsonl` 네 개뿐이고 **`PAIRS_META.json` 이 아예 없다.** 그 파일은
   v2 전용이므로, 이 폴더가 v2 사본이 아니라는 증거다. 09-08 백업 리허설에서 25개를 임시
   경로에 복사했지만 그 디렉터리는 리허설 끝에 지웠다.
2. **전 세션 스크래치를 훑어도 없다.** `%TEMP%\claude` 아래 모든 세션 디렉터리에서
   `pairs.jsonl`·`PAIRS_META.json` 을 찾으면 v1 것 하나뿐이다.
3. **재생성 불가.** 입력 `SNAP = data/processed/aihub71761_rt_v1_pilot3000` 이 같은 삭제로
   함께 사라졌다. 그 스냅샷 없이는 `make_pairs_pilot` 이 **시작조차 못 한다.**

남는 기록은 이것뿐이다 — 5파일의 이름·크기·sha256(매니페스트 + 24번 부록 C)과
`snapshot_digest 472a5d40ddd2a0308510ea5ff767081b02914a107ed03f7da22e2d38ec3c87c2`
(81번 330줄). **무엇이 있었는지는 알지만 실물은 없다.**

총괄 판정대로 **만들지 않는다.** 없는 것을 없다고 기록하는 편이 낫다. v2 를 인용하는 문서
(`81_체크리스트_B.md`, 24번 부록 C)는 이 절을 가리키는 주석이 필요하다 — 그것은 다음 지시를
기다린다.

## A-4. 무사한 것

`cycle_pilot` 9/9 · `cycle_pilot_v2` 7/7 이 매니페스트와 일치한다(작업 트리 실물 기준).
`corpus/generate/` 는 정션이 아니라 워크트리 실물이라 이번 삭제에 닿지 않았다.

## A-5. 아직 남은 것

- **`aihub71761_rt_v1_pilot3000` 계열 3개는 여전히 소실 상태다** (§4). 사본도 파일별 해시
  기록도 없다. v2 재생성도, 파일럿 스크립트 10여 곳도 전부 여기서 막힌다. **A 트랙 소관.**
- **백업은 아직 실행되지 않았다.** 이번에 복원된 v1 을 포함해 `verify_backup plan` 이 지금
  20개 / 3.27 MB 를 든다(v2 5개가 빠져 25 → 20). 구조본과 매니페스트가 `E:` 로컬 한 곳에만
  있으므로, **이번 복원의 유일한 증명이 다시 단일 사본이다.** 국내 공유 드라이브 적재가
  남았다 — 24번 부록 C 절차 그대로다.
- **워크트리 동결 중.** `wt_C_pr` 의 정션 5개를 포함해 47개가 살아 있고, 총괄이 안전 절차를
  정할 때까지 아무도 워크트리를 지우지 않는다.
- **`verify_contract` 구멍 수리는 이번 구조 뒤로 승인됐다** (§8-4 나). 대기한다.

---

# 부록 B. 사람 검증 시트 교체 (게시 차단, 09-13 총괄 승인)

29번 §2-2 가 올린 건이다. 추적 산출물 `corpus/validate/judge_labels/sheet_v1.jsonl`
(커밋 `8ad3885`, 09-02)이 기계 판정을 그대로 담고 있었다.

**선행 조건 확인 (3).** `labels_v1.jsonl` 이 없다 — 사람 응답이 아직 하나도 없으므로
교체로 잃는 라벨이 없다. 지금이 가장 싼 시점이라는 29번 판단이 유효했다.

## B-1. 전후

| | 교체 전 | 교체 후 |
|---|---|---|
| `judge_pass`/`judge_fail` 문자열 | **26건** | **0** |
| `stratum` 키 | **25건** | **0** |
| `judge_*` 키 | 0 | 0 |
| 층 블록 순서 (인접 층 변화) | **1** (통과분 18 → 기각분 7, 순서가 곧 판정) | **8** (섞임) |
| 행 키 | `…, stratum, …` | `axis·basis·human_ok·labeler·note·sample_id·text` |
| **표집된 sample_id 집합** | 25 | **동일** |

표본 자체는 바뀌지 않았다. `rng.shuffle` 이 층별 추출 **뒤**에 오므로 무엇을 뽑는지는
그대로고 순서만 섞인다.

## B-2. 층별 메타 — 삭제가 아니라 분리 (총괄 조건 1)

Codex 패치는 `_shortfall` 을 `{"total": 75}` 로 뭉갰다. 그러면 층별 표집확률 `p = k/N` 을
낼 근거가 사라지고, 층 가중을 못 하면 정밀도·재현율이 어느 모집단의 값인지 말할 수 없다 —
**층화 표집을 한 이유 자체가 없어진다.** 그래서 지우지 않고 옮겼다.

`build_sheet` 이 이제 `(사람용 시트, 표집 메타)` 둘을 돌려준다. 산출도 셋으로 갈린다.

| 파일 | 내용 | 추적 |
|---|---|---|
| `sheet_v1.jsonl` | 사람이 보는 것. 층 없음, 순서 섞임 | 추적 |
| `sheet_v1.meta.json` | **집계** — 층별 `N_population`·`k_drawn`·`target`·`shortfall`·`p_sampling`, 시드, 층화 기준 후보 | 추적 |
| `sheet_v1.strata.json` | **항목별 층** (= 항목별 후보 판정) | **미추적** |

집계는 추적한다 — 감사가 "층별 결과·표집확률을 보고한다"고 한 것이고 판정을 담지 않는다.
**항목별 층은 추적하지 않는다.** 그것은 곧 그 항목의 후보 판정이라, 시트 옆에 추적돼 있으면
라벨러가 저장소만 열어도 답을 본다. 필요하면 `_stratum()` 으로 다시 계산하면 되므로 저장할
이유가 없다. `.gitignore` 에 `*.strata.json` 과 `labels_*.jsonl` 을 넣었다.

실측된 집계:

```
조항검색_기준서술|judge_pass   N=18  k=18  목표=25  부족= 7  p=1.0
조항검색_기준서술|judge_fail   N= 7  k= 7  목표=25  부족=18  p=1.0
조치서술|judge_pass            N= 0  k= 0  목표=25  부족=25  p=null
조치서술|judge_fail            N= 0  k= 0  목표=25  부족=25  p=null
총 부족분 75
```

여기서 드러나는 사실 하나 — **지금 층화는 이름뿐이다.** 두 유효 층에서 모집단을 **전부**
뽑았으므로(`p = 1.0`) 가중이 필요 없고, 조치 축 두 층은 모집단이 0이다(조치 축 레코드에
판정 키가 없다). 목표 100건 중 25건만 채운 것이 그 결과다. 층 가중이 실제로 문제가 되는
것은 모집단이 커진 뒤이고, **그때를 위해 지금 기록을 남겨 두는 것**이다.

## B-3. 짚어야 할 것 둘

**(가) 이 교체는 HEAD 만 고친다. git 이력의 누설본은 남는다.** 누설 시트는 `8ad3885`
(09-02)에 커밋돼 있고, 공개 GitHub 저장소에서 그 커밋을 꺼내면 그대로 읽힌다. 파일을
바꿔도 이력은 안 바뀐다. 지우려면 이력 재작성이 필요한데 그건 이 지시 범위 밖이고,
총괄이 Codex 에 "이력 재작성 금지"를 이미 걸어 두었다. **총괄 판정 사항으로 올린다.**

**(나) 블라인드는 기술적 통제가 아니라 절차적 통제다.** 시트에서 판정을 빼도,
**추적 중인** `corpus/generate/cycle_pilot_v2/EVIDENCE.jsonl` 이 `sample_id` 별로
`judge`·`judge_pass` 를 담고 있다. 시트 25건 **전부** 그 파일에서 판정을 찾을 수 있다.
즉 저장소에 접근할 수 있는 라벨러는 여전히 답을 볼 수 있다. 시트 교체는 "우연히 보이는 것"을
막지 "찾으면 못 찾게" 하지는 못한다. 실질 통제는 **라벨러에게 저장소를 주지 않는 것**이고,
그 전제를 라벨링 절차에 명시해야 한다. EVIDENCE 는 통과율의 근거 보존물이라(74번 P5)
지울 수 없다 — 상충하는 두 요구가 만나는 자리다. **판정 요청.**

## B-4. 보존 (총괄 조건 2)

원본 시트와 대응표를 `_workspace/2026-09-13-sheet-replace/` 에 뒀다(`.gitignore:87`로 미추적).
`sheet_v1_ORIGINAL_leaked.jsonl`(18,135 B) + `correspondence.json`(25행: 원본 행 ↔ 신규 행
↔ 층). 실제 경로는 미추적 `dispatch_LOCAL_B_경로표.md`.

## B-5. Codex 패치와의 관계

**내 변경이 Codex 미커밋 패치의 상위집합이다.** 같은 파일을 건드리므로 게이트에서 충돌한다.
Codex 패치의 네 가지(행에서 `stratum` 제거 · `rng.shuffle` · `reasoning_pending.jsonl` 읽기 ·
출력 문구)는 그대로 들어갔고, `_shortfall` 뭉개기만 메타 분리로 바꿨다. **내 쪽을 취하면 된다.**
시험도 29번 F14-P2 지적대로 고쳤다 — 키만이 아니라 **값** 안의 `judge_pass` 를 훑고, 순서가
층 블록이 아닌지 인접 층 변화 수로 잰다(옛 시험은 키만 봐서 값 누설을 구조적으로 못 봤다).

---

# 부록 C. 규약 2-6 위생 — 로컬 절대경로 분리

총괄이 지목한 4파일 11건에, 내가 09-13 에 쓴 30번 6건을 더해 **17건**을 처리했다.
방식은 main 이 같은 날 `11_재부팅_복구절차.md` 에 쓴 것과 같다 — 자리표시자 치환 +
미추적 파일 분리.

| 파일 | 건수 | 조치 |
|---|---|---|
| `corpus/parse/extract_candidates.py` | 3 | **코드 변경** — 아래 C-1 |
| `docs/corpus_후보_판정.md` | 2 | `<SHARE>/…` 치환, 실패 사유는 "원본 폴더 이름에 한글·공백" 으로 일반화 |
| `24_동결가드_B.md` | 5 | `<REPO>`·`<WT_B>` 치환 |
| `30_processed_소실_경위_B.md` | 6 | `<REPO>`·`<RESCUE>`·`<SCRATCH>`·`<SHARE>` 치환 (세션 식별자 포함) |
| `tests/corpus/test_release_and_backup.py` | 1 | **의도적 — 두었다.** C-2 |

실제 경로는 `docs/dev_log/2026-09-03-본실험/dispatch_LOCAL_B_경로표.md` 에 뒀다
(`.gitignore:81` 의 `dispatch_*.md` 로 미추적). 자리표시자 ↔ 실제 경로 대응, 왜 가리는지,
구조본·보존본 위치가 거기 있다.

## C-1. `extract_candidates.py` — 위생이자 이식성

`SRC`/`DST` 가 공유 드라이브 경로로 박혀 있었다. 총괄 지적대로 **경로 한 줄이 "어느 조직이
어떤 자료를 어디에 두는가"를 말한다** — 폴더명이 학회 이름이고, 그 아래가 비반출 자료의
적재 위치다. 그리고 다른 기계에서는 아예 돌지 않는다.

인자·환경변수로 뺐다. 둘 다 없으면 무엇을 줘야 하는지 말하고 멈춘다.

```
$ uv run python -m corpus.parse.extract_candidates
원본·추출물 폴더를 줘라 — --src/--dst 또는 $WELDFL_CORPUS_SRC/$WELDFL_CORPUS_DST.
  경로를 소스에 박지 않는다 (규약 2-6): 저장소가 공개라 비반출 자료의 적재 위치가
  드러나고, 다른 기계에서는 돌지 않는다.
```

`extract_one()` 이 모듈 전역 `DST` 를 읽던 것도 인자로 바꿨다 — 전역이 사라지면 조용히
`NameError` 가 날 자리였다. 주석의 공유 드라이브 경로 언급도 일반화했다.

## C-2. 시험 픽스처는 의도적이다 — 보고만 한다

`tests/corpus/test_release_and_backup.py:55`

```python
hits = S.screen_text(r"<저장소 루트>\corpus 에서 읽었다", set())
assert "local_path" in [h[0] for h in hits]
```

**공개 검수기의 `local_path` 탐지를 거는 픽스처다.** 실제 적재 위치를 알리는 문장이 아니라
"이런 모양이 잡히는가"를 보는 합성 문자열이다. 게다가 탐지기
(`screen_public_release.LOCAL_PATH`)의 정규식 자체가 `<저장소 이름>` 을 대안 중 하나로 들고 있어야
동작하므로, 이 토큰은 어차피 소스에 남는다. **바꾸지 않았다.**

## C-3. 확인

B 소관 전 경로(`corpus/` · `tests/corpus/` · B 보고서 전부)를 훑어 남은 것은 위 픽스처
한 줄뿐이다. 새 lint 0(파일별로 `git show HEAD` 판본과 대조), `tests/corpus` 510 통과,
전체 **1,444 통과 · 17 skip**. skip 17 은 내 변경 탓이 아니다 — 파일럿 산출물 부재로 이미
건너뛰던 것 13건과, `pairs_pilot_v2` 소실을 정직하게 보고하는 내 시험 2건, 워크트리에 추적
밖 구성원이 없어 생기는 `incomplete_tree` 2건이다.

---

# 부록 D. 감사 신규 시험 수리 — 그리고 왜 내가 못 봤는가

## D-1. 무엇이 깨졌나

main `035410c` 가 외부 감사 패치를 커밋하면서 `tests/corpus/test_audit_acceptance.py` 가
들어왔다. 그중 `test_human_sheet_omits_judge_information_and_block_order` 가 깨진다.

```python
sheet = judge_labels.build_sheet(recs, cfg, "deepseek")          # ← (시트, 메타) 튜플이 온다
assert "judge_pass" not in json.dumps(sheet) ...                 # ← 메타까지 직렬화된다
```

내 `e697a49` 가 `build_sheet` 의 반환을 `(시트, 메타)` 로 바꿨는데 그 시험은 옛 서명대로
결과 하나를 받아 통째로 `json.dumps` 한다. 메타의 층 이름
(`조항검색_기준서술|judge_pass`)이 직렬화에 섞여 단언이 걸린다.

**코드가 아니라 시험이 서명 변경을 못 따라간 것이다.** 시트 자체는 깨끗하고(누설 0),
메타에 층 집계가 남는 것이 F14-P1 설계다. 총괄 판정도 같다.

## D-2. 어떻게 고쳤나

튜플을 풀어 **시트에만** 블라인드 단언을 걸고, **메타에는 층 집계가 있어야 한다**는 단언을
덧붙였다.

```python
sheet, meta = judge_labels.build_sheet(recs, cfg, "deepseek")
blob = json.dumps(sheet, ensure_ascii=False)        # 시트만
assert "judge_pass" not in blob and "judge_fail" not in blob
...
# 층 기록은 지워지지 않았다 (F14-P1)
assert set(meta["cells"]) == {"|".join(g) for g in set(groups)}
for cell in meta["cells"].values():
    assert {"N_population", "k_drawn", "target", "shortfall", "p_sampling"} <= set(cell)
    assert cell["k_drawn"] <= cell["N_population"]
assert set(meta["stratum_of"]) == {r["sample_id"] for r in sheet}
```

뒤쪽 단언을 덧붙인 이유는 총괄이 짚은 그대로다 — 없으면 "시트를 비웠다"만 지켜지고
**메타를 통째로 없애도 시험이 통과한다.** 그러면 F14-P1 설계가 시험으로 보호되지 않는다.
이빨 확인: `cells` 를 빈 사전으로 바꾸면 그 단언이 실제로 실패한다.

## D-3. **왜 내가 못 봤는가** — 미추적 파일은 트랙에 안 보인다

29번 검수 때(09-11) 나는 이렇게 적었다.

> **F13-P6. 시험이 없다.** diff 의 `tests/` 변경은 F14 뿐이다.

**반은 틀렸다.** (09-15 정정 — 처음엔 "두 파일이 있었다" 고 적었는데 과장이었다.)
감사 패치에는 `test_audit_acceptance.py`(채택 연결 시험)가 **있었다.** 그러나
`test_audit_snapshot_acceptance.py` 는 **없었다** — main 작업 트리의 그 파일은 09-11 19:49:57 에
생겼고(mtime 19:52:10), `monitor_2026-09-11.md` 의 "19:52 KST — F13 추가 적용" 이 그것이다.
내 검수(15:33–15:43)보다 뒤이고, 내 F13-P1 을 받아 쓰인 것이다. 즉 "스냅샷 소비자 시험이 없다"
는 당시 맞는 관찰이었고, 틀린 것은 채택 연결 시험 하나를 못 본 것이다. 둘 다 `035410c`
(09-13 17:57)에 커밋됐다. 못 본 이유는 둘이다.

1. **내가 받은 diff 에 없었다.** `full.diff` 의 `tests/` 항목은 `test_canonical_wiring.py`
   하나뿐이었다. `test_audit_acceptance.py` 는 **신규 미추적 파일**이라 `git diff`(HEAD 대비)에
   잡히지 않는다. `git status` 의 `??` 로만 보이고, 그 `??` 목록은 main 작업 트리에서만 보인다.
2. **내 워크트리에 파일이 없었다.** 나는 `c516387` 을 임시 트리에 떠서 검수했다. 미추적
   파일은 커밋에 없으므로 그 트리에 존재하지 않았다. 내가 "없다"고 본 것은 **정확한 관찰**
   이었고, 관찰 범위가 틀렸다.

그래서 나는 **있는 시험을 없다고 보고했고**, 뒤이어 그 시험과 충돌하는 서명 변경을 했다.

### 같은 구조로 두 번 물렸다

| | 09-11 (F13-P6) | 09-13 (이번) |
|---|---|---|
| 내가 본 것 | `tests/` 변경이 F14 뿐 | 서명 바꿔도 걸리는 시험 없음 |
| 실제 | 채택 연결 시험 1파일이 미추적으로 존재 (스냅샷 시험은 그날 저녁 19:52 에 생김) | 그 시험이 옛 서명을 전제 |
| 원인 | **미추적 파일이 diff·워크트리 어디에도 안 나타난다** | 같음 |

이것은 앞서 `data/processed` 사고와도 같은 계열이다. **내 도구가 못 보는 것을 "없다"고
읽었다.** 24번 A-2 에서 `verify_contract` 가 계약서 없는 디렉터리를 `no_contract` 로 조용히
넘긴 것, `frozen_out` 이 5개가 사라졌는데 "깨짐 0" 을 낸 것과 같은 모양이다. 부재를
정상으로 읽는 검사는 부재를 못 잡는다.

## D-4. 재발 방지 제안 추가 (§8 에 더한다, 실행 안 함)

**8-6. 패치 검수는 `git status` 를 함께 받는다.** diff 만으로는 신규 미추적 파일을 볼 수
없다. 검수 요청 시 `git status --porcelain` 출력이나 `git add -A` 후의 diff 를 함께 주면
된다. 어느 쪽이든 **"diff 에 없다 = 존재하지 않는다" 라는 추론을 끊는 것**이 요점이다.

**8-7. 검수 트리를 커밋에서 뜨면 미추적분이 빠진다는 것을 절차에 명시한다.** 임시 워크트리
검수는 재현성이 좋은 대신 미추적 파일을 구조적으로 잃는다. 그 한계를 검수 보고서에 한 줄로
적게 한다 — 이번 29번에 그 한 줄이 있었다면 "시험이 없다" 가 "받은 범위에는 시험이 없다"
로 쓰였을 것이고, 총괄이 그 차이를 바로 짚었을 것이다.

**8-8. 공개 인터페이스 서명을 바꾸면 호출자를 저장소 전체에서 찾는다.** 이번에
`build_sheet` 의 반환을 바꾸면서 내 워크트리 안의 호출자 둘만 고쳤다. `grep -rn "build_sheet"`
를 **main 기준으로** 돌렸다면 `test_audit_acceptance.py` 가 나왔다. 트랙 워크트리는 main 의
최신을 항상 담고 있지 않으므로, 서명 변경 전에 `git grep <이름> main` 을 한 번 거는 것이
싸다.

## D-5. 회귀 확인

```
1,576 수집 · 1,558 통과 · 18 skip · 0 실패   (3분 15초)
```

skip 18 = 파일럿 산출물(`outputs/pilot_c`·`pilot_d`) 부재로 이미 건너뛰던 14건(D 소관) +
`pairs_pilot_v2` 소실·`incomplete_tree` 를 정직하게 보고하는 내 시험 4건.

> 총괄이 말한 **1,549 통과와 9건 차이**가 난다. 내 쪽이 수집 2건 많고(내 `e697a49` 가
> `test_canonical_wiring.py` 에 더한 시험 2개 — main 22 → 24) skip 이 7건 적다. skip 은
> 환경 의존이라(파일럿 산출물·봉인본 유무) 트리마다 갈린다. **실패 0 은 양쪽이 같다.**
> 숫자를 맞추려고 손대지 않았다 — 차이의 출처를 적는 편이 낫다.

충돌 해소는 두 파일 다 내 판이다. merge-base 대비 main 쪽 변경이 Codex 패치 그것뿐이고
내 판이 상위집합인 것을 diff 로 확인했다.

> `git merge main --ff-only` 는 쓸 수 없었다 — `wt/B` 가 main 에 없는 커밋 4개를 들고 있어
> 갈라져 있다(내 커밋은 아직 main 에 없다). rebase 대신 머지 커밋을 택했다. rebase 는
> 총괄이 계속 인용해 온 커밋 해시를 바꾼다.

---

# 부록 E. `cycle_pilot` 추적 구성원 3개의 줄바꿈 — F 31번 확인 (09-15)

## E-1. 사실이다. 어느 3개인지, 어디서 갈리는지

`corpus/generate/cycle_pilot` 계약 구성원 8개 중 git 이 추적하는 3개가 전부 걸린다.
`cycle_pilot_v2` 의 추적 구성원 1개와 A 의 `data/mock/*` 2곳(추적 3+3)은 **문제없다.**

| 구성원 | 계약서 | `wt_B` 작업 파일 | git blob (`cat-file -p`) | blob 을 CRLF 로 | CR 수 |
|---|---|---|---|---|---|
| `_phi_report.json` | `fb1eef4a…` | 일치 (CRLF) | `1609571d…` 불일치 (LF) | **일치** | 49 |
| `cycle_corpus_report.json` | `381d4351…` | 일치 (CRLF) | `8cacb1a9…` 불일치 (LF) | **일치** | 50 |
| `judge_agreement.json` | `7e19ef0a…` | 일치 (CRLF) | `7f7db7d0…` 불일치 (LF) | **일치** | 33 |

`main` 체크아웃의 작업 파일은 blob 과 같은 LF 라 **3/3 불일치**다. 이력 전 구간에서 세 파일의
blob 은 처음 커밋(`e259989`·`dfd6954`·`92eefae`, 08-24)부터 CR 0바이트다. `.gitattributes`
(`* text=auto eol=lf`)는 초기 공개 커밋(08-21)부터 있었다.

**경위.** 세 파일은 08-24 에 `write_text` 로 쓰였다 — 09-01 `newline=""` 수정 **전** 코드라
디스크에는 CRLF 로 떨어졌다. `git add` 가 속성대로 LF 로 정규화해 저장했고, 작업 파일은 그대로
CRLF 로 남았다. 09-01·09-02 봉인은 **작업 파일 바이트**를 해시했다. 즉 계약서는 git 이 한 번도
저장한 적 없는 바이트 형태를 적었다. `git status` 는 wt_B 를 깨끗하다고 본다 — 색인에 넣을 때
같은 정규화를 거쳐 blob 과 같아지기 때문이다. **wt_B 에서만 계약이 맞는 것은 이 트리가 그
파일들을 다시 체크아웃한 적이 없어서다.** 다른 어떤 체크아웃에서도 안 맞는다.

**같은 사각이다.** 내가 `snapshot_cycle_pilot --check` 를 돌린 자리는 늘 wt_B 였고, 그래서 늘
초록이었다. 새 clone 에서 `--check` 를 한 번만 돌렸어도 09-02 에 잡혔다 — 부록 D 의 "내
트리가 보는 것을 전부라고 읽었다" 와 같은 모양이다.

## E-2. 해결안 — 제안만 (봉인 파일·blob 은 건드리지 않았다)

버리는 저장소 둘을 만들어 A·C 를 실측했다. 각각 `autocrlf=true`·`false` 로 새로 clone 해 작업
파일 해시를 계약과 대조하고, LF 로 남은 기존 트리(main 과 같은 상태)에서 `git status` 를 봤다.

**A. `.gitattributes` 에 세 경로만 `text eol=crlf` — 권고.**

```
corpus/generate/cycle_pilot/_phi_report.json          text eol=crlf
corpus/generate/cycle_pilot/cycle_corpus_report.json  text eol=crlf
corpus/generate/cycle_pilot/judge_agreement.json      text eol=crlf
```

- 실측: 새 clone 작업 파일 → 계약 일치 **3/3** (`autocrlf` 양쪽 모두).
- **blob 을 바꾸지 않는다.** 저장은 LF 그대로, 체크아웃 때만 CRLF 로 편다. 이력 영향 0.
- 영향 범위: 속성이 바뀐 뒤 **이미 LF 로 체크아웃된 트리**(main·다른 워크트리)는 세 파일이
  `M` 으로 뜬다(실측). `git checkout -- corpus/generate/cycle_pilot/` 로 작업 파일만 다시
  펴면 사라진다 — blob 쓰기가 아니라 체크아웃이다. wt_B 는 이미 CRLF 라 아무 일도 없다.
- 남는 것: blob 을 직접 읽는 도구(`git show`, GitHub raw)는 LF 를 본다. 그 해시는 계약과
  다르다. **"계약 바이트는 체크아웃 결과" 라고 어딘가에 적어야 한다** — 이 부록이 그 기록이고,
  `frozen_out` 명부 항목에 한 줄 더 두는 것을 제안한다.
- `.gitattributes` 는 A 소관(10_spec_A §6-9)이다. **배분은 총괄.**

**B. 대조기가 줄바꿈을 정규화 — 통과 기준으로는 기각, 진단으로는 채택 제안.**

통과 기준으로 삼으면 봉인이 약해진다. `sha256sum -c` 같은 외부 대조는 여전히 실패하고, 논문에
실은 해시가 "어느 바이트의 해시인지" 가 대조기 구현에 매이게 된다. 대신 **불일치가 났을 때만**
CRLF↔LF 정규화 해시를 추가로 재 보고 맞으면 `eol_mismatch` 로 이유를 붙이는 진단은 값이 있다 —
F 가 손으로 알아낸 것을 도구가 말해 준다. 자리는 구성원을 실제로 해시하는
`snapshot_cycle_pilot --check` 와 `verify_backup verify` 다(`verify_contract` 는 존재·추적만 본다).
이번에 넣지 않았다 — (2) 는 제안만 하라는 지시였다.

**C. 세 경로에 `-text` 를 주고 blob 을 CRLF 바이트로 다시 커밋 — 기각.**

- 실측: blob 이 계약과 **바이트 동일**, 새 clone 3/3 (양쪽 `autocrlf`). 결과만 보면 가장 정확하다.
- 그러나 **봉인 구성원 3개의 blob 을 새로 쓰는 커밋**이 생긴다. `git log -- <파일>` 에 봉인 뒤
  수정 이력이 남고, GitHub 에서는 49·50·33줄 전체가 바뀐 diff 로 보인다. 그 커밋 이전 어느
  시점을 체크아웃해도 다시 불일치다. 이력 재작성은 아니지만 **"봉인된 파일을 커밋으로 건드렸다"**
  는 사실이 영구히 남고, A 로 같은 체크아웃 결과를 blob 변경 0 으로 얻을 수 있으므로 택할
  이유가 없다.

**D. 재봉인 — 지시대로 하지 않는다.** 규약 1-6 이고, 논문에 실을 digest `fb316682…` 가 바뀐다.

## E-3. 재발 방지 (§8 에 더한다)

**8-9. 봉인 해시는 git 이 저장하는 바이트로 잰다, 아니면 새 clone 에서 대조한다.** 작업 파일을
해시하면 그 트리에서만 맞는 계약이 나올 수 있다. `snapshot_cycle_pilot` 이 추적 구성원은
`git show HEAD:<path>` 바이트를 해시하거나, 봉인 직후 새 clone 에서 `--check` 를 한 번 돌리는
것을 봉인 절차에 넣는다. 둘 중 뒤쪽이 더 정직하다 — 도구가 아니라 다른 트리가 확인한다.

---

# 부록 F. `verify_contract` — 계약서 부재를 조용히 넘기던 구멍 수리 (§8-4 나)

09-13 승인분. 계약서가 있는 디렉터리**만** 찾으면 디렉터리째 사라진 봉인본은 목록에서 같이
사라진다. 09-11 에 `data/processed` 5곳이 비었을 때 이 도구가 "깨짐 0" 을 낸 이유다.

## F-1. 무엇을 바꿨나

`corpus/generate/frozen_out.py` 에 **있어야 할 봉인처 명부** `EXPECTED_SEALED` 를 두고,
`verify_contract` 가 명부의 기대를 겹쳐 판정한다. 옛 판정 넷은 `_verify_present` 로 그대로 남겼다.

| 새 판정 | 뜻 | 실패? |
|---|---|---|
| `missing_contract` | 있어야 하는데 계약서가 없다 — 저장 뿌리가 이 트리에 있거나 git 이 계약서를 추적하는데도 | **실패** |
| `absent_tree` | 저장 뿌리 자체가 이 트리에 없다 (정션 안 붙인 새 clone 등). 소실인지 판단 못 함 | 알림 |
| `lost_recorded` | 소실이 기록된 자리이고 실제로 없다 | 알림 — **매번** |
| `lost_but_present` | 소실 기록된 이름에 계약서가 있다 — 명부가 낡았거나 같은 이름 재생성(규약 1-6 위반) | **실패** |

기록된 소실을 매번 실패로 만들면 사람들은 이 검사를 끈다. 그래서 실패 대신 **매번 말한다.**
명부 항목에 `owner` 와 `record`(소실·복원 기록 한 줄)를 달아 출력에 같이 찍는다.

CLI (`python -m corpus.generate.frozen_out`)는 인자가 없으면 **명부를 먼저** 대조하고, 명부 밖에서
계약서가 발견된 곳을 더한다. 실물 트리 실측:

```
○ corpus/generate/cycle_pilot                              구성원 8  [ok]  (소관 B)
○ corpus/generate/cycle_pilot_v2                           구성원 6  [ok]  (소관 B)
○ data/processed/pairs_pilot_v1                            구성원 3  [ok]  (소관 B)
! data/processed/pairs_pilot_v2                            구성원 0  [lost_recorded]  (소관 B)
    09-11 소실 확정 · 사본 없음 · 입력 부재로 재생성 불가 (30번 부록 A-3)
! data/processed/aihub71761_rt_v1_pilot3000                구성원 0  [lost_recorded]  (소관 A)
! data/processed/aihub71761_rt_v1_pilot3000_crop_only      구성원 0  [lost_recorded]  (소관 A)
! data/processed/aihub71761_rt_v1_pilot3000_scale_control  구성원 0  [lost_recorded]  (소관 A)
봉인처 7곳 · 깨짐 0 · 소실 기록 4 · 이 트리에 없음 0
```

**같은 트리, 명부를 비우면:** exit 0, 소실 언급 0, `pairs_pilot_v2` 라는 글자 자체가 안 나온다.
이것이 09-11 의 출력이었다.

## F-2. 시험 7건 (`tests/corpus/test_frozen_out.py`)

이빨을 둘 걸었다. ① 저장 뿌리는 있는데 봉인 디렉터리가 계약서째 없는 가짜 트리 → `missing_contract`,
exit 1. ② **같은 트리에서 명부를 비우면** exit 0, "깨짐 0" — 즉 ①의 실패는 명부 때문이다.
그 밖에: 뿌리 없는 트리는 알림만 · git 이 계약서를 추적하면 뿌리가 없어도 실패 · 소실 기록은
알리되 실패 아님 · 소실 기록된 이름이 되살아나면 실패 · 명부가 실물 봉인처를 전부 덮는지 ·
실물 트리 출력에 `pairs_pilot_v2` 소실이 매번 나오는지.

`test_봉인_구성원이_실물과_이름이_맞는다` 도 바꿨다 — 디렉터리가 없으면 그냥 건너뛰던 것을
명부 판정으로 바꿨고, 명부에 없는 봉인처는 **시험이 실패**한다.

## F-3. 짚을 것

- 명부는 B 파일 안에 있는데 **A 소관 자산 3개**(pilot3000 계열)를 `lost` 로 들고 있다. A 가
  처리를 판정하면 그 항목을 A 가 고쳐야 하는데 파일은 B 것이다. 명부를 `configs/` 같은 공용
  자리로 옮기는 것이 맞아 보인다 — **배분은 총괄.**
- 부록 E 의 줄바꿈 표기(A 안 채택 시 "계약 바이트는 체크아웃 결과")도 이 명부 항목에 `note`
  로 붙이는 자리가 있다.

---

# 부록 G. 위생 2차 — 부록 C 스윕이 놓친 것 (09-15)

총괄이 30번 2건을 짚었고, 같은 정규식 문제로 놓친 것을 더 찾아 **추적 파일 3개 6건 + 쓰는 코드**
를 고쳤다.

| 파일 | 건수 | 무엇 |
|---|---|---|
| `30_processed_소실_경위_B.md` | 3 | §1 ② mklink 원문 1 · 부록 C 산문 1 · 부록 C-2 코드 인용 1 |
| `corpus/validate/verify_backup.py` | 1 | 모듈 문서의 본체 경로 |
| `corpus/parse/extracted_manifest.json` | 2 | `src`·`dst` — 공유 드라이브 뿌리 + **학회 폴더명** (F 31번 §1-5 가 같은 것을 지목) |

**왜 놓쳤나.** 부록 C 의 정규식이 역슬래시 **하나**만 잡았다. Python 소스·JSON 은 역슬래시를
두 개로 이스케이프하므로 그 꼴은 빠졌다. 30번의 셋은 부록 C 가 원문을 **인용**하면서 다시
들여온 것이다 — 위생 보고서가 위생 위반을 재생산했다.

명령 원문을 보존하는 §1 은 지시대로 **경로만 `<저장소 루트>` 로** 바꾸고 명령 형태는 그대로
뒀다. 같은 절의 `<REPO>` 둘도 한 절 안에서 자리표시자를 하나로 맞췄고, 미추적 경로표에 별칭을
한 줄 더했다.

`extracted_manifest.json` 은 값을 폴더 이름만(`corpus_candidate`·`corpus_extracted`) 남기고
"실제 경로는 기록하지 않는다" 를 `paths` 키로 적었다. **쓰는 쪽도 같이 고쳤다** —
`extract_candidates.py` 가 다음 실행에서 `SRC.name` 만 적고, `newline=""` 로 연다. 안 고치면
다음 실행이 실제 경로를 도로 박는다. 소비자는 `docs/corpus_후보_판정.md:293` 의 언급뿐이라
`src`·`dst` 값을 읽는 코드는 없다.

넓힌 정규식(이스케이프된 역슬래시·학회 폴더명·`AppData`)으로 B 소관 추적 파일을 다시 훑어 남은
것은 `test_release_and_backup.py:55` 의 의도적 픽스처 한 줄뿐이다.

## 회귀

`tests/corpus` 547 통과 · 4 skip. 전체 **1,566 통과 · 18 skip · 0 실패**. skip 18 은 전과 같다
(파일럿 산출물 부재 14 · 소실 기록 2 · `incomplete_tree` 2). 봉인 파일·blob 은 쓰지 않았다 —
실측은 전부 버리는 저장소와 읽기로 했다. 워크트리 삭제 동결 유지.
