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
mkdir -p data && cmd //c "mklink /J \"$(cygpath -w "$W/data/processed")\" \"E:\\Fedvlm_for_welding\\data\\processed\""
```

→ `파일 이름, 디렉터리 이름 또는 볼륨 레이블 구문이 잘못되었습니다.` 따옴표·경로 변환 문제로
정션이 만들어지지 않았다. 이때 `tests/corpus` 가 1건 실패했고(`test_v1_은_덮어쓸_수_없다`),
**그 실패가 "정션이 없다"는 신호였는데 나는 그것을 환경 잡음으로 읽고 정션을 붙이는 쪽으로 갔다.**

**③ 정션 부착 — 성공** (PowerShell)

```powershell
if (-not (Test-Path "$W\data\processed")) {
  New-Item -ItemType Junction -Path "$W\data\processed" -Target "E:\Fedvlm_for_welding\data\processed"
}
```

**④ 패치 트리 `wt_base` 생성 + 정션** (PowerShell)

```powershell
git worktree add --detach $B c516387
New-Item -ItemType Directory -Force "$B\data"
New-Item -ItemType Junction -Path "$B\data\processed" -Target "E:\Fedvlm_for_welding\data\processed"
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
> Copy-Item -Recurse "C:\Users\my\AppData\Local\Temp\claude\E--Fedvlm-for-welding-wt-B\fbb61323-9b55-492a-b120-9508e343d363\scratchpad\live\pairs_v1" "G:\<보관처>\pairs_pilot_v1_구조"
> Copy-Item "C:\Users\my\AppData\Local\Temp\claude\E--Fedvlm-for-welding-wt-B\fbb61323-9b55-492a-b120-9508e343d363\scratchpad\backup_manifest.json" "G:\<보관처>\"
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
