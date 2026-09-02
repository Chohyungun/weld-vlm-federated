"""재실행 전 인용 금지 표식 — 배치와 검사. 80번 G11-4.

파일럿 산출물 일부는 **고장이 고쳐지기 전 코드의 것**이라 논문·보고서에 수치를 인용하면
안 된다. 80번은 그 목록을 정하고 "디렉터리에 `DO_NOT_CITE.md` 를 두고 채점·인용 스크립트가
그 파일을 보면 실패"하게 하라고 했다.

    uv run python scripts/citation_ban.py --write     # 표식 배치·갱신
    uv run python scripts/citation_ban.py             # 표식이 제자리에 있는지 검사
    uv run python scripts/citation_ban.py --check <경로> [<경로> ...]   # 인용 가부 판정

## 왜 디렉터리 표식만으로는 안 되나

`sep_fed` 는 **논문 본체인 분리형 칸**이다. 금지 대상은 회계 두 파일(`accounting.csv` ·
`audit.json`)뿐이고 가중치·예측은 멀쩡하다. `predictions/` 도 마찬가지로 `uni_*` 두 개만
금지고 `sep_*` 검출 예측은 쓸 수 있다. 디렉터리를 통째로 막으면 **써도 되는 것까지
막혀서** 다음 사람이 표식을 지운다. 그래서 표식은 디렉터리에 두되 **파일 단위 범위**를
기계가 읽을 수 있게 담는다.

## 왜 이 스펙이 스크립트 안에 있나

`outputs/` 는 `.gitignore` 대상이고 본체와 정션으로 공유된다. 표식 파일 자체는 커밋되지
않으므로, 표식이 지워지면 근거가 사라진다. **스펙을 커밋되는 스크립트에 두고 표식을
생성하게** 하면 언제든 같은 내용으로 복원된다.

배선(채점·인용 스크립트가 이 검사를 호출하는 것)은 C·D 소관이다. A 는 표식과 검사기까지다.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER = "DO_NOT_CITE.md"


@dataclass(frozen=True)
class Ban:
    """한 디렉터리의 인용 금지 범위와 사유."""

    directory: str
    banned: tuple[str, ...]
    reason_id: str
    reason: str
    evidence: tuple[str, ...]
    rerun_condition: str
    allowed: tuple[str, ...] = field(default_factory=tuple)
    allowed_note: str = ""


#: 금지 목록. 80번 G11-4 + §7 "착수 전 반드시 확정할 부수 사항" 이 정본이다.
#: 증거는 2026-09-02 트랙 A 가 산출물에서 직접 재확인했다.
BANS: tuple[Ban, ...] = (
    Ban(
        directory="outputs/pilot_c/uni_central",
        banned=("*",),
        reason_id="C3 · E3",
        reason=(
            "LoRA A 초기화 시드 고정(fec53f7) **이전** 코드로 만든 어댑터다. "
            "공통 초기 어댑터 파일(adapter_initial.npz)이 없어 ⑥과 ⑦이 같은 난수에서 "
            "출발했다는 증거 자체가 없다 — 66번의 통합형 유지율 73.5%는 공통 출발점을 "
            "공유하지 않은 두 칸의 비다."
        ),
        evidence=(
            ("adapter_last.meta.json 에 init_proof·epochs_ran·optimizer·lr·init_seed 전무 "
             "(현재 필드는 optimizer_steps·supervised_tokens·peak_vram_gb·payload_bytes·"
             "param_l2·wall_s·seed 일곱 개뿐)"),
            "adapter_initial.npz 부재",
        ),
        rerun_condition=(
            "체크리스트 1(좌표 규약 ABS_ORIG)·4(전역 오프셋 cosine)·5(손실 정규화)가 닫히고 "
            "[E] 배경지식 단계를 거친 뒤 재학습. 재실행본은 init_proof 를 반드시 남긴다."
        ),
    ),
    Ban(
        directory="outputs/pilot_c/uni_fed",
        banned=("*",),
        reason_id="C3 · E3 · F8",
        reason=(
            "위와 같은 시드 미고정 런이고, r0 에서 세 클라이언트가 각자 난수 A 로 출발해 "
            "가중 평균에서 **상쇄**됐다. r0 로컬 학습분 144스텝(전체 432의 33%)이 사실상 "
            "폐기됐고 그 상태가 r1·r2 로 이월된다. 회계 CSV 도 fec53f7 이전 스키마다."
        ),
        evidence=(
            ("atomic_log.csv r0: 클라이언트 param_l2 31.7405 인데 집계 직후 "
             "global_l2 20.5755 — 독립 난수 가중합의 지문(공유 초기값이면 ≈31.7)"),
            "이월: r1 20.7508 · r2 20.8797",
            ("accounting.csv 에 stopper_class·stopper_calls·value_source·"
             "optimizer_updates·resumed_from_epoch 다섯 열 없음"),
        ),
        rerun_condition=(
            "체크리스트 15(⑦을 Flower 경로로)·16(무이빨 가드 셋)·2(FedAvg 가중 단위 = "
            "감독 토큰 총합)가 닫힌 뒤 재실행. 집계 후 norm 이 20.6 대가 아니라 31.7 대로 "
            "나오는지가 수정 확인 지표다."
        ),
    ),
    Ban(
        directory="outputs/pilot_c/predictions",
        banned=("uni_central.generations.jsonl", "uni_fed.generations.jsonl"),
        allowed=("sep_central.detections.jsonl", "sep_fed.detections.jsonl",
                 "sep_local_c0.detections.jsonl", "sep_local_c1.detections.jsonl",
                 "sep_local_c2.detections.jsonl"),
        allowed_note=(
            "검출 예측 5종은 금지 대상이 아니다. 분리형 가중치는 시드 문제와 무관하다."
        ),
        reason_id="C3 파생",
        reason=(
            "위 두 디렉터리의 오염된 어댑터에서 뽑은 생성문이다(653×2행). "
            "생성 코드가 아니라 **입력 가중치**가 문제이므로 재생성 외에는 방법이 없다."
        ),
        evidence=("uni_central·uni_fed 어댑터 파생물",),
        rerun_condition="위 두 어댑터가 재학습된 뒤 다시 export.",
    ),
    Ban(
        directory="outputs/pilot_c/sep_fed",
        banned=("accounting.csv", "audit.json"),
        allowed=("global_r001.npz", "global_r002.npz", "global_r003.npz", "latest.npz",
                 "atomic_log.csv", "accounting.json", "runs"),
        allowed_note=(
            "**가중치·예측·atomic_log 는 금지 대상이 아니다.** sep_fed 는 논문 본체인 "
            "분리형 칸이고 LoRA 시드 문제와 무관하다. 막히는 것은 회계 두 파일뿐이다."
        ),
        reason_id="F7 · F8",
        reason=(
            "회계 산출물이 fec53f7 이전 스키마다. 조기 종료 부재를 증명해야 할 검사가 "
            "**구조적으로 실패할 수 없는 상태**에서 만들어졌다 — stopper_true_count 가 "
            "리터럴 0 이고, 재구성값임을 표시할 value_source 열 자체가 없다. "
            "조기 종료 부재의 실질 증거는 이 회계가 아니라 results.csv 행 수와 "
            "optimizer_steps 쪽에 있다(그쪽으로는 확인됨)."
        ),
        evidence=(
            "accounting.csv 9행 전부 stopper_true_count=0 (리터럴)",
            ("accounting.csv 에 stopper_class·stopper_calls·value_source·"
             "optimizer_updates·resumed_from_epoch 다섯 열 없음"),
            ("audit.json 이 ok·failures·total_epochs_by_client·total_optimizer_steps "
             "네 필드뿐"),
        ),
        rerun_condition=(
            "체크리스트 16(무이빨 가드 셋)·14(회계 finally + audit.json)가 닫힌 뒤 "
            "회계를 다시 생성. 가중치 재학습은 필요 없다."
        ),
    ),
)


def render(ban: Ban) -> str:
    lines = [
        "# 인용 금지 — 재실행 전까지",
        "",
        "**이 표식이 있는 동안 아래 파일의 수치를 논문·보고서·발표에 인용하지 마라.**",
        "",
        (f"근거: 80번 방법론 재검증 G11-4 · 결함 {ban.reason_id}. "
         "배치: 2026-09-02 트랙 A (지시 `dispatch_A_위생.md` 1항)."),
        "",
        "## 금지 대상",
        "",
    ]
    lines += [f"- `{p}`" for p in ban.banned]
    if ban.allowed:
        lines += ["", "## 금지 대상이 **아닌** 것 (같은 디렉터리)", ""]
        lines += [f"- `{p}`" for p in ban.allowed]
        if ban.allowed_note:
            lines += ["", ban.allowed_note]
    lines += ["", "## 왜", "", ban.reason, "", "## 실측 근거", ""]
    lines += [f"- {e}" for e in ban.evidence]
    lines += [
        "", "## 언제 풀리나", "", ban.rerun_condition, "",
        "재실행이 끝나면 이 파일을 지운다. **먼저 지우지 마라** — 표식이 없으면 다음 사람은",
        "이 산출물이 멀쩡하다고 읽는다.",
        "",
        "## 기계 판독용",
        "",
        "```json",
        json.dumps({
            "spec": "do-not-cite/1",
            "directory": ban.directory,
            "banned": list(ban.banned),
            "allowed": list(ban.allowed),
            "reason_id": ban.reason_id,
        }, ensure_ascii=False, indent=2),
        "```",
        "",
        "표식 생성·검사는 `scripts/citation_ban.py` 다. 스펙이 그 스크립트에 있으므로",
        "이 파일이 지워져도 `--write` 로 복원된다 (`outputs/` 는 git 추적 대상이 아니다).",
        "",
    ]
    return "\n".join(lines)


def parse_marker(path: Path) -> dict | None:
    """표식에서 기계 판독 블록을 꺼낸다."""
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start < 0:
        return None
    end = text.find("```", start + 7)
    if end < 0:
        return None
    try:
        return json.loads(text[start + 7:end])
    except json.JSONDecodeError:
        return None


def is_banned(target: Path, root: Path = REPO_ROOT) -> tuple[bool, str]:
    """경로 하나가 인용 금지인가. 표식을 **디스크에서 읽어** 판단한다.

    스펙 상수가 아니라 표식을 읽는 이유는, 표식이 지워졌으면 그 사실이 드러나야 하기
    때문이다. 상수만 보면 표식을 지워도 검사가 통과한다고 착각하게 된다.
    """
    p = Path(target).resolve()
    for parent in [p, *p.parents]:
        marker = parent / MARKER
        if not marker.is_file():
            continue
        spec = parse_marker(marker)
        if spec is None:
            return True, f"{marker} 를 읽을 수 없다 — 안전하게 금지로 본다"
        rel = p.relative_to(parent).as_posix() if p != parent else "*"
        for pat in spec.get("allowed", []):
            if rel == pat or rel.startswith(f"{pat}/") or fnmatch.fnmatch(rel, pat):
                return False, ""
        for pat in spec.get("banned", []):
            if pat == "*" or rel == pat or fnmatch.fnmatch(rel, pat):
                # `outputs/` 는 본체를 가리키는 정션이라 resolve 하면 리포 밖 경로가 된다.
                # 상대 경로화가 실패해도 **사유는 반드시 보여야** 하므로 스펙의 값을 쓴다.
                where = spec.get("directory") or str(marker.parent)
                return True, f"{spec.get('reason_id', '?')} · {where}/{MARKER}"
        return False, ""
    return False, ""


def cmd_write() -> int:
    n = 0
    for ban in BANS:
        d = REPO_ROOT / ban.directory
        if not d.is_dir():
            print(f"  ! {ban.directory} 가 없다 — 건너뛴다")
            continue
        (d / MARKER).write_text(render(ban), encoding="utf-8", newline="\n")
        print(f"  기록 {ban.directory}/{MARKER}  (금지 {len(ban.banned)}종 · "
              f"허용 명시 {len(ban.allowed)}종)")
        n += 1
    print(f"표식 {n}개 배치")
    return 0


def cmd_verify() -> int:
    ok = True
    for ban in BANS:
        d = REPO_ROOT / ban.directory
        marker = d / MARKER
        if not d.is_dir():
            print(f"  [건너뜀] {ban.directory} 없음")
            continue
        if not marker.is_file():
            print(f"  [실패] {ban.directory}/{MARKER} 이 없다")
            ok = False
            continue
        spec = parse_marker(marker)
        if spec is None or list(spec.get("banned", [])) != list(ban.banned):
            print(f"  [실패] {ban.directory}/{MARKER} 의 범위가 스펙과 다르다")
            ok = False
            continue
        # 금지 대상이 실제로 막히고 허용 대상이 실제로 통과하는지 — 이빨 확인
        probe = ban.banned[0] if ban.banned[0] != "*" else "adapter_last.npz"
        banned_ok, _ = is_banned(d / probe)
        allow_ok = True
        if ban.allowed:
            allow_ok = not is_banned(d / ban.allowed[0])[0]
        mark = "통과" if (banned_ok and allow_ok) else "실패"
        ok &= banned_ok and allow_ok
        print(f"  [{mark}] {ban.directory}  금지 {len(ban.banned)}종 · 허용 {len(ban.allowed)}종")
    print("표식 검사 " + ("전부 통과" if ok else "**실패**"))
    return 0 if ok else 1


def cmd_check(paths: list[str]) -> int:
    bad = []
    for t in paths:
        banned, why = is_banned(Path(t))
        print(f"  [{'금지' if banned else '가능'}] {t}" + (f"   ← {why}" if banned else ""))
        if banned:
            bad.append(t)
    if bad:
        print(f"\n인용 금지 산출물 {len(bad)}건이 포함됐다. 재실행 전까지 수치를 쓰지 마라.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="표식을 배치·갱신한다")
    ap.add_argument("--check", nargs="+", metavar="PATH",
                    help="주어진 경로들의 인용 가부를 판정한다 (금지가 있으면 종료 1)")
    args = ap.parse_args()
    if args.write:
        return cmd_write()
    if args.check:
        return cmd_check(args.check)
    return cmd_verify()


if __name__ == "__main__":
    raise SystemExit(main())
