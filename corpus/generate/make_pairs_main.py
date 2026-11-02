"""D4 이미지-판정문 페어 — 본실험. 동결 매니페스트 `manifest_v1` 의 train·val 에서 만든다.

**얇은 모듈이다.** 조립(`assemble_record`)·검증(`check_pair`)·빌드(`build_pairs`)·쓰기(`write_outputs`)는 전부
`make_pairs_pilot` 의 것을 그대로 부른다 — 판정 논리는 한 곳에 있어야 한다(06번 §2-나). 여기서 정하는 것은
입력 스냅샷, 경로 실재 검사, 그리고 **산출물이 스스로 말해야 하는 메타**뿐이다.

판정 축은 파일럿과 같다: 조항 검색 + 기준 서술, `verdict=null`(`clause_only`). 두께·화소당 실치수가 전량 결측이라
치수 기준 합부는 원리적으로 성립하지 않는다(함정 #10). 균열의 판정 축도 열지 않는다(09-21 판정 6).

eval 분할은 만들지 않는다(개발규약 1-4). 빌드는 train·val 의 `image_id`·`group_id` 가 eval 과 겹치지 않음을
먼저 단언하고, 산출 뒤에 페어의 id 집합으로 다시 세어 메타에 적는다.

`--out` 은 **필수**이고 봉인된 디렉터리에는 쓰지 않는다(계약 `SNAPSHOT.sha256` 으로 판정). 봉인은 검수 승인
**뒤에** `data/processed/pairs_main_v1/` 로 한다 — 그 전의 빌드는 전부 임시 경로다.
한 번 봉인하면 다시 만들지 않는다(규약 1-6). 결정이 바뀌면 `pairs_main_v2` 로 새 경로에 만든다.

**판 등록부 `EDITIONS`** — 판마다 표 · 서술 계약 · 메타의 상태 문장이 한 행이다. `--edition` 은 기본값 없는 필수 인자다.
1 판(`v1`)은 지금 값 그대로이고(1 판 다시 짓기의 메타 키가 같다), 2 판(`v2` — 부분 B, C 08 2 판 §3)은 신판 표와 2 판 서술
(`pairs_text_v2`)을 쓰고 메타에 신원 블록(`identity`)과 문장 계약 판(`text_contract`)을 더 싣는다. 신판 경로는 이 등록부 한 자리에만
둔다 — 산출물 검사 · 봉인 관문은 빌드 메타의 `limits_table.path` 로 여기서 판을 찾는다. 독립 대조기는 이 등록부를 읽지 않는다
(봉인 관문이 넘긴 `--limits` 를 직접 읽는다 — 221 M-01).

실행: python -m corpus.generate.make_pairs_main --edition v1 --out <임시 경로> --date 2026-09-23
산출: pairs.jsonl · discarded.jsonl · counts.json · PAIRS_META.json · SNAPSHOT.sha256 (전부 LF)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus.generate import make_pairs_pilot as P
from corpus.generate.frozen_out import assert_not_frozen

REPO = P.REPO
SNAP = REPO / "data/interim/manifest_v1"
LIMITS_CSV = P.LIMITS_CSV
ASSET = "d4_pairs_main"
VERSION = "v1"

#: 허용치 표의 상태 — 산출물이 들고 다닌다. 표를 검수·개정하면 이 문장과 표의 sha256 이 함께 바뀐다.
LIMITS_STATUS = ("파일럿 초판 · 사람 이중 검수 전. 공개 규정에서 1인이 전사했고 12행 전부 material=ST 다."
                 " 로더는 pilot 모드(검사 다섯 면제)로 읽는다")

#: 사람 검증의 상태 (06번 §7). 통과율은 **형식 통과율**이지 사실성 지표가 아니다.
HUMAN_VERIFICATION = {
    "limits_table_double_check": "미실시 — 12행의 사람 이중 검수는 팀 과제다(09-21 판정 3)",
    "skeleton_factuality": "사람이 본 적 없다 — 규칙 검증만(validated_by=rule)",
    "gold_sheet": "라벨 0건",
    "how_to_name": ("규칙으로 생성하고 형식·스키마를 규칙으로 검사한 조항 서술 페어."
                    " '검증된 판정 데이터' · '전문가 검수' · '합부 판정 학습' 이라고 부르지 않는다"),
}

#: 학습 답안 계약 (09-21 판정 1 — 팀 확인 항목). 페어는 골격과 서술을 둘 다 싣고, 타깃 선택은 학습기 몫이다.
ANSWER_CONTRACT = ("기본값은 구조화 JSON 만 학습한다(골격의 type·bbox_px·verdict·clauses)."
                   " target_text 는 실려 있지만 학습 타깃이 아니다. 팀이 뒤집으면 허용치 표의 값·부등호·구간·부기의"
                   " 변경이 곧 재학습 사유가 된다")

#: 분할이 상속하는 알려진 한계 (06번 §1-나, 09-21 판정 9). **학습 자료는 건드리지 않는다** — 닫는 자리는
#: 평가 목록이다(D). 화소 쌍둥이 수는 빌더가 세지 않는다(이미지를 전부 읽어야 한다) — 09-21 의 측정값이고 하한이다.
NEAR_DUPLICATE_NOTE = {
    "train_val_vs_eval_pixel_twins_lower_bound": 33,
    "measured_on": "2026-09-21",
    "method": ("256비트 지각 해시의 해밍거리 16 이하인 train·val↔eval 1,606쌍을 전 해상도 그레이로 대조 —"
               " 평균 절대차 < 2 이고 상관 > 0.99. 무작위 대조군 300쌍에서는 0. 거리 16 까지만 봤으므로 하한이다"),
    "status": "분할 쪽에서 정량화 중이다. 페어는 D1 분할을 그대로 상속한다",
}

#: 2 판(부분 B)의 상태 문장들(C 08 2 판 §3 의 표). 1 판 값은 위 상수 그대로다.
LIMITS_STATUS_V2 = ("2 판(부분 B) — 1 판의 행 · 값 · 구간 · 조항은 그대로 두고 측정량 · 집계 · 종별 · 조건 칸을 더했다."
                    " 사람 이중 검수 · 용접 검사 전문가 검수 전이다. 재수록 수치 사용은 연구 책임자가 결정했다(전문가 검수가 아니다)."
                    " 로더는 pilot 모드로 읽는다")
HUMAN_VERIFICATION_V2 = {
    **HUMAN_VERIFICATION,
    "limits_table_double_check": ("미실시 — 2 판도 사람 이중 검수 · 전문가 검수 전이다. 원문 대조(검토 192 · 화면 대조 · 검토 193)만"
                                  " 했다"),
    "user_decision": "재수록 수치 사용 — 연구 책임자 결정(2026-10-24), 전문가 검수가 아니다",
}
ANSWER_CONTRACT_V2 = ("학습 타깃은 학습 레시피가 정한다. json_only 는 구조화 JSON 만, json_line_then_basis 는 JSON 한 줄 + 줄바꿈 +"
                      " target_text 다. 2 차 레시피에서는 허용치 표 · 서술 계약의 변경이 곧 재학습 사유다")

#: 2 판 메타의 신원 블록(C 08 2 판 §9-4, D 07 의 §18-3 규칙). 묶음 칸은 `lf_bundle_v1` 이다 — 파일마다 `\r\n` → `\n` 뒤의 sha256 을
#: 경로 정렬로 `경로 + 탭 + sha256 + LF` 줄로 이은 UTF-8 바이트의 sha256(평가 쪽 `limits_loader_bundle_sha256` 과 같은 값 — 시험이 맞댄다).
IDENTITY_CONTRACT_VERSION = 1
BUNDLE_RULE = "lf_bundle_v1"
#: 로더 묶음 — 정답 계산 · 조건 해석에 드는 허용치 규칙 모듈 넷(평가 쪽 `evaluation.provenance.LIMITS_RULE_FILES` 와 같은 목록 — 시험이 맞댄다).
LIMITS_LOADER_FILES: tuple[str, ...] = ("corpus/rules/clause_text.py", "corpus/rules/limit_eval.py",
                                        "corpus/rules/limits_loader.py", "corpus/rules/schema.py")
#: 생성 경로 묶음 — 2 판 서술을 내는 세 파일.
GENERATOR_FILES: tuple[str, ...] = ("corpus/generate/make_pairs_main.py", "corpus/generate/make_pairs_pilot.py",
                                    "corpus/generate/pairs_text_v2.py")


@dataclass(frozen=True)
class Edition:
    """판 등록부의 한 행. `text_contract` 가 None 이면 메타에 신원 블록을 싣지 않는다(1 판 — 메타 키를 바꾸지 않는다)."""

    version: str
    limits_csv: Path
    text_contract: str | None
    limits_status: str
    human_verification: dict
    answer_contract: str

    @property
    def limits_rel(self) -> str:
        return self.limits_csv.relative_to(REPO).as_posix()


#: **판 등록부** — 신판 경로의 C 쪽 한 자리(C 08 2 판 §1-6 · T8). 1 판 행은 지금 값 그대로다.
EDITIONS: dict[str, Edition] = {
    "v1": Edition("v1", LIMITS_CSV, None, LIMITS_STATUS, HUMAN_VERIFICATION, ANSWER_CONTRACT),
    "v2": Edition("v2", REPO / "corpus/rules/limits_v2_pilot.csv", "v2", LIMITS_STATUS_V2, HUMAN_VERIFICATION_V2,
                  ANSWER_CONTRACT_V2),
}


def edition_for_limits_path(rel: Any) -> str | None:
    """메타의 `limits_table.path` → 판 이름. 등록부에 없는 경로면 None(호출자가 거부한다)."""
    return next((name for name, e in EDITIONS.items() if e.limits_rel == rel), None)


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def lf_bundle(files: tuple[str, ...], repo: Path = REPO) -> dict:
    """묶음 블록 `{rule, files, sha256}` — 파일마다의 sha 를 싣는다(어느 파일이 바뀌었는지 말하게 — D 07 §17-3)."""
    entries = [{"path": rel, "sha256": lf_sha256(Path(repo) / rel)} for rel in sorted(files)]
    body = "".join(f"{e['path']}\t{e['sha256']}\n" for e in entries)
    return {"rule": BUNDLE_RULE, "files": entries, "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}


def identity_block(limits_table: dict, repo: Path = REPO) -> dict:
    """2 판 메타의 `identity`. `limits` 는 `limits_table` 에서 `{path, sha256}` 두 칸만 추출한 닫힌 객체다(221 I-02 — `status` 는
    `limits_table` 에 보존한다). 조건 자료는 표 안이라(안 1) `{in_table: true}` 를 생략하지 않는다."""
    return {"contract_version": IDENTITY_CONTRACT_VERSION,
            "limits": {"path": limits_table["path"], "sha256": limits_table["sha256"]},
            "limits_loader": lf_bundle(LIMITS_LOADER_FILES, repo),
            "generator": lf_bundle(GENERATOR_FILES, repo),
            "conditions": {"in_table": True}}


def same_bytes_within_train_val(manifest) -> dict:
    """train·val **안**에서 sha256 이 같은 행 — 같은 화소에 서로 다른 서술이 나가는 자리다. 매니페스트만으로 센다."""
    tv = manifest[manifest["split"].isin(["train", "val"])]
    dup = tv[tv.duplicated("sha256", keep=False)]
    keys = dup.groupby("sha256")
    return {
        "n_keys": int(keys.ngroups),
        "n_rows": len(dup),
        "n_keys_label_conflict": int(sum(1 for _, g in keys if g["has_defect"].nunique() > 1)),
        "n_keys_n_defects_differ": int(sum(1 for _, g in keys if g["n_defects"].nunique() > 1)),
    }


def build_meta(res, manifest, *, manifest_digest: str, date: str, edition: str = "v1") -> dict:
    ed = EDITIONS[edition]
    meta = P.base_meta(res, asset=f"{ASSET}_{ed.version}")
    meta["uncovered_policy"] = (("덮는 행이 없는 재질은 조항을 특정하지 않고" if ed.text_contract is None else
                                 "덮는 행이 없는 재질은 대응 조항을 특정하지 않고")
                                + " 그 사실을 문장과 skeleton.uncited_codes 에 적는다"
                                " (안 B — 09-21 판정으로 확정). 채점은 정답 조항이 공집합인 레코드를 오답이 아니라"
                                " '적용 대상 아님' 으로 세야 한다")
    limits_table = {"path": ed.limits_rel, "sha256": hashlib.sha256(ed.limits_csv.read_bytes()).hexdigest(),
                    "status": ed.limits_status}
    # 2 판만 — 신원 블록과 문장 계약 판을 `limits_table` 바로 뒤에 둔다. 1 판 메타 키는 그대로다.
    identity = ({} if ed.text_contract is None else
                {"identity": identity_block(limits_table), "text_contract": ed.text_contract})
    meta.update({
        "built_on": date,
        # 회계의 `generated_by.git_commit` 이 무엇을 가리키는지 산출물이 스스로 말한다.
        "run_identity": ("counts.json 의 generated_by.git_commit 은 **이 빌드를 돌린 작업 트리의 HEAD** 다 —"
                         " 입력의 판본이 아니다. 입력의 판본은 input_snapshot.snapshot_digest 와"
                         " limits_table.sha256 이 가리킨다. 그래서 문서만 바뀐 커밋에서도 counts.json 과"
                         " SNAPSHOT.sha256 의 다이제스트가 달라진다. '같은 입력이면 같은 바이트' 는"
                         " pairs.jsonl·discarded.jsonl 에 대한 주장이다"),
        "input_snapshot": {"path": str(SNAP.relative_to(REPO)).replace("\\", "/"),
                           "snapshot_digest": manifest_digest, "splits_used": ["train", "val"]},
        "limits_table": limits_table,
        **identity,
        "answer_contract": ed.answer_contract,
        "clause_granularity": ("clauses 는 조항 id 수준이다(두께 미상). candidate_rules 가 행별 두께 구간을 함께 싣는다 —"
                               " 반열림 [thickness_min, thickness_max), 0.01 그리드, 상한 공란은 null."
                               " 두께가 생기면 레코드만으로 행을 고를 수 있다"),
        "prompt_material": "프롬프트에 재질을 주지 않는다(현행 — 팀 결정 항목). 인용 축이 재질 추정에 기댄다는 한계가 있다",
        "human_verification": ed.human_verification,
        "pass_rate": {"value": round(len(res.made) / res.n_input, 6), "meaning": "형식 통과율 — 사실성 지표가 아니다"},
        "inherited_split_limits": {"same_bytes_within_train_val": same_bytes_within_train_val(manifest),
                                   "near_duplicates_with_eval": NEAR_DUPLICATE_NOTE},
        # 빌드가 **실제로** 확인한 수다. 상수가 아니다 — 검사를 끄면 0 이 실린다.
        "image_paths_checked": {"n": res.n_paths_checked, "of": res.n_input,
                                "basis": "저장소 루트 기준 상대경로의 실재. 타일 내용 해시는 보지 않는다"},
    })
    return meta


def _iso_date(s: str) -> str:
    return _dt.date.fromisoformat(s).isoformat()


def main(argv: list[str] | None = None) -> int:
    import argparse

    from corpus.generate.run_cycle_corpus import defect_names
    from corpus.rules import limits_loader
    from corpus.rules.skeleton_gen import load_defect_lexicon
    from data.manifest_io import load_snapshot, verify_snapshot

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--edition", required=True, choices=sorted(EDITIONS),
                    help="판 — 표 · 서술 계약 · 메타의 판(등록부 `EDITIONS`). **기본값 없음**")
    ap.add_argument("--out", required=True,
                    help="산출 디렉터리. **기본값 없음.** 봉인 전의 빌드는 임시 경로에 한다")
    ap.add_argument("--date", required=True, type=_iso_date,
                    help="메타와 counts 에 적는 빌드 날짜(YYYY-MM-DD). **기본값 없음** — 시계를 읽으면 같은 입력의"
                         " 두 빌드가 날짜만으로 갈린다")
    args = ap.parse_args(argv)
    ed = EDITIONS[args.edition]
    out_dir = Path(args.out)
    assert_not_frozen(out_dir, what="--out 대상", flag="--out")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} 가 비어 있지 않다 — 새 경로를 지정하라")

    manifest_digest = verify_snapshot(SNAP)            # 계약 4/4 — 어긋나면 여기서 멈춘다
    snap = load_snapshot(SNAP)
    table = limits_loader.load_limits(str(ed.limits_csv), pilot=True)
    res = P.build_pairs(snap.manifest, snap.annotations, table, defect_names(), load_defect_lexicon(),
                        root=REPO, check_paths=True, edition=ed.version)
    meta = build_meta(res, snap.manifest, manifest_digest=manifest_digest, date=args.date, edition=args.edition)
    digest = P.write_outputs(out_dir, res, asset=ASSET, version=ed.version, date=args.date,
                             limits_csv=ed.limits_csv, manifest_csv=SNAP / "manifest.csv", meta=meta)
    P.report(res, digest, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
