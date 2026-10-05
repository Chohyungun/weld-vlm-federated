"""동결 기록이 보증하는 것 — **목 스냅샷 복원** — 을 현행 생성기로 다시 잰다.

기록(`data/mock/FROZEN_RECORD.yaml`)의 `restoration_verified_inputs` 는 "아래 사상표·프로필을 아래
생성기에 넣으면 두 동결본의 구성 파일 8개가 잠금 기록까지 바이트 단위로 복원된다" 고 말한다.
생성기 파일의 해시가 바뀌었을 때 병기의 근거가 되는 것은 이 성질이 **이어지는가** 다.

절차. 추적 파일만 담은 스크래치 사본(`git archive HEAD`)을 풀고, 그 안의 두 입력을 기록이 가리키는
blob 으로 바꿔 끼운 뒤 생성기를 빈 경로에 돌린다. 작업 트리와 동결 디렉터리는 건드리지 않는다.
기대값은 기록의 `snapshots` 에서 읽는다 — 동결 디렉터리를 열지 않는다.

사용: python reverify_mock_restoration.py <스크래치 디렉터리>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "data" / "mock" / "FROZEN_RECORD.yaml"
LOCK = "SNAPSHOT.sha256"


def _git(*args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, input=stdin, capture_output=True,
                          check=True).stdout


def main(argv: list[str]) -> int:
    scratch = Path(argv[1]).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        print(f"스크래치가 비어 있지 않다: {scratch}", file=sys.stderr)
        return 2
    scratch.mkdir(parents=True, exist_ok=True)
    rec = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    rvi = rec["restoration_verified_inputs"]

    head = _git("rev-parse", "HEAD").decode().strip()
    tar_path = scratch.parent / (scratch.name + ".tar")
    tar_path.write_bytes(_git("archive", "--format=tar", "HEAD"))
    with tarfile.open(tar_path) as tf:
        tf.extractall(scratch, filter="data")
    tar_path.unlink()

    # 입력 둘을 기록의 blob 으로 바꿔 끼운다. 내용 해시가 기록과 같은지 먼저 본다.
    swapped = {}
    for key in ("label_map", "mock_profile"):
        ent = rvi[key]
        data = _git("cat-file", "blob", ent["git_blob"])
        sha = hashlib.sha256(data).hexdigest()
        if sha != ent["sha256"]:
            print(f"{key}: blob 의 내용 해시가 기록과 다르다 {sha}", file=sys.stderr)
            return 2
        (scratch / ent["path"]).write_bytes(data)
        swapped[key] = {"path": ent["path"], "sha256": sha}

    # 생성기 코드는 현행이다. 기록의 병기 값과 같은지 적는다(같지 않으면 무엇을 잰 것인지 흐려진다).
    code = {}
    for rel, ent in rvi["generator"].items():
        data = (scratch / rel).read_bytes().replace(b"\r\n", b"\n")
        sha = hashlib.sha256(data).hexdigest()
        want = (ent.get("after_public_cleanup") or {}).get("sha256", ent["sha256"])
        code[rel] = {"sha256": sha, "기록의_현행값과_같음": sha == want}

    with tempfile.TemporaryDirectory(dir=scratch.parent) as out:
        r = subprocess.run([sys.executable, "-X", "utf8", "-B", "scripts/make_mock_manifest.py",
                            "--out-root", out], cwd=scratch, capture_output=True, text=True,
                           encoding="utf-8")
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
            return 2
        result = {}
        n_same = n_total = 0
        for snap_name, want in rec["snapshots"].items():
            got_dir = Path(out) / snap_name
            files = {}
            for fname, want_sha in {**want["files"], LOCK: want["lock_file"]["sha256"]}.items():
                got = hashlib.sha256((got_dir / fname).read_bytes()).hexdigest()
                files[fname] = got == want_sha
                n_total += 1
                n_same += got == want_sha
            result[snap_name] = files

    out_doc = {
        "측정일": date.today().isoformat(),     # 실행한 날. 글자로 박지 않는다
        "스크래치_출처_커밋": head,
        "바꿔_끼운_입력": swapped,
        "생성기_코드": code,
        "구성_파일_바이트_일치": result,
        "일치_수": f"{n_same}/{n_total}",
        "전건_일치": n_same == n_total,
    }
    print(json.dumps(out_doc, ensure_ascii=False, indent=2))
    return 0 if n_same == n_total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
