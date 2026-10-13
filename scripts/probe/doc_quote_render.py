"""python -m scripts.probe.doc_quote_render --spec <인용.json> --template <틀.md> --out <보고.md>

보고서에 다른 문서의 수를 인용할 때 **손으로 옮기지 않는다.** 인용 명세(JSON)가 원천 파일마다 바이트 sha256(LF 정규화)을
적고, 인용마다 정규식 하나를 적는다. 틀의 `⟦q:이름⟧` 는 그 정규식의 첫 묶음으로, `⟦src:원천⟧` 은 경로와 해시로, `⟦line:이름⟧` 은
맞은 줄 번호로 채운다. 원천의 해시가 다르거나, 정규식이 정확히 한 번 맞지 않거나, 틀에 채우지 못한 자리가 남으면 멈춘다.

명세의 꼴:
    {"sources": {"원천": {"path": "...", "sha256": "..."}},
     "quotes": {"이름": {"src": "원천", "pattern": "..."}}}
`path` 는 저장소 루트 기준이다. 이 작업 트리에 없는 다른 트랙 커밋의 문서는 `path` 대신 `"git": "<커밋>:<경로>"` 로 적고,
실행할 때 `--copy 원천=<git show 로 꺼낸 사본>` 으로 사본을 준다 — 해시로 맞대므로 사본이 달라지면 멈춘다(명세에 로컬 경로를 싣지 않는다).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def lf_bytes(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build(spec: dict, copies: dict[str, Path]) -> dict[str, str]:
    texts: dict[str, str] = {}
    vals: dict[str, str] = {}
    for name, s in spec["sources"].items():
        if "git" in s:
            if name not in copies:
                raise SystemExit(f"원천 {name} 은 `{s['git']}` 의 사본을 --copy 로 줘야 한다")
            where, cite = copies[name], f"{s['git']}"
        else:
            where, cite = resolve(s["path"]), s["path"]
        raw = lf_bytes(where)
        got = hashlib.sha256(raw).hexdigest()
        if got != s["sha256"].lower():
            raise SystemExit(f"원천 {name} 의 해시가 다르다 — {got} 대 {s['sha256']}")
        texts[name] = raw.decode("utf-8")
        vals[f"src:{name}"] = f"`{cite}`(`{got[:16]}…`)"
    for name, q in spec["quotes"].items():
        text = texts[q["src"]]
        hits = list(re.finditer(q["pattern"], text))
        if len(hits) != 1:
            raise SystemExit(f"인용 {name}: 꼴이 {len(hits)} 번 맞는다 — 한 번이어야 한다")
        m = hits[0]
        vals[f"q:{name}"] = m.group(1) if m.groups() else m.group(0)
        vals[f"line:{name}"] = str(text.count("\n", 0, m.start()) + 1)
    return vals


def render(spec_path: Path, template: Path, out: Path, copies: dict[str, Path]) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    vals = build(spec, copies)
    tpl = template.read_text(encoding="utf-8")
    used = set(re.findall(r"⟦([^⟧]+)⟧", tpl))
    missing = sorted(used - set(vals))
    if missing:
        raise SystemExit(f"틀의 자리 {missing} 에 값이 없다")
    out.write_text(re.sub(r"⟦([^⟧]+)⟧", lambda m: vals[m.group(1)], tpl), encoding="utf-8", newline="\n")
    print(json.dumps({"filled": len(used), "unused": sorted(set(vals) - used)}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy", action="append", default=[], metavar="원천=경로")
    args = ap.parse_args()
    copies = {}
    for c in args.copy:
        k, _, v = c.partition("=")
        copies[k] = Path(v)
    render(args.spec, args.template, args.out, copies)


if __name__ == "__main__":
    main()
