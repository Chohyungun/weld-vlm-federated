"""VT 라벨 파싱과 일관성 단언 (1판 3-2 · 4판 5-1).

라벨 zip 열 개(TL_ · VL_)를 읽어 장마다 기록 하나를 낸다. 원천의 `case` 문자열을 그대로 싣고
L2 키 · 코드로 바꾸지 않는다 — 사상 파일(순서 2)이 서기 전이다.

**멈춘다**(전 장이 같아야 하는 것이 어긋날 때) — 폴더 · 파일명 코드 · `information` · `info.type` · `info.material`
· 파일명 id = `info.id` · 같은 id 가 두 번 · 정상 폴더에 결함 주석 · 결함 폴더에 이름 있는 결함 주석 없음 · 어휘 밖 `case`.
`case` 가 빈 결함 주석(설정 `labels.missing_case`)은 멈추지 않고 장에 남긴다 — 세어 싣고 사상은 순서 2 가 정한다.
**장째 빠지고 제외 장부에 적힌다** — 스키마가 어긋난 장(1판 3-2), 라벨 공간 밖 `case` 가 든 장(균열, 1판 8절).
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REASON_SCHEMA = "schema_mismatch"
REASON_OUT_OF_SPACE = "out_of_label_space"


class LabelInconsistency(RuntimeError):
    """전 장이 같아야 하는 값이 어긋났다 — 빌드를 멈춘다."""


@dataclass(frozen=True)
class Ann:
    cls: str
    case: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]


@dataclass(frozen=True)
class VtLabel:
    vt_id: int
    image_id: str
    folder: str
    author_split: str            # 저자 분할(TL = Training, VL = Validation) — 이 스냅샷의 분할이 아니다
    file_stem: str
    width: int
    height: int
    is_normal: bool
    anns: tuple[Ann, ...] = field(repr=False)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def cases(self, defect_class: str) -> list[str]:
        return [a.case for a in self.anns if a.cls == defect_class]


@dataclass
class ParseResult:
    labels: list[VtLabel]
    excluded: list[dict]           # {"vt_id", "folder", "reason", "detail"}


def author_split_of(zip_name: str, cfg: dict) -> str:
    """zip 접두 → 저자 분할(설정 `copy.author_split`)."""
    for split, pre in cfg["copy"]["author_split"].items():
        if zip_name.startswith((pre["image"], pre["label"])):
            return split
    raise LabelInconsistency(f"저자 분할을 정할 수 없는 zip 이름: {zip_name}")


def zip_name_of(author_split: str, folder: str, kind: str, cfg: dict) -> str:
    """저자 분할 · 폴더 → zip 이름. `kind` 는 image · label."""
    pre = cfg["copy"]["author_split"][author_split][kind]
    return cfg["copy"]["zip_name"].format(prefix=pre, folder=folder)


def zip_member_index(zf: zipfile.ZipFile, ext: str = ".jpg") -> dict[str, str]:
    """줄기(확장자 뺀 이름) → 멤버 이름. 같은 줄기가 둘이면 멈춘다 — 조용히 하나로 덮이지 않게(06b m-12)."""
    idx: dict[str, str] = {}
    for n in zf.namelist():
        if n.lower().endswith(ext):
            stem = Path(n).stem
            if stem in idx:
                raise LabelInconsistency(f"zip 안에 같은 줄기의 멤버가 둘이다: {stem}")
            idx[stem] = n
    return idx


def folder_of_zip(name: str) -> str:
    """`TL_VTST_결함_1. 기공.zip` → `결함_1. 기공`."""
    return Path(name).stem.split("_", 2)[2]


def _schema_ok(d: dict) -> bool:
    try:
        info, img, anns = d["info"], d["image_data"], d["annotations"]
        int(info["id"]), str(info["type"]), str(info["material"])
        str(img["file_name"]), str(img["information"]), int(img["width"]), int(img["height"])
        if not isinstance(anns, list):
            return False
        for a in anns:
            xs, ys = a["coordinate"]["x"], a["coordinate"]["y"]
            if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
                return False
            str(a["class"]), str(a.get("case", ""))
        return True
    except (KeyError, TypeError, ValueError):
        return False


def parse_one(raw: bytes, *, member: str, folder: str, author_split: str, cfg: dict,
              name_re: re.Pattern) -> VtLabel | dict:
    """한 장. 스키마가 어긋나면 제외 장부 행(dict)을 준다. 일관성이 어긋나면 멈춘다."""
    lab = cfg["labels"]
    m = name_re.search(member)
    if not m or m["ext"] != "json":
        raise LabelInconsistency(f"라벨 파일명이 규칙 밖이다: {member}")
    file_id = int(m["id"])
    d = json.loads(raw.decode("utf-8"))
    if not _schema_ok(d):
        return {"vt_id": file_id, "folder": folder, "reason": REASON_SCHEMA, "detail": member}
    info, img = d["info"], d["image_data"]
    fspec = lab["folders"].get(folder)
    if fspec is None:
        raise LabelInconsistency(f"모르는 폴더: {folder}")
    checks = {
        "파일명 코드": (m["code"], fspec["code"]),
        "information": (img["information"], fspec["information"]),
        "info.type": (info["type"], lab["info_type"]),
        "info.material": (info["material"], lab["info_material"]),
        "파일명 id = info.id": (file_id, int(info["id"])),
        "file_name": (img["file_name"], Path(member).stem),
    }
    for what, (got, want) in checks.items():
        if got != want:
            raise LabelInconsistency(f"{what} 이 어긋난다 — {member}: {got!r} ≠ {want!r}")
    anns = tuple(Ann(str(a["class"]), str(a.get("case", "")), tuple(a["coordinate"]["x"]),
                     tuple(a["coordinate"]["y"])) for a in d["annotations"])
    is_normal = folder == lab["normal_folder"]
    defect_cases = [a.case for a in anns if a.cls == lab["defect_class"]]
    other = [a.cls for a in anns if a.cls not in (lab["normal_class"], lab["defect_class"])]
    if other:
        raise LabelInconsistency(f"모르는 class: {member} {other}")
    if is_normal and defect_cases:
        raise LabelInconsistency(f"정상 폴더의 장에 결함 주석이 있다: {member}")
    named = [c for c in defect_cases if c != lab["missing_case"]]
    if not is_normal and not named:
        raise LabelInconsistency(f"결함 폴더의 장에 이름 있는 결함 주석이 없다: {member}")
    vocab = set(lab["in_space_cases"]) | set(lab["out_of_space_cases"]) | {lab["missing_case"]}
    unknown = sorted(set(defect_cases) - vocab)
    if unknown:
        raise LabelInconsistency(f"어휘 밖 case: {member} {unknown}")
    rec = VtLabel(vt_id=file_id, image_id=f"{cfg['source_key']}:{file_id}", folder=folder,
                  author_split=author_split, file_stem=str(img["file_name"]),
                  width=int(img["width"]), height=int(img["height"]), is_normal=is_normal, anns=anns)
    oos = sorted(set(defect_cases) & set(lab["out_of_space_cases"]))
    if oos:
        return {"vt_id": file_id, "folder": folder, "reason": REASON_OUT_OF_SPACE, "detail": ",".join(oos),
                "label": rec}
    return rec


def parse_label_zips(label_dir: Path, cfg: dict) -> ParseResult:
    """라벨 zip 열 개 전부. 같은 id 가 두 번이면 멈춘다."""
    c = cfg["copy"]
    name_re = re.compile(cfg["labels"]["name_pattern"])
    zips = sorted(p for p in label_dir.glob("*.zip") if p.name.startswith(tuple(c["label_prefixes"])))
    if len(zips) != 10:
        raise LabelInconsistency(f"라벨 zip 이 열 개가 아니다: {len(zips)}")
    labels: list[VtLabel] = []
    excluded: list[dict] = []
    seen: set[int] = set()
    for z in zips:
        folder = folder_of_zip(z.name)
        split = author_split_of(z.name, cfg)
        with zipfile.ZipFile(z) as zf:
            for member in sorted(zf.namelist()):
                if not member.lower().endswith(".json"):
                    continue
                r = parse_one(zf.read(member), member=member, folder=folder, author_split=split,
                              cfg=cfg, name_re=name_re)
                vid = r.vt_id if isinstance(r, VtLabel) else r["vt_id"]
                if vid in seen:
                    raise LabelInconsistency(f"같은 id 가 두 번이다: {vid}")
                seen.add(vid)
                if isinstance(r, VtLabel):
                    labels.append(r)
                else:
                    excluded.append(r)     # 라벨 공간 밖 장은 `label` 을 들고 간다 — 근사 중복 후보에는 넣는다(1판 5-2)
    labels.sort(key=lambda r: r.vt_id)
    return ParseResult(labels, excluded)
