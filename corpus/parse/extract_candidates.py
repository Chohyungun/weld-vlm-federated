"""corpus 후보 문서 전량 추출 — 텍스트·표·그림 + 메타데이터 (작업지시서 7~9절).

배제는 색인·학습 투입 단계의 판정이고 추출은 전량 한다. 검사 방식이 달라도 결함 종류
정의·발생 원인·조치는 공통이라, 판정 축이 원인·조치로 옮겨가면 그대로 재료가 된다.
다만 허용치 수치는 검사 방식마다 다르므로 추출물마다 inspection_method 를 단다
(계약 #3 LimitRow 와 같은 축). 이게 없으면 RT·UT 수치가 섞였는지 나중에 판별할 수 없다.

산출물은 공유 드라이브에 두고 저장소에는 스크립트와 메타데이터만 남긴다.

실행: uv run python -m corpus.parse.extract_candidates [--only 파일명조각] [--no-ocr]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path

# sm_120/Windows: triton 부재로 torch.compile 이 docling layout 모델에서 치명 오류가 된다
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

SRC = Path("G:/공유 드라이브/대한산업공학회_추계학술대회/corpus_candidate")
DST = Path("G:/공유 드라이브/대한산업공학회_추계학술대회/corpus_extracted")
REPO = Path(__file__).resolve().parents[2]
META_OUT = REPO / "corpus/parse/extracted_manifest.json"

SCAN_TEXT_THRESHOLD = 50   # 문자/쪽 미만이면 텍스트 레이어 없음으로 본다

# ---- 검사 방식 판정 (inspection_method) -------------------------------------
METHOD_PAT = {
    "RT": re.compile(r"방사선|radiograph|\bRT\b|X-?ray|엑스선", re.I),
    "UT": re.compile(r"초음파|ultrasonic|\bUT\b|phased\s*array|TOFD", re.I),
    "VT": re.compile(r"육안검사|외관검사|visual\s+(?:inspection|test)|\bVT\b", re.I),
    "MT": re.compile(r"자분탐상|magnetic\s+particle|\bMT\b", re.I),
    "PT": re.compile(r"침투탐상|liquid\s+penetrant|dye\s+penetrant|\bPT\b", re.I),
}

# ---- 내용 유형 판정 (content_type) ------------------------------------------
CONTENT_PAT = {
    "허용치수치": re.compile(
        r"(허용|합격기준|판정기준|acceptance\s+criteria|allowable|permissible|max(?:imum)?\s+"
        r"(?:length|size|depth))", re.I),
    "원인": re.compile(r"발생\s*원인|원인|cause[sd]?\b|due\s+to|기인", re.I),
    "조치": re.compile(r"대책|방지|예방|보수|재용접|조치|remedy|prevention|countermeasure|repair", re.I),
    "결함정의": re.compile(
        r"기공|균열|슬래그|융합불량|언더컷|용입불량|porosity|crack|slag|undercut|"
        r"lack\s+of\s+fusion|incomplete\s+penetration|discontinuit", re.I),
    "공정": re.compile(r"용접\s*공정|welding\s+process|SMAW|GMAW|GTAW|FCAW|SAW\b|MIG|TIG", re.I),
}

# ---- 결함 형상 예시 그림 판정 ------------------------------------------------
# D4 페어가 전량 합성이라 실물 시각 근거가 부족하다. 결함 형상 예시는 따로 표시한다.
FIGURE_DEFECT_PAT = re.compile(
    r"(기공|균열|슬래그|융합불량|언더컷|용입|결함|defect|porosity|crack|slag|undercut|"
    r"discontinuit|radiograph|macrograph|appearance)", re.I)

# ---- 저작권 등급 (6절 전수 판정 결과를 그대로 옮긴다) -------------------------
# 공개가 아니면 usable_for 는 참고전용으로 고정한다 (작업지시서 7절).
COPYRIGHT = {
    "applsci-10-08629-v2.pdf": ("공개", "MDPI Applied Sciences 2020, CC BY 오픈액세스 명시"),
    "Shipyard_Industry_Standards.pdf": ("공개", "OSHA 2268-11R 2015, 미국 정부간행물"),
    "API_1104_22_nd_EDITION_UNDERSTANDING_WEL.pdf": ("유료표준", "API 1104 합격기준 수치 전재"),
    "ESAB_WELDING_HANDBOOK.pdf": ("상용출판물", "ESAB 시판 핸드북"),
    "Pipelines_Welding_Handbook_Welding_techn.pdf": ("상용출판물", "시판 핸드북"),
    "The_Welding_Handbook.pdf": ("상용출판물", "시판 핸드북"),
    "The_Welding_Handbook (1).pdf": ("상용출판물", "시판 핸드북 (다른 판본)"),
    "Welding_Working_Title.pdf": ("상용출판물", "Welding - Modern Topics 단행본"),
    "000000031173_20260823231838.pdf": ("학술지", "학위논문 CC BY-NC-ND 2.0 KR (변경금지)"),
    "000000162723_20260823224735.pdf": ("학술지", "학위논문 CC BY-NC-ND 2.0 KR (변경금지)"),
    "000000172086_20260823232451.pdf": ("학술지", "학위논문 CC BY-NC-ND 2.0 KR (변경금지)"),
    "AI 기반 전기차 알루미늄 부품 마찰교반용접부 비파괴 품질 평가.pdf": ("학술지", "누리미디어 배포"),
    "Review on Ultrasonic Welding Quality Monitoring Te.pdf": ("학술지", "누리미디어 배포"),
    "알루미늄 MIG 용접부 기공 특성과 대책.pdf": ("학술지", "누리미디어 배포"),
    "원전 압력용기 용접부 초음파탐상, 결함크기 평가 및 결함 수리 경험.pdf": ("학술지", "누리미디어 배포"),
    "인공지능 기반 조선해양 용접 품질 정보 관리 및 결함 검사 플랫폼 개발.pdf": ("학술지", "누리미디어 배포"),
    "초음파 결함 크기 측정 기법.pdf": ("학술지", "누리미디어 배포"),
    "Common_Welding_Methods_And_Weld_Defects.docx": ("미확인", "Marine Insight 웹 기사 2017"),
}
DEFAULT_COPYRIGHT = ("미확인", "발행처·저자·연도 미확인 (스캔본이라 메타데이터도 없음)")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def probe_text_layer(pdf_path: Path) -> dict:
    """텍스트 레이어 유무 — OCR 필요 여부를 정한다."""
    import pdfplumber
    n_pages = n_low = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                if len((page.extract_text() or "")) < SCAN_TEXT_THRESHOLD:
                    n_low += 1
    except Exception as e:
        return {"n_pages": 0, "n_low_text": 0, "needs_ocr": True, "probe_error": str(e)[:120]}
    ratio = n_low / n_pages if n_pages else 1.0
    return {"n_pages": n_pages, "n_low_text": n_low, "low_text_ratio": round(ratio, 3),
            "needs_ocr": ratio > 0.5}


def classify_method(text: str) -> tuple[str, dict]:
    counts = {k: len(p.findall(text)) for k, p in METHOD_PAT.items()}
    top = max(counts, key=lambda k: counts[k])
    return (top if counts[top] >= 3 else "무관"), counts


def classify_content(text: str) -> list[str]:
    return [k for k, p in CONTENT_PAT.items() if p.search(text)]


# docling-parse(C++ 백엔드)는 Windows 에서 한글·공백이 섞인 경로를 열지 못한다.
# 원본이 "G:\공유 드라이브\..." 에 있어 전량 실패하므로 ASCII 임시 경로로 옮겨 넣는다.
# (지정 함정 구간 #7 의 실패 사례 — 문서 결함이 아니라 경로 인코딩 문제다)
STAGE = Path(os.environ.get("TEMP", "/tmp")) / "weldfl_stage"


def stage_ascii(src: Path, idx: int) -> Path:
    STAGE.mkdir(parents=True, exist_ok=True)
    dst = STAGE / f"doc{idx:03d}{src.suffix.lower()}"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        dst.write_bytes(src.read_bytes())
    return dst


def convert(pdf_path: Path, use_ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (EasyOcrOptions, PdfPipelineOptions,
                                                    TableFormerMode)
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    opts.do_ocr = use_ocr
    if use_ocr:
        opts.ocr_options = EasyOcrOptions(lang=["ko", "en"], force_full_page_ocr=True)
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return conv.convert(pdf_path)


def extract_one(path: Path, args) -> dict:
    rec: dict = {"file": path.name, "sha256": sha256(path),
                 "size_mb": round(path.stat().st_size / 1e6, 2)}
    cls, why = COPYRIGHT.get(path.name, DEFAULT_COPYRIGHT)
    rec["copyright_class"] = cls
    rec["copyright_note"] = why
    # 공개가 아니면 색인·학습 투입을 막는다. 판단은 총괄이 한다.
    rec["usable_for"] = "색인후보" if cls == "공개" else "참고전용"

    out_dir = DST / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".docx":
        import zipfile
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        txt = " ".join(re.sub(r"<[^>]+>", " ", xml).split())
        (out_dir / f"{path.stem}.md").write_text(txt, encoding="utf-8")
        method, counts = classify_method(txt)
        rec.update({"status": "ok", "engine": "zipfile(docx)", "ocr": False,
                    "n_pages": None, "n_tables": 0, "n_pictures": 0,
                    "inspection_method": method, "method_counts": counts,
                    "content_types": classify_content(txt), "chars": len(txt)})
        return rec

    probe = probe_text_layer(path)
    rec.update(probe)
    use_ocr = probe["needs_ocr"] and not args.no_ocr
    rec["ocr"] = use_ocr

    t0 = time.time()
    staged = stage_ascii(path, args._idx)
    rec["staged_ascii"] = True   # 한글 경로 우회 (docling-parse 제약)
    try:
        result = convert(staged, use_ocr)
        doc = result.document
    except Exception as e:
        rec.update({"status": "docling_failed", "engine": "docling", "elapsed_sec": round(time.time()-t0, 1),
                    "error_type": type(e).__name__, "error": str(e)[:300],
                    "traceback_tail": traceback.format_exc()[-400:]})
        return rec

    md = doc.export_to_markdown()
    (out_dir / f"{path.stem}.md").write_text(md, encoding="utf-8")

    # 표 — 구조를 살려 CSV 로
    tables = []
    for i, tb in enumerate(doc.tables):
        try:
            df = tb.export_to_dataframe()
        except Exception:
            continue
        page = tb.prov[0].page_no if tb.prov else None
        name = f"{path.stem}_t{i:03d}.csv"
        df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
        tables.append({"idx": i, "page": page, "rows": int(df.shape[0]),
                       "cols": int(df.shape[1]), "csv": name})

    # 그림 — 캡션과 함께. 결함 형상 예시는 따로 표시한다
    pics_dir = out_dir / "figures"
    pictures = []
    for i, pic in enumerate(doc.pictures):
        cap = ""
        try:
            cap = pic.caption_text(doc) or ""
        except Exception:
            pass
        page = pic.prov[0].page_no if pic.prov else None
        is_defect = bool(FIGURE_DEFECT_PAT.search(cap)) if cap else False
        entry = {"idx": i, "page": page, "caption": cap[:220], "defect_example": is_defect}
        try:
            img = pic.get_image(doc)
            if img is not None:
                pics_dir.mkdir(exist_ok=True)
                fn = f"{path.stem}_fig{i:03d}.png"
                img.save(pics_dir / fn)
                entry["png"] = fn
        except Exception as e:
            entry["image_error"] = f"{type(e).__name__}"
        pictures.append(entry)

    method, counts = classify_method(md)
    rec.update({
        "status": "ok", "engine": "docling", "docling_status": str(result.status),
        "elapsed_sec": round(time.time() - t0, 1),
        "n_tables": len(tables), "n_pictures": len(pictures),
        "n_defect_figures": sum(1 for p in pictures if p["defect_example"]),
        "inspection_method": method, "method_counts": counts,
        "content_types": classify_content(md), "chars": len(md),
        "tables": tables, "pictures": pictures,
    })
    (out_dir / "extract_meta.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="파일명 부분 일치만 처리")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--resume", action="store_true", help="이미 성공한 문서는 건너뛴다")
    args = ap.parse_args()

    DST.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.resume and META_OUT.exists():
        done = {r["file"]: r for r in json.loads(META_OUT.read_text(encoding="utf-8"))["documents"]
                if r.get("status") == "ok"}

    files = sorted(p for p in SRC.iterdir() if p.is_file())
    if args.only:
        files = [p for p in files if args.only.lower() in p.name.lower()]

    out = []
    for k, p in enumerate(files, 1):
        if p.name in done:
            print(f"[{k}/{len(files)}] skip(done) {p.name}", flush=True)
            out.append(done[p.name]); continue
        print(f"[{k}/{len(files)}] {p.name}", flush=True)
        args._idx = k
        rec = extract_one(p, args)
        print(f"    -> {rec.get('status')} {rec.get('inspection_method','')} "
              f"표{rec.get('n_tables',0)} 그림{rec.get('n_pictures',0)} "
              f"({rec.get('elapsed_sec','?')}s)", flush=True)
        out.append(rec)
        META_OUT.write_text(json.dumps(
            {"src": str(SRC), "dst": str(DST), "documents": out},
            ensure_ascii=False, indent=1), encoding="utf-8")

    ok = sum(1 for r in out if r.get("status") == "ok")
    print(f"\n성공 {ok} / 실패 {len(out)-ok}  메타: {META_OUT}")


if __name__ == "__main__":
    main()
