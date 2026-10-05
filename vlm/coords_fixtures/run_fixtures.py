#!/usr/bin/env python3
"""골든 픽스처 독립 재실행 스크립트 (독립 검증용).

**import도 설치도 없이** 이 파일 하나를 실행해 좌표 규약 구현을 검증할 수 있게
만들었다. 계약 #4 §3-2의 "D는 C의 CI 통과를 신뢰하지 않고 다시 돈다"를 위한 것이다.

    python vlm/coords_fixtures/run_fixtures.py
    python vlm/coords_fixtures/run_fixtures.py --coords /경로/coords.py --json /경로/fixtures.json

종료 코드 0이면 통과, 1이면 실패다. 실패한 케이스는 무엇이 어떻게 어긋났는지 값을 찍는다.

기대값은 `golden_fixtures.json`에 손계산으로 들어 있다. **구현 출력으로 기대값을 갱신하는
경로를 이 스크립트는 제공하지 않는다** — 그런 스위치가 있으면 구현이 틀렸을 때 기대값을
맞추는 쪽으로 손이 가고, 그 순간 픽스처가 아무것도 검증하지 못하게 된다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_COORDS = HERE.parent / "coords.py"
DEFAULT_JSON = HERE / "golden_fixtures.json"


def load_coords(path: Path):
    """coords.py를 파일 경로에서 직접 로드한다 (패키지 설치·PYTHONPATH 불필요)."""
    spec = importlib.util.spec_from_file_location("_weldfl_coords_under_test", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"coords 모듈을 로드할 수 없다: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 가 타입 힌트를 해석할 때 sys.modules 에서 자기 모듈을 찾으므로
    # exec_module 전에 등록해야 한다. 빠뜨리면 @dataclass 에서 AttributeError 가 난다.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def check_case(coords, case: dict, tol: float) -> list[str]:
    """한 케이스를 검사하고 실패 사유 목록을 돌려준다 (빈 목록이면 통과)."""
    fails: list[str] = []
    cfg = coords.CoordCfg(**case["cfg"])
    geom = coords.ImageGeom(**case["geom"])
    box = case["bbox_px"]
    exp = case["expect"]

    model = coords.to_model(box, geom, cfg)
    if not all(close(a, b, tol) for a, b in zip(model, exp["model_float"])):
        fails.append(f"to_model 불일치: 실제 {list(model)} != 기대 {exp['model_float']}")

    q = coords.quantize(model)
    if list(q) != list(exp["model_quantized"]):
        fails.append(f"quantize 불일치: 실제 {list(q)} != 기대 {exp['model_quantized']}")

    back = coords.to_px(q, geom, cfg)
    if not all(close(a, b, tol) for a, b in zip(back, exp["back_px"])):
        fails.append(f"to_px 불일치: 실제 {list(back)} != 기대 {exp['back_px']}")

    err = coords.roundtrip_error_px(box, geom, cfg)
    if not close(err, exp["max_roundtrip_err_px"], tol):
        fails.append(
            f"왕복 오차 불일치: 실제 {err!r} != 기대 {exp['max_roundtrip_err_px']!r}"
        )

    deg = coords.is_degenerate(q)
    if bool(deg) != bool(exp["degenerate_after_quantize"]):
        fails.append(
            f"퇴화 판정 불일치: 실제 {deg} != 기대 {exp['degenerate_after_quantize']}"
        )

    # 왕복 예산 — 퇴화 케이스는 박스가 뭉개진 것이라 예산 개념이 성립하지 않으므로 건너뛴다.
    # 예산 공식은 규약마다 다르므로 coords.roundtrip_budget_px 를 단일 소스로 쓴다.
    if not exp["degenerate_after_quantize"]:
        budget = coords.roundtrip_budget_px(geom, cfg)
        if err > budget + tol:
            fails.append(f"왕복 예산 초과: {err!r} > {budget!r}")

    return fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="좌표 골든 픽스처 재실행")
    ap.add_argument("--coords", type=Path, default=DEFAULT_COORDS, help="검증할 coords.py 경로")
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="골든 픽스처 JSON 경로")
    ap.add_argument("--quiet", action="store_true", help="통과 케이스는 출력하지 않는다")
    args = ap.parse_args(argv)

    coords = load_coords(args.coords)
    data = json.loads(args.json.read_text(encoding="utf-8"))
    tol = float(data["tolerance"]["abs_tol_px"])

    total = 0
    failed = 0
    for case in data["cases"]:
        total += 1
        try:
            fails = check_case(coords, case, tol)
        except Exception as exc:  # 예외도 실패다 — 조용히 넘기지 않는다
            fails = [f"예외 발생: {type(exc).__name__}: {exc}"]
        if fails:
            failed += 1
            print(f"[FAIL] {case['id']}")
            for f in fails:
                print(f"       {f}")
        elif not args.quiet:
            print(f"[ok]   {case['id']}")

    print("-" * 60)
    print(f"골든 픽스처 {total}건 중 통과 {total - failed}건 / 실패 {failed}건")
    print(f"coords.py sha256: {coords.coords_source_sha256()}")

    if failed:
        print("FIXTURES: FAIL — 좌표 규약 구현이 골든 기대값과 어긋난다.")
        return 1
    print("FIXTURES: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
