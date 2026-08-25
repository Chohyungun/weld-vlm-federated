"""P1 규격 항등성 · P4 인코딩 지문 — 결정론적 100% 게이트. §5-2·§5-3.

**100% 아니면 머지 금지다.** 확률적 프로브(P2·P3)와 달리 여기는 판정에 여지가 없다.

두 프로브 모두 **파일 헤더 판독값을 쓰고 라벨 JSON 값을 믿지 않는다.** RIAWELC 에서
라벨이 224 라고 적혀 있는데 실제 파일이 227 이었던 전례가 있다. 라벨을 믿으면 규격이
통일됐다고 보고하면서 실제로는 안 통일된 상태가 된다.

구현(A)과 판정(D)이 갈린다 — 생성 측이 자기 산출물을 검증하지 않는다(규약 3-5 취지).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


@dataclass(frozen=True)
class ImageHeader:
    """**파일 헤더에서 직접 읽은** 값. 매니페스트 값이 아니다."""

    image_id: str
    width_px: int
    height_px: int
    mode: str = ""
    n_channels: int = 0
    subsampling: str = ""
    progressive: bool = False
    quant_table_hash: str = ""
    file_bytes: int = 0

    @property
    def encoding_fingerprint(self) -> tuple:
        """P4 가 종수를 세는 조합 — (양자화표, 서브샘플링, 프로그레시브, mode, 채널수)."""
        return (
            self.quant_table_hash, self.subsampling,
            self.progressive, self.mode, self.n_channels,
        )


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    passed: bool
    detail: str
    violations: tuple[str, ...] = field(default=())
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "probe": self.probe, "passed": self.passed, "detail": self.detail,
            "n_violations": len(self.violations),
            "violations_head": list(self.violations[:10]),
            **self.stats,
        }


def p1_spec_identity(headers: Iterable[ImageHeader]) -> ProbeResult:
    """P1 — 처리 후 RT 전 행이 1280×720 인가. **위반 1건이면 빌드 실패.**

    이 항등식이 성립하면 절대 규격·화소수·종횡비·시야·VLM 토큰 수가 **구성적으로
    0비트**가 된다. 상수 입력에서 클래스를 가르는 함수는 존재하지 않기 때문이다.
    """
    items = list(headers)
    bad = tuple(
        f"{h.image_id}({h.width_px}x{h.height_px})"
        for h in items
        if h.width_px != TARGET_WIDTH or h.height_px != TARGET_HEIGHT
    )
    n = len(items)
    rate = (n - len(bad)) / n if n else 0.0
    return ProbeResult(
        probe="P1",
        passed=n > 0 and not bad,
        detail=(
            f"{n}장 전량 {TARGET_WIDTH}x{TARGET_HEIGHT}"
            if n and not bad
            else f"규격 위반 {len(bad)}건 / {n}장 — 빌드 실패"
            if n
            else "검사 대상이 0장이다 — 통과로 처리하지 않는다"
        ),
        violations=bad,
        stats={"n_images": n, "conformance": rate},
    )


def p4_encoding_fingerprint(headers: Iterable[ImageHeader]) -> ProbeResult:
    """P4 — RT 전 파일의 인코딩 지문 조합 **종수 == 1**.

    양자화표가 여러 종이면 압축 세대 차이가 남아 출처가 새는 경로가 된다. 규격을
    통일해도 이쪽이 열려 있으면 P2 가 그것을 잡아낸다 — 미리 막는 편이 싸다.
    """
    items = list(headers)
    groups: dict[tuple, list[str]] = {}
    for h in items:
        groups.setdefault(h.encoding_fingerprint, []).append(h.image_id)
    kinds = len(groups)
    minority = tuple(
        img
        for fp, imgs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[1:]
        for img in imgs
    )
    return ProbeResult(
        probe="P4",
        passed=kinds == 1,
        detail=(
            f"{len(items)}장 인코딩 지문 1종 — 통일 확인"
            if kinds == 1
            else f"인코딩 지문 {kinds}종 — 1종이어야 한다"
        ),
        violations=minority,
        stats={
            "n_images": len(items), "n_fingerprints": kinds,
            "group_sizes": sorted((len(v) for v in groups.values()), reverse=True),
        },
    )


def read_header(path: str | Path, image_id: str | None = None) -> ImageHeader:
    """이미지 파일 헤더를 직접 읽는다. 매니페스트를 참조하지 않는다.

    호출 측이 `data/raw` 를 수정하지 않도록 **읽기 전용으로만** 연다.
    """
    import hashlib

    from PIL import Image, JpegImagePlugin

    p = Path(path)
    with Image.open(p) as im:
        info = im.info
        # JPEG 의 양자화표는 `info` 가 아니라 `im.quantization` 에 있다. `info` 만 읽으면
        # 전 파일이 빈 값이 되어 P4 가 **공허하게 통과**한다. 미도달을 통과처럼 쓰지
        # 않으려면 여기서 정확한 출처를 읽어야 한다.
        qt = getattr(im, "quantization", None) or info.get("quantization")
        quant_hash = ""
        if qt:
            blob = b"".join(bytes(qt[k]) for k in sorted(qt))
            quant_hash = hashlib.sha256(blob).hexdigest()[:16]
        subsampling = info.get("subsampling")
        if subsampling is None and im.format == "JPEG":
            try:
                subsampling = JpegImagePlugin.get_sampling(im)
            except Exception:                      # noqa: BLE001
                subsampling = None
        return ImageHeader(
            image_id=image_id or p.name,
            width_px=im.width,
            height_px=im.height,
            mode=im.mode,
            n_channels=len(im.getbands()),
            subsampling="" if subsampling is None else str(subsampling),
            progressive=bool(info.get("progressive", 0)),
            quant_table_hash=quant_hash,
            file_bytes=p.stat().st_size,
        )


def compare_header_to_manifest(
    headers: Iterable[ImageHeader],
    manifest_wh: dict[str, tuple[int, int]],
) -> ProbeResult:
    """헤더 판독값 ↔ 매니페스트 선언값 전수 대조.

    RIAWELC 227 대 224 전례가 이 검사의 존재 이유다. 어긋나면 **매니페스트가 틀린 것**
    이므로 A 에 회부하고, 그 전에는 규격이 통일됐다고 보고하지 않는다.
    """
    items = list(headers)
    bad = tuple(
        f"{h.image_id}: 헤더 {h.width_px}x{h.height_px} vs 매니페스트 "
        f"{manifest_wh[h.image_id][0]}x{manifest_wh[h.image_id][1]}"
        for h in items
        if h.image_id in manifest_wh
        and (h.width_px, h.height_px) != manifest_wh[h.image_id]
    )
    missing = tuple(h.image_id for h in items if h.image_id not in manifest_wh)
    return ProbeResult(
        probe="P1-header-vs-manifest",
        passed=not bad and not missing,
        detail=(
            f"{len(items)}장 헤더·매니페스트 일치"
            if not bad and not missing
            else f"불일치 {len(bad)}건 / 매니페스트 누락 {len(missing)}건 — A 회부"
        ),
        violations=bad + tuple(f"{m}: 매니페스트에 없음" for m in missing),
        stats={"n_images": len(items)},
    )


def gate(results: Sequence[ProbeResult]) -> tuple[bool, str]:
    """결정론적 게이트 종합. **하나라도 실패면 머지 금지**(§5-3)."""
    failed = [r.probe for r in results if not r.passed]
    if failed:
        return False, f"결정론적 프로브 실패: {', '.join(failed)} — 머지 금지"
    return True, f"결정론적 프로브 {len(results)}종 전부 통과"
