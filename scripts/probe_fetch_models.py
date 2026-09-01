"""프로브 2 준비 — 크기-시간 곡선용 가중치 확보. 다운로드만 한다."""
import sys
from huggingface_hub import snapshot_download

for mid in sys.argv[1:]:
    p = snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    print(f"{mid} -> {p}", flush=True)
