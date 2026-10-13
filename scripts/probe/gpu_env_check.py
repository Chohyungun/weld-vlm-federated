"""드라이버 위에서 고정 휠이 도는지 — 가용성 · 아키텍처 · bf16 행렬곱 · 4 비트 적재 · 작은 순전파 · 역전파."""
import json
import platform
import subprocess
import sys
import time

out: dict = {"python": sys.version.split()[0], "platform": platform.platform()}


def step(name, fn):
    t0 = time.time()
    try:
        out[name] = {"ok": True, "value": fn(), "s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001 — 실패 사유를 그대로 남긴다
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "s": round(time.time() - t0, 3)}


try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
except Exception as e:  # noqa: BLE001
    smi = f"err {e}"
out["nvidia_smi"] = smi

import torch

out["torch"] = torch.__version__
out["torch_cuda"] = torch.version.cuda
out["cudnn"] = torch.backends.cudnn.version()
step("cuda_available", torch.cuda.is_available)
step("device", lambda: {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                        "count": torch.cuda.device_count()})
step("arch_list", torch.cuda.get_arch_list)
step("driver_api", lambda: torch._C._cuda_getDriverVersion() if hasattr(torch._C, "_cuda_getDriverVersion") else None)


def bf16_mm():
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    ref = (a.float() @ b.float())
    c = a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    rel = ((c.float() - ref).norm() / ref.norm()).item()
    return {"rel_err": rel, "tflops": round(2 * 4096 ** 3 / dt / 1e12, 1), "bf16_supported": torch.cuda.is_bf16_supported()}


step("bf16_matmul", bf16_mm)


def bnb4():
    import bitsandbytes as bnb
    lin = torch.nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16)
    q = bnb.nn.Linear4bit(1024, 1024, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False, quant_type="nf4")
    q = q.cuda()
    lin = lin.cuda()
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = q(x)
    ref = lin(x)
    y.float().sum().backward()
    torch.cuda.synchronize()
    rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
    return {"bnb": bnb.__version__, "weight_dtype": str(q.weight.dtype), "rel_err_vs_bf16": rel,
            "grad_finite": bool(torch.isfinite(x.grad).all().item())}


step("bnb_4bit", bnb4)


def fwd_bwd():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(512, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 512)).cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    losses = []
    for _ in range(5):
        loss = (m(x).float() - x.float()).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 5))
    return {"losses": losses, "decreasing": losses[-1] < losses[0]}


step("fwd_bwd", fwd_bwd)

import transformers

out["transformers"] = transformers.__version__
for mod in ("peft", "accelerate", "bitsandbytes", "flwr", "ultralytics", "torchvision"):
    try:
        out[mod] = __import__(mod).__version__
    except Exception as e:  # noqa: BLE001
        out[mod] = f"err {type(e).__name__}: {e}"
print(json.dumps(out, ensure_ascii=False, indent=1))
