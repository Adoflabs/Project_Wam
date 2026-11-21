"""Minimal Kaggle GPU smoke test: confirms a GPU is attached and usable, writes result.json."""
import json, platform, subprocess, time

result = {"python": platform.python_version()}
try:
    result["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:
    result["nvidia_smi"] = f"unavailable: {e}"

import torch
result["torch"] = torch.__version__
result["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    dev = torch.device("cuda")
    result["device_count"] = torch.cuda.device_count()
    result["device_name"] = torch.cuda.get_device_name(0)
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    n, t0 = 20, time.time()
    for _ in range(n):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    result["matmul_fp16_tflops"] = round(2 * 4096**3 * n / dt / 1e12, 2)
    # correctness check against CPU on a small slice
    ref = (a[:64].float().cpu() @ b[:, :64].float().cpu())
    result["max_abs_err_vs_cpu"] = float((c[:64, :64].float().cpu() - ref).abs().max())

result["status"] = "PASS" if result["cuda_available"] else "FAIL: no CUDA device"
print(json.dumps(result, indent=2))
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
