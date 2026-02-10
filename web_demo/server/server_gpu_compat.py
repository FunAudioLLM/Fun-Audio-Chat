"""
GPU-compat launcher for FunAudioChat server.

Use this when running on new architectures (e.g., sm_121) with older CUDA/NVRTC.
It sets conservative env defaults before importing torch to avoid NVRTC arch errors
and prefers PTX fallback where possible.

Usage:
  FUN_AUDIOCHAT_CUDA_COMPAT=1 python web_demo/server/server_gpu_compat.py --host 0.0.0.0 --port 11235
"""
import os

# Only apply when explicitly enabled.
if os.environ.get("FUN_AUDIOCHAT_CUDA_COMPAT") == "1":
    # Prefer PTX fallback for new architectures when SASS is unavailable.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")
    # Avoid aggressive kernel fusion paths that may JIT-compile with NVRTC.
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("XFORMERS_DISABLE_TRITON", "1")
    os.environ.setdefault("FLASH_ATTENTION_DISABLE", "1")

from web_demo.server import server


if __name__ == "__main__":
    server.main()
