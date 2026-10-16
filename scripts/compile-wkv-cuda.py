import ctypes
import os
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
runtime_root = root / ".codex-tmp" / "cuda-python" / "nvidia" / "cuda_nvrtc" / "bin"
libraries = sorted(runtime_root.glob("nvrtc64_*.dll"))
if not libraries:
    raise SystemExit("NVRTC runtime not found under .codex-tmp/cuda-python")

os.environ["PATH"] = str(runtime_root) + os.pathsep + os.environ.get("PATH", "")
dll_search = os.add_dll_directory(str(runtime_root)) if hasattr(os, "add_dll_directory") else None
nvrtc = ctypes.CDLL(str(libraries[-1]))

program = ctypes.c_void_p()
source_path = root / "crates" / "search-core" / "src" / "rwkv" / "rwkv7_cuda.cu"
output_path = source_path.with_suffix(".ptx")
source = source_path.read_bytes()

nvrtc.nvrtcCreateProgram.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

status = nvrtc.nvrtcCreateProgram(
    ctypes.byref(program), source, source_path.name.encode(), 0, None, None
)
if status != 0:
    raise SystemExit(f"nvrtcCreateProgram failed: {status}")

try:
    options = (ctypes.c_char_p * 2)(b"--gpu-architecture=compute_52", b"--std=c++14")
    status = nvrtc.nvrtcCompileProgram(program, len(options), options)
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
    if log_size.value > 1:
        log = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(program, log)
        sys.stderr.write(log.value.decode(errors="replace"))
    if status != 0:
        raise SystemExit(f"nvrtcCompileProgram failed: {status}")

    ptx_size = ctypes.c_size_t()
    status = nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
    if status != 0:
        raise SystemExit(f"nvrtcGetPTXSize failed: {status}")
    ptx = ctypes.create_string_buffer(ptx_size.value)
    status = nvrtc.nvrtcGetPTX(program, ptx)
    if status != 0:
        raise SystemExit(f"nvrtcGetPTX failed: {status}")
    output_path.write_bytes(ptx.raw.rstrip(b"\0"))
    print(f"Generated {output_path} ({output_path.stat().st_size} bytes)")
finally:
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
