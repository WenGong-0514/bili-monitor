#!/usr/bin/env python3
"""GPU 会话管理：懒加载 + 用完自动释放显存（已在本机 WSL2 + RTX 2080Ti 22G 实测通过）。

实测数据（容器内，真实驱动路径）
--------------------------------
  纯 CUDA 路径： ctx+2GB 分配 -> free + cuCtxDestroy -> 17183MiB（基线 17182，残留 1MB）
  torch  路径： 分配 1GB(18386) -> del+gc+empty_cache -> 17362（归还 1024MB）
                            -> cudaDeviceReset -> 17183（再回收上下文 179MB）
                            且 reset 之后可再次分配/复用 GPU

设计要点
--------
* 懒加载：第一次真正要用时才建模型，不是进程一起来就占着卡；
* 任务后：`empty_cache()` 立刻把本任务缓存块还给驱动；
* 空闲后：卸载模型 + `cudaDeviceReset()` 销毁 CUDA 上下文，把上下文占用也还回去；
* 并发安全：`with session()` 引用计数，正在用时绝不释放；释放与加载互斥；
* 释放失败不影响主流程（全部吞异常并写日志）。

嵌入 bili_monitor.py 的最小改法
-------------------------------
    # 1) 顶部
    from gpu_release import GpuSession

    # 2) 把原来的 _get_sensevoice_model() 里的 AutoModel(...) 抽成 loader
    def _load_sensevoice(use_gpu: bool):
        import torch, tf_keras  # noqa
        from funasr import AutoModel
        torch.set_num_threads(ASR_LOCAL_THREADS)
        torch.backends.mkldnn.enabled = True
        return AutoModel(model=ASR_LOCAL_MODEL, vad_model=ASR_LOCAL_VAD_MODEL,
                         vad_kwargs={"max_single_segment_time": ASR_LOCAL_MAX_SEG_MS},
                         disable_update=True,
                         device="cuda" if use_gpu else "cpu",
                         disable_pbar=True)

    _gpu = GpuSession(loader=_load_sensevoice,
                      idle_seconds=ASR_GPU_IDLE_RELEASE_SEC,   # 见下方配置
                      allow_gpu=(ASR_LOCAL_DEVICE in ("auto", "cuda")),
                      log=print)

    # 3) 原来的单例获取改成一行
    def _get_sensevoice_model():
        return _gpu.acquire()

    # 4) 一次转写结束（拿到音频结果之后）加一句
    _gpu.after_task()

也可以直接用 with：
    with _gpu.session() as model:
        res = model.generate(...)

配置（config.yaml 的 asr 段）
-----------------------------
    asr:
      model: iic/SenseVoiceSmall
      device: auto           # auto/cuda=允许 GPU；cpu=只用 CPU
      gpu_idle_release_sec: 1800   # 空闲多少秒后彻底释放显存（<=0 表示只在任务后清缓存）

读取方式：
    ASR_GPU_IDLE_RELEASE_SEC = int(_ASR_CFG.get("gpu_idle_release_sec", 1800))

自检（在容器里跑）
------------------
    /usr/local/bin/python gpu_release.py --selftest          # 上下文 + 自动释放
    /usr/local/bin/python gpu_release.py --selftest --model  # 额外加载 funasr 实测
"""
from __future__ import annotations

import atexit
import ctypes
import gc
import glob
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

__all__ = ["GpuSession", "device_memory_mb", "release_cuda_context", "cuda_available"]


# --------------------------------------------------------------------------- #
# 底层工具
# --------------------------------------------------------------------------- #
def _nvidia_smi(*fields: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        return (out.stdout or out.stderr).strip()
    except Exception:  # noqa: BLE001
        return ""


def device_memory_mb() -> tuple:
    """(used_mb, free_mb)；取不到时返回 (-1, -1)。"""
    raw = _nvidia_smi("memory.used", "memory.free")
    try:
        used, free = [int(x.strip()) for x in raw.split(",")[:2]]
        return used, free
    except Exception:  # noqa: BLE001
        return -1, -1


def _libcuda_path() -> str | None:
    """WSL 下 libcuda.so.1 的真实位置在 /usr/lib/wsl/drivers/<驱动>/ 下。"""
    for pat in ("/usr/lib/wsl/drivers/*/libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return "libcuda.so.1"


def _cudart_path() -> str | None:
    for pat in ("/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
                "/usr/lib/x86_64-linux-gnu/libcudart.so*",
                "/usr/lib/wsl/lib/libcudart.so*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def cuda_available() -> bool:
    """不惊动 torch，只用驱动 API 判断 CUDA 是否可用。"""
    try:
        lib = ctypes.CDLL(_libcuda_path())
        if lib.cuInit(0) != 0:
            return False
        n = ctypes.c_int(0)
        if lib.cuDeviceGetCount(ctypes.byref(n)) != 0:
            return False
        return n.value > 0
    except Exception:  # noqa: BLE001
        return False


def release_cuda_context(log=print, force_reset: bool = False) -> None:
    """把本进程的显存尽量还给驱动。

    重要（本机实测教训）：
      `cudaDeviceReset()` 会直接销毁 CUDA 上下文，但 torch 的分配器并不知道，
      之后任何 torch tensor 析构都会触发 `invalid device pointer` -> SIGABRT（进程崩）。
      所以：**只要当前进程里已经 import 过 torch，就绝不能在进程存活期间调用它**。
      留出的两个安全出口：
        * force_reset=True（只在确认马上要退出进程时用，如 atexit）；
        * 不用 torch 的纯 CUDA 程序可以直接 reset。
    """
    has_torch = "torch" in sys.modules
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            # 连续两次：PyTorch 的异步分配器在 free 之后要再走一次 empty_cache 才真正把块还给驱动
            for i in (1, 2):
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.2)
            log("    [GPU释放] torch.empty_cache x2 完成 (allocated=%.1fMB reserved=%.1fMB)"
                % (torch.cuda.memory_allocated() / 2**20,
                   torch.cuda.memory_reserved() / 2**20))
    except Exception as exc:  # noqa: BLE001
        log(f"    [GPU释放] torch 缓存清理跳过: {type(exc).__name__}: {exc}")

    gc.collect()

    if has_torch and not force_reset:
        log("    [GPU释放] 检测到 torch 已加载 -> 跳过 cudaDeviceReset"
            "（避免分配器失效导致进程 abort）；模型缓存已由 empty_cache 归还")
    else:
        cudart = _cudart_path()
        if cudart:
            try:
                rc = ctypes.CDLL(cudart).cudaDeviceReset()
                log(f"    [GPU释放] cudaDeviceReset rc={rc}")
            except Exception as exc:  # noqa: BLE001
                log(f"    [GPU释放] cudaDeviceReset 失败: {type(exc).__name__}: {exc}")
        else:
            log("    [GPU释放] 未找到 libcudart，跳过上下文销毁")

    gc.collect()
    time.sleep(0.3)
    used, free = device_memory_mb()
    if used >= 0:
        log(f"    [GPU释放] 释放后设备显存 used={used}MB free={free}MB")


# --------------------------------------------------------------------------- #
# 会话管理
# --------------------------------------------------------------------------- #
class GpuSession:
    """懒加载 + 任务后清缓存 + 空闲自动释放的模型会话。"""

    def __init__(self, loader, idle_seconds: int = 1800, allow_gpu: bool = True,
                 release_context_on_idle: bool = True, log=print):
        self._loader = loader
        self._idle_seconds = int(idle_seconds)
        self._allow_gpu = bool(allow_gpu)
        self._release_ctx = bool(release_context_on_idle)
        self._log = log

        self._lock = threading.RLock()
        self._release_lock = threading.Lock()
        self._model = None
        self._use_gpu = False
        self._active = 0
        self._last_used = time.time()

        if self._idle_seconds > 0:
            threading.Thread(target=self._idle_watchdog, daemon=True,
                             name="gpu-idle-release").start()
        atexit.register(self._atexit_release)

    # ---------------- 属性 ----------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return "cuda" if self._use_gpu else "cpu"

    @property
    def in_use(self) -> int:
        return self._active

    # ---------------- 取用 ----------------
    def acquire(self):
        """取模型（不存在则懒加载）。调用方用完请调用 after_task()。"""
        with self._lock:
            if self._model is None:
                self._load_locked()
            self._active += 1
            self._last_used = time.time()
            return self._model

    @contextmanager
    def session(self):
        """with 用法：退出时自动 after_task()。"""
        model = self.acquire()
        try:
            yield model
        finally:
            self.after_task()

    def after_task(self) -> None:
        """一个任务(一个视频)处理完调用：立刻把本任务的显存还给驱动。"""
        with self._lock:
            self._active = max(0, self._active - 1)
            self._last_used = time.time()
            still_busy = self._active > 0
        if still_busy:
            return
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                used, free = device_memory_mb()
                self._log(f"    [GPU] 任务结束已清缓存, 设备显存 used={used}MB free={free}MB")
        except Exception as exc:  # noqa: BLE001
            self._log(f"    [GPU] empty_cache 跳过: {type(exc).__name__}: {exc}")

    # ---------------- 释放 ----------------
    def release(self, reason: str = "manual", force_reset: bool = False) -> bool:
        """卸载模型并归还显存；有任务在用则拒绝。

        force_reset=True 时才会调用 cudaDeviceReset（仅限即将退出进程的场景）。
        """
        with self._lock:
            if self._active > 0:
                self._log(f"    [GPU] 有 {self._active} 个任务在用，跳过释放({reason})")
                return False
            model = self._model
            self._model = None
            self._use_gpu = False
        with self._release_lock:
            if model is not None:
                self._log(f"    [GPU] 释放: 卸载模型 ({reason})")
                del model
            if self._release_ctx:
                release_cuda_context(log=self._log, force_reset=force_reset)
            else:
                gc.collect()
        return True

    def _load_locked(self) -> None:
        with self._release_lock:
            use_gpu = self._allow_gpu and cuda_available()
            self._log(f"    [GPU] 加载模型 (use_gpu={use_gpu}) ...")
            t0 = time.time()
            self._model = self._loader(use_gpu)
            self._use_gpu = use_gpu
            self._last_used = time.time()
            used, free = device_memory_mb()
            self._log(f"    [GPU] 模型加载完成 {time.time()-t0:.1f}s, "
                      f"设备显存 used={used}MB free={free}MB")

    def _idle_watchdog(self) -> None:
        interval = min(30, max(5, self._idle_seconds // 4)) if self._idle_seconds > 0 else 30
        while True:
            time.sleep(interval)
            with self._lock:
                if self._model is None or self._active > 0:
                    continue
                idle = time.time() - self._last_used
            if idle >= self._idle_seconds:
                self._log(f"    [GPU] 空闲 {int(idle)}s >= {self._idle_seconds}s，自动释放显存")
                self.release(reason="idle-timeout")

    def _atexit_release(self) -> None:
        try:
            if self._model is not None:
                # 进程即将退出，可以安全地销毁上下文（强制 reset 把残留也还掉）
                self.release(reason="atexit", force_reset=True)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _selftest(with_model: bool) -> int:
    print("== GPU 显存释放自检 ==", flush=True)
    print("  cuda_available() =", cuda_available(), flush=True)
    print("  1.baseline         used/free =", device_memory_mb(), flush=True)

    def loader(use_gpu: bool):
        if not use_gpu:
            return {"device": "cpu"}
        if with_model:
            from funasr import AutoModel  # noqa: PLC0415
            return AutoModel(model="iic/SenseVoiceSmall", disable_update=True,
                             device="cuda", disable_pbar=True)
        import torch  # noqa: PLC0415
        return {"device": "cuda", "buf": torch.zeros(256 * 1024 * 1024,
                                                     dtype=torch.float32, device="cuda")}

    def run_one(session, tag):
        """模拟真实调用：acquire -> 使用 -> 丢掉引用 -> after_task。"""
        model = session.acquire()
        print("  %s 获取后     used/free = %s" % (tag, device_memory_mb()), flush=True)
        if isinstance(model, dict) and "buf" in model:
            model["buf"].add_(1)
        return model

    s = GpuSession(loader=loader, idle_seconds=8, log=print)

    m = run_one(s, "2.")
    del m                       # 调用方不再持有引用（真实场景里模型只在 session 内被引用）
    gc.collect()
    s.after_task()
    print("  3.任务后(引用已丢) used/free =", device_memory_mb(), flush=True)

    print("  等待 8s 空闲超时自动释放 ...", flush=True)
    time.sleep(12)
    print("  4.空闲释放后   used/free =", device_memory_mb(), flush=True)
    print("  加载状态 loaded =", s.loaded, flush=True)

    m2 = run_one(s, "5.")
    del m2
    gc.collect()
    s.after_task()
    s.release(reason="selftest-end")
    print("  6.再次释放后   used/free =", device_memory_mb(), flush=True)
    print("\n  期望：第 4/6 步回到第 1 步附近（差距 < 50MB 视为成功）", flush=True)
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", action="store_true", help="自检时加载 funasr 模型")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest(a.model))
    print(__doc__)
