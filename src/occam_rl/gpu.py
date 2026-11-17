"""Measure system-wide free VRAM, including other Windows/WSL applications."""

import subprocess
import threading


def gpu_info() -> dict:
    """nvidia-smi includes memory used by applications outside this Python process."""
    row = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                                   "--format=csv,noheader,nounits", "-i", "0"], text=True, timeout=10).strip()
    name, total, free, driver = [part.strip() for part in row.split(",")]
    return {"name": name, "total_mib": int(total), "free_mib": int(free), "driver": driver}


class MemoryMonitor:
    """Background samples supplement allocator peaks; sampling cannot prove zero transient peaks."""

    def __init__(self, headroom: int, log, label: str):
        self.headroom, self.log, self.label = headroom, log, label
        self.minimum, self.samples, self.error = float("inf"), 0, None
        self.stop = threading.Event()

    def _sample(self):
        """Record a low-water mark without writing thousands of duplicate samples."""
        while not self.stop.is_set():
            try:
                self.minimum = min(self.minimum, gpu_info()["free_mib"])
                self.samples += 1
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(0.25)

    def __enter__(self):
        self.minimum = gpu_info()["free_mib"]
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.worker.join(timeout=15)
        self.log.emit("gpu_peak", label=self.label, minimum_free_mib=self.minimum,
                      samples=self.samples, required_free_mib=self.headroom, sampling_error=self.error)
        if kind is None:
            self.check()

    def check(self):
        """Fail explicitly when fit is unverified or measured headroom is insufficient."""
        if self.error or self.minimum < self.headroom:
            raise RuntimeError(f"GPU fit failed for {self.label}: free minimum {self.minimum} MiB; {self.error or ''}")
