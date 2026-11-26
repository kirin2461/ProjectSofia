"""GGUF-модели и llama-server: поиск, автовыбор, запуск, контроль.

Модель выбирается под железо: сначала та, что целиком влезает в VRAM,
иначе — крупнейшая, которая помещается в VRAM + половину RAM (частичная
выгрузка слоёв на CPU), иначе — самая маленькая из найденных.
"""

import logging
import os
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
           4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
           10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_AUX_MARKERS = ("mmproj", "mtp", "imatrix", "clip", "vocab", "lora", "adapter")
_BUILD_RE = re.compile(r"b(\d{4,})")


# ═══════════════════════════════════════════════════════════
# GGUF
# ═══════════════════════════════════════════════════════════
def gguf_meta(path: Path, max_pairs: int = 512) -> dict:
    """Прочитать скалярные метаданные GGUF (block_count, architecture, type)."""
    meta: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return meta
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                return meta
            struct.unpack("<Q", f.read(8))  # tensor_count
            (kv_count,) = struct.unpack("<Q", f.read(8))

            def read_str() -> str:
                (n,) = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            for _ in range(min(kv_count, max_pairs)):
                key = read_str()
                (vtype,) = struct.unpack("<I", f.read(4))
                if key.startswith("tokenizer."):
                    break
                if vtype == 8:
                    value = read_str()
                elif vtype == 9:
                    etype, count = struct.unpack("<IQ", f.read(12))
                    if etype == 8:
                        for _ in range(count):
                            read_str()
                    elif etype in _SCALAR:
                        f.seek(_SCALAR[etype][1] * count, 1)
                    else:
                        break
                    value = None
                elif vtype in _SCALAR:
                    fmt, size = _SCALAR[vtype]
                    (value,) = struct.unpack(fmt, f.read(size))
                else:
                    break
                if value is not None:
                    meta[key] = value
    except (OSError, struct.error):
        return meta
    return meta


def _read_model_yml(folder: Path) -> dict:
    path = folder / "model.yml"
    data: dict = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return data
    for line in text.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        data[key.strip()] = value.strip()
    return data


# ═══════════════════════════════════════════════════════════
# Модели
# ═══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ModelInfo:
    path: Path
    name: str
    size_gb: float
    layers: Optional[int] = None
    embedding: bool = False

    @property
    def size_bytes(self) -> int:
        return int(self.size_gb * 1024 ** 3)

    def __str__(self) -> str:  # pragma: no cover - для логов/меню
        layers = f", {self.layers} сл." if self.layers else ""
        return f"{self.name} ({self.size_gb:.1f} ГБ{layers})"


def _is_aux_file(name: str) -> bool:
    low = name.lower()
    return any(marker in low for marker in _AUX_MARKERS)


def scan_models(dirs: Iterable[Path], min_size_gb: float = 0.05) -> List[ModelInfo]:
    """Найти GGUF-модели в указанных каталогах (без mmproj/mtp/imatrix)."""
    found: List[ModelInfo] = []
    seen = set()
    for root in dirs:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in root.rglob("*.gguf"):
            if _is_aux_file(path.name) or path.name.endswith(".tmp"):
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError:
                continue
            size_gb = size_bytes / 1024 ** 3
            if size_gb < min_size_gb:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)

            meta = gguf_meta(path)
            if (meta.get("general.type") or "").lower() not in ("", "model"):
                continue
            arch = str(meta.get("general.architecture") or "")
            if arch in ("clip", "imatrix"):
                continue
            layers = meta.get(f"{arch}.block_count") if arch else None
            if layers is None:
                layers = next(
                    (v for k, v in meta.items()
                     if k.endswith(".block_count") and isinstance(v, int)),
                    None,
                )

            yml = _read_model_yml(path.parent)
            name = yml.get("name") or meta.get("general.name") or path.parent.name
            embedding = (yml.get("embedding", "").lower() == "true"
                         or arch == "bert")
            found.append(
                ModelInfo(path.resolve(), str(name), size_gb,
                          int(layers) if layers else None, embedding)
            )
    return found


def pick_model(models: List[ModelInfo], vram_gb: float, ram_gb: float) -> Optional[ModelInfo]:
    """Лучшая модель под доступную память (см. docstring модуля)."""
    candidates = [m for m in models if not m.embedding]
    if not candidates:
        return None
    headroom = 0.92  # служебные буферы llama.cpp + контекст
    full = [m for m in candidates if m.size_gb <= vram_gb * headroom]
    if full:
        return max(full, key=lambda m: m.size_gb)
    budget = vram_gb + ram_gb * 0.5
    partial = [m for m in candidates if m.size_gb <= budget]
    if partial:
        return max(partial, key=lambda m: m.size_gb)
    return min(candidates, key=lambda m: m.size_gb)


# ═══════════════════════════════════════════════════════════
# llama-server
# ═══════════════════════════════════════════════════════════
def find_llama_server(dirs: Iterable[Path]) -> Optional[Path]:
    """Новейшая сборка llama-server.exe в указанных каталогах."""
    best: Optional[tuple] = None
    for root in dirs:
        root = Path(root)
        if not root.is_dir():
            continue
        for exe in root.rglob("llama-server.exe"):
            match = _BUILD_RE.search(str(exe.parent))
            build = int(match.group(1)) if match else 0
            try:
                mtime = exe.stat().st_mtime
            except OSError:
                mtime = 0.0
            score = (build, mtime)
            if best is None or score > best[0]:
                best = (score, exe)
    return best[1] if best else None


def build_command(
    server: Path,
    model: ModelInfo,
    port: int,
    ctx: int = 8192,
    ngl: object = "auto",
    threads: int = 8,
    alias: Optional[str] = None,
    parallel: int = 1,
) -> List[str]:
    """Аргументы llama-server (зеркало scripts/start_llm_server.ps1)."""
    return [
        str(server),
        "--model", str(model.path),
        "--alias", alias or model.name,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(ctx),
        "--n-gpu-layers", str(ngl),
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--cache-reuse", "256",
        "--threads", str(threads),
        "--batch-size", "1024",
        "--ubatch-size", "512",
        "--parallel", str(parallel),
        "--jinja",
        "--no-webui",
        "--temp", "0.7",
        "--top-p", "0.9",
        "--top-k", "40",
        "--min-p", "0.05",
        "--repeat-penalty", "1.08",
    ]


def server_env() -> dict:
    env = dict(os.environ)
    # Qwen3 и родственные: «размышления» выключаем на уровне сервера.
    # Старая переменная LLAMA_ARG_CHAT_TEMPLATE_KWARGS='{"enable_thinking":
    # false}' устарела — сервер отвечает на неё варнингом и советует
    # --reasoning off. Флаг --reasoning читает LLAMA_ARG_REASONING
    # (рядом в llama-common.dll: LLAMA_ARG_REASONING_EFFORT, _PRESERVE).
    env.setdefault("LLAMA_ARG_REASONING", "off")
    env.setdefault("LLAMA_ARG_FIT", "on")
    return env


def server_up(base_url: str, timeout: float = 2.0) -> bool:
    """llama-server отвечает на /v1/models?"""
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


class LlamaServer:
    """Процесс llama-server: запуск, ожидание готовности, остановка."""

    def __init__(self, server: Path, model: ModelInfo, port: int,
                 ctx: int = 8192, ngl: object = "auto", threads: int = 8,
                 log_file: Optional[Path] = None):
        self.server = Path(server)
        self.model = model
        self.port = port
        self.ctx = ctx
        self.ngl = ngl
        self.threads = threads
        self.log_file = Path(log_file) if log_file else None
        self.process: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> List[str]:
        return build_command(self.server, self.model, self.port,
                             ctx=self.ctx, ngl=self.ngl, threads=self.threads)

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        flags = 0
        for name in ("CREATE_NO_WINDOW", "CREATE_NEW_PROCESS_GROUP"):
            flags |= getattr(subprocess, name, 0)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            out = open(self.log_file, "ab", buffering=0)
        else:
            out = subprocess.DEVNULL
        self.process = subprocess.Popen(
            self.command(),
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            env=server_env(),
            cwd=str(self.server.parent),
        )

    def wait_ready(self, timeout: float = 240.0, interval: float = 0.5) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if server_up(self.base_url):
                return True
            if self.process and self.process.poll() is not None:
                logger.error("llama-server завершился с кодом %s",
                             self.process.returncode)
                return False
            time.sleep(interval)
        return False

    def alive(self) -> bool:
        return bool(self.process and self.process.poll() is None)

    def stop(self, timeout: float = 8.0):
        if not self.process or self.process.poll() is not None:
            self.process = None
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
        except OSError:
            pass
        finally:
            self.process = None


# ═══════════════════════════════════════════════════════════
# Железо
# ═══════════════════════════════════════════════════════════
def _nvidia_smi(field: str) -> Optional[float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def gpu_name() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def free_vram_gb() -> float:
    mb = _nvidia_smi("memory.free")
    return round(mb / 1024, 1) if mb else 0.0


def total_vram_gb() -> float:
    mb = _nvidia_smi("memory.total")
    return round(mb / 1024, 1) if mb else 0.0


def total_ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        return 16.0


def has_nvidia_gpu() -> bool:
    return gpu_name() is not None
