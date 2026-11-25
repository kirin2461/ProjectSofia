"""Подготовка окружения при запуске: каталоги, .env, модель, llama-server.

Всё, что лаунчер делает «сам», собрано здесь и не зависит от Qt, поэтому
покрыто юнит-тестами. Прогресс сообщается через колбэк report(text, percent).
"""

import logging
import os
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app import paths
from app.envfile import read_env, upsert_env
from app.llm import ModelInfo, find_llama_server, pick_model, scan_models

logger = logging.getLogger(__name__)

Report = Callable[[str, Optional[int]], None]

DEFAULT_ENV = {
    "SOFIA_PROFILE": "target",
    "LLM_PROVIDER": "llamacpp",
    "API_HOST": "127.0.0.1",
    "SAFETY_MODE": "standard",
}

# Модель по умолчанию, если на диске нет ни одной GGUF (≈2.3 ГБ).
DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
)
DEFAULT_MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"

LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
LLAMA_FALLBACK_TAG = "b12681"
LLAMA_DOWNLOAD = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"


# ═══════════════════════════════════════════════════════════
# Каталоги и .env
# ═══════════════════════════════════════════════════════════
def prepare_home(base: Path) -> Path:
    """Создать структуру каталогов и .env с безопасными значениями."""
    base = Path(base)
    paths.ensure_dirs(base)
    env_file = paths.env_path(base)
    current = read_env(env_file)
    missing = {k: v for k, v in DEFAULT_ENV.items() if k not in current}
    if missing or not env_file.exists():
        upsert_env(env_file, missing)
    _seed_game_profiles(base)
    return base


def _seed_game_profiles(base: Path):
    dst = base / "data" / "game_profiles"
    if any(dst.glob("*.json")):
        return
    src = paths.bundled_dir() / "data" / "game_profiles"
    for item in src.glob("*.json") if src.is_dir() else []:
        try:
            shutil.copy2(item, dst / item.name)
        except OSError as e:
            logger.warning("Не удалось скопировать профиль %s: %s", item, e)


CHROMA_MODEL = "all-MiniLM-L6-v2"
XTTS_REPO_DIR = "models--tts-hub--XTTS-v2"


def configure_bundled_caches(bundle: Optional[Path] = None,
                             env: Optional[dict] = None) -> List[str]:
    """Направить XTTS и Chroma на модели, упакованные в сборку.

    Тогда первый запуск не требует интернета. Возвращает список
    задействованных кэшей (для журнала).
    """
    bundle = Path(bundle) if bundle else paths.bundled_dir()
    env = os.environ if env is None else env
    used = []

    hub = bundle / "hf-cache" / "hub"
    if (hub / XTTS_REPO_DIR / "snapshots").is_dir() and "HF_HUB_CACHE" not in env:
        env["HF_HUB_CACHE"] = str(hub)
        env.setdefault("HF_HUB_OFFLINE", "1")
        used.append("xtts")

    onnx = bundle / "chroma-cache" / "onnx_models" / CHROMA_MODEL
    if (onnx / "onnx" / "model.onnx").is_file():
        try:
            from chromadb.utils.embedding_functions import onnx_mini_lm_l6_v2 as ef

            user_copy = Path(ef.ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / "onnx" / "model.onnx"
            if not user_copy.is_file():
                ef.ONNXMiniLM_L6_V2.DOWNLOAD_PATH = onnx
                used.append("chroma")
        except ImportError:
            pass
    return used


def load_env_into_process(base: Path) -> Dict[str, str]:
    """Перенести .env в os.environ (не перекрывая уже заданное)."""
    values = read_env(paths.env_path(base))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def build_env_updates(
    provider: str,
    model: str,
    api_port: int,
    base_url: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict[str, str]:
    """Переменные процесса, которыми лаунчер настраивает config.settings."""
    updates = {
        "LLM_PROVIDER": provider,
        "MODEL_BRAIN": model,
        "API_PORT": str(api_port),
        "API_HOST": "127.0.0.1",
    }
    if provider == "ollama":
        updates["OLLAMA_HOST"] = base_url or "http://localhost:11434"
    else:
        updates["MODEL_EXTRACTOR"] = model
        if base_url:
            updates["LLM_BASE_URL"] = base_url
    if model_path:
        updates["SOFIA_MODEL_PATH"] = model_path
    return updates


def pick_threads(cores: Optional[int] = None) -> int:
    """Потоки CPU для llama-server: физические ядра, но не всё железо."""
    if cores is None:
        try:
            import psutil

            cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        except Exception:  # noqa: BLE001
            cores = os.cpu_count() or 4
    return max(1, min(16, cores if cores <= 4 else cores - 1))


# ═══════════════════════════════════════════════════════════
# Модели
# ═══════════════════════════════════════════════════════════
def model_search_dirs(base: Path) -> List[Path]:
    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    dirs = [
        paths.model_dir(base),
        appdata / "Jan" / "data" / "llamacpp" / "models",
        home / ".lmstudio" / "models",
        home / ".cache" / "lm-studio" / "models",
        local / "llama.cpp" / "models",
        home / "models",
    ]
    extra = os.environ.get("SOFIA_MODELS_DIR")
    if extra:
        dirs.insert(0, Path(extra))
    return dirs


def apply_model_choice(base: Path, model: ModelInfo):
    """Запомнить выбранную модель (меню «Модель» в трее)."""
    upsert_env(paths.env_path(base), {"SOFIA_MODEL_PATH": str(model.path)})


def choose_model(base: Path, models: List[ModelInfo], vram_gb: float,
                 ram_gb: float) -> Optional[ModelInfo]:
    saved = read_env(paths.env_path(base)).get("SOFIA_MODEL_PATH")
    if saved:
        saved_path = Path(saved)
        for m in models:
            if _same_path(m.path, saved_path):
                return m
        if saved_path.is_file():
            extra = scan_models([saved_path.parent])
            for m in extra:
                if _same_path(m.path, saved_path):
                    return m
        logger.warning("Сохранённая модель недоступна (%s) — автовыбор", saved)
    return pick_model(models, vram_gb=vram_gb, ram_gb=ram_gb)


def _same_path(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a).lower() == str(b).lower()


def ensure_model(base: Path, vram_gb: float, ram_gb: float,
                 report: Report) -> ModelInfo:
    report("Поиск моделей GGUF…", None)
    models = scan_models(model_search_dirs(base))
    model = choose_model(base, models, vram_gb, ram_gb)
    if model:
        return model
    report("Моделей нет — скачиваю Qwen3-4B (≈2.3 ГБ, один раз)…", 0)
    target = paths.model_dir(base) / "Qwen3-4B" / DEFAULT_MODEL_FILE
    download(DEFAULT_MODEL_URL, target, report, "Загрузка модели")
    models = scan_models([target.parent])
    if not models:
        raise RuntimeError(f"Скачанная модель не читается: {target}")
    return models[0]


# ═══════════════════════════════════════════════════════════
# llama-server
# ═══════════════════════════════════════════════════════════
def server_search_dirs(base: Path) -> List[Path]:
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    dirs = [paths.bin_dir(base), paths.bundled_dir() / "bin", local / "llama.cpp"]
    extra = os.environ.get("SOFIA_LLAMA_DIR")
    if extra:
        dirs.insert(0, Path(extra))
    return dirs


def ensure_llama_server(base: Path, nvidia: bool, report: Report) -> Path:
    report("Поиск llama-server…", None)
    found = find_llama_server(server_search_dirs(base))
    if found:
        return found
    flavour = "cuda-12.4-x64" if nvidia else "vulkan-x64"
    report(f"llama.cpp не найден — скачиваю сборку {flavour}…", 0)
    tag, assets = _latest_llama_assets(flavour, nvidia)
    target = paths.bin_dir(base) / f"llama-{tag}"
    target.mkdir(parents=True, exist_ok=True)
    for name in assets:
        archive = paths.bin_dir(base) / name
        download(LLAMA_DOWNLOAD.format(tag=tag, name=name), archive, report,
                 f"Загрузка {name}")
        report(f"Распаковка {name}…", None)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        archive.unlink(missing_ok=True)
    found = find_llama_server([target])
    if not found:
        raise RuntimeError("В архиве llama.cpp нет llama-server.exe")
    return found


def _latest_llama_assets(flavour: str, nvidia: bool):
    """Тег свежей сборки и имена нужных архивов (с fallback на известный тег)."""
    import json

    want = re.compile(rf"^llama-(b\d+)-bin-win-{re.escape(flavour)}\.zip$")
    try:
        req = urllib.request.Request(LLAMA_RELEASES_API,
                                     headers={"User-Agent": "Sofia-Launcher"})
        with urllib.request.urlopen(req, timeout=20) as r:
            releases = json.load(r)
        for rel in releases:
            names = [a.get("name", "") for a in rel.get("assets", [])]
            for name in names:
                m = want.match(name)
                if not m:
                    continue
                assets = [name]
                cudart = "cudart-llama-bin-win-cuda-12.4-x64.zip"
                if nvidia and cudart in names:
                    assets.append(cudart)
                return m.group(1), assets
    except Exception as e:  # noqa: BLE001
        logger.warning("GitHub API недоступен (%s) — беру %s", e, LLAMA_FALLBACK_TAG)
    tag = LLAMA_FALLBACK_TAG
    assets = [f"llama-{tag}-bin-win-{flavour}.zip"]
    if nvidia:
        assets.append("cudart-llama-bin-win-cuda-12.4-x64.zip")
    return tag, assets


# ═══════════════════════════════════════════════════════════
# Загрузка файлов (с докачкой)
# ═══════════════════════════════════════════════════════════
def download(url: str, target: Path, report: Report, label: str,
             chunk: int = 1 << 20) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Sofia-Launcher"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0  # сервер не умеет докачку — начинаем заново
        total = int(r.headers.get("Content-Length") or 0) + have
        mode = "ab" if have else "wb"
        done = have
        last_pct = -1
        with open(part, mode) as f:
            while True:
                block = r.read(chunk)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last_pct:
                        last_pct = pct
                        report(f"{label}: {done / 2**30:.2f} / {total / 2**30:.2f} ГБ", pct)
    if total and done < total:
        raise RuntimeError(f"Загрузка прервана: {done} из {total} байт")
    os.replace(part, target)
    return target
