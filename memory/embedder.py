"""Локальные эмбеддинги для памяти и библиотеки (ONNX, offline).

Модель по умолчанию — multilingual-e5-small (100+ языков, 384 измерения).
Нужна именно многоязычная: встроенная в Chroma all-MiniLM-L6-v2 обучена на
английском и на русских книгах даёт близкий к случайному поиск.

e5 требует префиксы: "query: " для вопроса, "passage: " для фрагмента.
Поэтому эмбеддинги считаются здесь явно, а не через embedding_function Chroma.

Файлы модели: assets/embed/<name>/ (onnx/*.onnx + tokenizer.json).
Если файлов нет — возвращается None, вызывающий код откатывается на прежний путь.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBED_TAG = "e5s"          # суффикс коллекций Chroma
EMBED_DIM = 384
MAX_TOKENS = 512
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def _model_dir() -> Path:
    """Папка с файлами модели: env → внутрь сборки (_internal) → корень."""
    env = os.environ.get("SOFIA_EMBED_DIR")
    if env:
        return Path(env)
    from app.paths import bundled_dir
    return bundled_dir() / "assets" / "embed" / "multilingual-e5-small"


class LocalEmbedder:
    """ONNX-эмбеддер: tokenizer → модель → mean pooling → L2-нормировка."""

    tag = EMBED_TAG

    def __init__(self, model_dir: Optional[Path] = None, threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir) if model_dir else _model_dir()
        tok_path = d / "tokenizer.json"
        onnx_path = self._pick_onnx(d / "onnx")
        if not tok_path.exists() or onnx_path is None:
            raise FileNotFoundError(f"нет файлов модели эмбеддингов в {d}")

        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.no_padding()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads <= 0:
            threads = max(1, min(4, (os.cpu_count() or 4) // 2))
        so.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._lock = threading.Lock()
        self.model_path = str(onnx_path)

    @staticmethod
    def _pick_onnx(onnx_dir: Path) -> Optional[Path]:
        for name in ("model_quantized.onnx", "model.onnx"):
            p = onnx_dir / name
            if p.exists():
                return p
        cands = sorted(onnx_dir.glob("*.onnx")) if onnx_dir.exists() else []
        return cands[0] if cands else None

    @property
    def dim(self) -> int:
        return EMBED_DIM

    def name(self) -> str:
        return f"local-{EMBED_TAG}"

    # ── ядро ────────────────────────────────────────────────
    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        encodings = self._tokenizer.encode_batch(texts)
        max_len = max(1, max(len(e.ids) for e in encodings))
        ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            n = len(enc.ids)
            ids[i, :n] = enc.ids
            mask[i, :n] = 1

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        with self._lock:
            hidden = self._session.run(None, feeds)[0]

        # mean pooling по маске
        m = mask[..., None].astype(np.float32)
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        vecs = summed / counts

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / np.clip(norms, 1e-9, None)).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        return self._encode(prefixed).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode([QUERY_PREFIX + text])[0].tolist()

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        """Совместимость с интерфейсом Chroma embedding_function."""
        return self.embed_documents(input)


_EMBEDDER: Optional[LocalEmbedder] = None
_EMBEDDER_TRIED = False
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Optional[LocalEmbedder]:
    """Ленивая инициализация (один раз на процесс). None — если модели нет."""
    global _EMBEDDER, _EMBEDDER_TRIED
    if _EMBEDDER_TRIED:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER_TRIED:
            return _EMBEDDER
        try:
            _EMBEDDER = LocalEmbedder()
            logger.info("Эмбеддинги: %s (%s)", _EMBEDDER.name(), _EMBEDDER.model_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Локальные эмбеддинги недоступны: %s", e)
            _EMBEDDER = None
        _EMBEDDER_TRIED = True
    return _EMBEDDER
