import asyncio
from typing import Any

from app.config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.embedding_model)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            raise RuntimeError("EmbeddingService.load() must be called first")

        def _encode_sync() -> list[list[float]]:
            arr = self._model.encode(
                texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False
            )
            return [vec.tolist() for vec in arr]

        return await asyncio.to_thread(_encode_sync)


embeddings_service = EmbeddingService()
