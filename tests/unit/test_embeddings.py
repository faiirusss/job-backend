import pytest

from app.ai.embeddings import EmbeddingService


@pytest.mark.asyncio
async def test_encode_returns_correct_shape():
    svc = EmbeddingService()
    svc.load()
    vectors = await svc.encode(["hello world", "another sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert all(isinstance(v, float) for v in vectors[0])


@pytest.mark.asyncio
async def test_encode_empty_list():
    svc = EmbeddingService()
    svc.load()
    vectors = await svc.encode([])
    assert vectors == []
