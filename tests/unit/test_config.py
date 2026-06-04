from app.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("PG_HOST", "test-host")
    monkeypatch.setenv("PG_PORT", "5555")
    monkeypatch.setenv("PG_DB", "testdb")
    monkeypatch.setenv("PG_USER", "tester")
    monkeypatch.setenv("PG_PASSWORD", "pw")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://tester:pw@test-host:5555/testdb")
    monkeypatch.setenv("USE_FAKE_LLM", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("CV_FILES_DIR", "/tmp/jhai-cv")

    s = Settings()
    assert s.pg_host == "test-host"
    assert s.use_fake_llm is True
    assert s.scraper_max_concurrent_browsers == 2  # default
    assert s.cache_ttl_seconds == 21600  # default


def test_settings_real_llm_requires_api_key(monkeypatch):
    monkeypatch.setenv("USE_FAKE_LLM", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    import pytest

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Settings()
