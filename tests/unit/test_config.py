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


def test_resolved_provider_defaults_to_fake(monkeypatch):
    monkeypatch.setenv("USE_FAKE_LLM", "true")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert Settings().resolved_llm_provider == "fake"


def test_use_fake_llm_false_back_compat_resolves_to_gemini(monkeypatch):
    # Old-style config: provider left at its "fake" default, only USE_FAKE_LLM toggled.
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("USE_FAKE_LLM", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert Settings().resolved_llm_provider == "gemini"


def test_explicit_qwen_provider_resolves_to_qwen(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "q-key")
    assert Settings().resolved_llm_provider == "qwen"


def test_qwen_provider_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "")
    import pytest

    with pytest.raises(ValueError, match="QWEN_API_KEY"):
        Settings()


def test_fallback_provider_chain_excludes_primary_and_duplicates(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("QWEN_API_KEY", "q-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDERS", "qwen, gemini, qwen")

    s = Settings()

    assert s.resolved_llm_provider == "gemini"
    assert s.resolved_llm_fallback_providers == ["qwen"]


def test_provider_cooldown_seconds_default_and_override(monkeypatch):
    monkeypatch.setenv("USE_FAKE_LLM", "true")
    assert Settings().llm_provider_cooldown_seconds == 300  # default

    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_SECONDS", "60")
    assert Settings().llm_provider_cooldown_seconds == 60


def test_qwen_fallback_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDERS", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "")
    import pytest

    with pytest.raises(ValueError, match="QWEN_API_KEY"):
        Settings()
