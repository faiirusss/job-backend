from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "jhai"
    pg_user: str = "jhai"
    pg_password: str = "jhai_dev_password"

    database_url: str = Field(
        default="postgresql+asyncpg://jhai:jhai_dev_password@localhost:5432/jhai"
    )

    # LLM provider selection. Explicit `llm_provider` (fake|gemini|qwen) is the
    # source of truth; `use_fake_llm` is kept for back-compat — when the provider
    # is left at its "fake" default but USE_FAKE_LLM=false, the resolver falls
    # back to "gemini" so existing deployments keep working unchanged.
    llm_provider: str = "fake"
    use_fake_llm: bool = True
    llm_fallback_providers: str = ""
    # When a provider hits a rate/quota limit it is "tripped" and skipped for
    # this many seconds instead of being re-retried on every call. Covers
    # RPM/TPM recovery; for daily (RPD) exhaustion it re-probes ~once per window.
    llm_provider_cooldown_seconds: int = 300

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_rpm_limit: int = 15

    # Qwen served over any OpenAI-compatible endpoint — OpenRouter (the default
    # base URL) or Alibaba Cloud Model Studio (DashScope compatible-mode), set
    # via QWEN_BASE_URL. Free tiers are heavily rate-limited, hence the
    # conservative RPM and the pipeline's retry/degradation behavior.
    qwen_api_key: str = ""
    qwen_model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    qwen_rpm_limit: int = 15
    qwen_base_url: str = "https://openrouter.ai/api/v1"
    # Qwen3 "thinking" toggle (Alibaba Model Studio): False strips the
    # reasoning_content step, cutting completion tokens ~10x and latency. Leave
    # unset (None) for endpoints that don't accept the param, e.g. OpenRouter.
    qwen_enable_thinking: bool | None = None

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    scraper_max_concurrent_browsers: int = 2
    scraper_timeout_seconds: int = 45
    cache_ttl_seconds: int = 21600

    # Glints service-account session. Pages 2+ of Glints search are login-gated
    # (HTTP 403 NO_PERMISSION); the backend reuses one authenticated session,
    # materialized as a Playwright storage_state, and re-mints it from these
    # credentials when it expires. All optional: with none set, Glints scraping
    # falls back to anonymous page-1 (30 jobs).
    glints_email: str = ""
    glints_password: str = ""
    glints_storage_state_path: str = "./data/glints_state.json"

    # LinkedIn authenticated session (Voyager API). Mirrors the Glints session
    # pattern: headless login mints a Playwright storage_state (cookies incl.
    # li_at + JSESSIONID) reused across searches and re-minted on expiry. All
    # optional: with none set, LinkedIn scraping stays on the anonymous guest
    # endpoint. NOTE: authenticated Voyager use risks account checkpoint/ban.
    linkedin_email: str = ""
    linkedin_password: str = ""
    linkedin_storage_state_path: str = "./data/linkedin_state.json"

    cv_files_dir: str = "./data/cv_files"

    @property
    def resolved_llm_provider(self) -> str:
        """The effective provider, applying the USE_FAKE_LLM back-compat rule.

        An explicit non-default ``llm_provider`` always wins. Otherwise (provider
        left at "fake") ``USE_FAKE_LLM=false`` is honored as a request for the
        legacy real provider, Gemini.
        """
        provider = self.llm_provider.lower()
        if provider != "fake":
            return provider
        return "fake" if self.use_fake_llm else "gemini"

    @property
    def resolved_llm_fallback_providers(self) -> list[str]:
        """Comma-separated fallback provider chain, excluding the primary.

        Example: ``LLM_PROVIDER=gemini`` and
        ``LLM_FALLBACK_PROVIDERS=qwen,fake`` means try Gemini first, switch to
        Qwen on quota/rate-limit errors, then FakeLLM as the final offline
        fallback if Qwen is also limited.
        """
        primary = self.resolved_llm_provider
        providers: list[str] = []
        seen = {primary}
        for raw in self.llm_fallback_providers.split(","):
            provider = raw.strip().lower()
            if not provider or provider in seen:
                continue
            seen.add(provider)
            providers.append(provider)
        return providers

    @model_validator(mode="after")
    def _check_llm_keys(self) -> "Settings":
        allowed = {"fake", "gemini", "qwen"}
        provider_chain = [self.resolved_llm_provider, *self.resolved_llm_fallback_providers]
        unknown = [p for p in provider_chain if p not in allowed]
        if unknown:
            raise ValueError(f"Unsupported LLM provider(s): {', '.join(unknown)}")

        if "gemini" in provider_chain and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when the LLM provider is gemini")
        if "qwen" in provider_chain and not self.qwen_api_key:
            raise ValueError("QWEN_API_KEY must be set when qwen is used as provider or fallback")
        return self


settings = Settings()
