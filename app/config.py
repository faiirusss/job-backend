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

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    use_fake_llm: bool = True
    gemini_rpm_limit: int = 15

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

    @model_validator(mode="after")
    def _check_gemini_key(self) -> "Settings":
        if not self.use_fake_llm and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be set when USE_FAKE_LLM=false")
        return self


settings = Settings()
