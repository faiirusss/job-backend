import asyncio
import gc
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger

from app.config import settings
from app.schemas import JobListingDTO, SearchParams

EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]

_DELAYS = {
    "page_load": (2.0, 5.0),
    "click": (1.0, 3.0),
    "keystroke": (0.08, 0.15),
    "pagination": (3.0, 7.0),
}

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.scraper_max_concurrent_browsers)
    return _semaphore


def humanizer_delay(kind: str) -> float:
    lo, hi = _DELAYS[kind]
    return random.uniform(lo, hi)


def random_user_agent() -> str:
    return random.choice(_USER_AGENTS)


async def apply_stealth(page: Any) -> bool:
    """Apply playwright-stealth evasions to ``page`` *before* it navigates.

    Supports playwright-stealth v2 (the ``Stealth`` class, current) and falls
    back to the v1 ``stealth_async`` function. Returns ``True`` iff evasions were
    applied. Failures are non-fatal — we log and continue without stealth.

    History: the prior code only ever did ``from playwright_stealth import
    stealth_async``, which raises ``ImportError`` on v2 (2.x renamed the API to a
    class). That import sat inside a broad ``except`` that logged a warning and
    continued, so stealth was *silently never applied* — which is what let
    Cloudflare's "Just a moment..." challenge wall off the Glints login form.
    """
    try:
        from playwright_stealth import Stealth  # v2 (class-based) API
    except ImportError:
        Stealth = None  # type: ignore[assignment]
    if Stealth is not None:
        try:
            # init_scripts_only: inject evasion scripts that run on the next
            # navigation; don't try to mutate an already-live CDP session.
            await Stealth(init_scripts_only=True).apply_stealth_async(page)
            return True
        except Exception as e:  # noqa: BLE001 — stealth is best-effort
            logger.warning(f"stealth (v2) failed: {e}; continuing without")
            return False
    try:
        from playwright_stealth import stealth_async  # type: ignore  # v1 API

        await stealth_async(page)
        return True
    except Exception as e:  # noqa: BLE001 — stealth is best-effort
        logger.warning(f"stealth unavailable: {e}; continuing without")
        return False


@asynccontextmanager
async def browser_session(portal: str, storage_state: str | None = None) -> AsyncIterator[Any]:
    """Yields a Playwright Page; ensures cleanup.

    When ``storage_state`` is given (a path to a Playwright storage_state JSON),
    the browser context is seeded with that authenticated session — cookies and
    localStorage — so login-gated requests succeed. ``None`` keeps the context
    anonymous (unchanged behavior).
    """
    from playwright.async_api import async_playwright

    sem = _get_semaphore()
    async with sem:
        async with async_playwright() as pw:
            browser = None
            context = None
            try:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                context_kwargs: dict[str, Any] = {
                    "user_agent": random_user_agent(),
                    "viewport": {"width": 1920, "height": 1080},
                }
                if storage_state:
                    context_kwargs["storage_state"] = storage_state
                context = await browser.new_context(**context_kwargs)
                page = await context.new_page()
                await apply_stealth(page)
                yield page
            finally:
                try:
                    if context is not None:
                        await asyncio.wait_for(context.close(), timeout=10)
                except (TimeoutError, Exception) as e:
                    logger.warning(f"{portal}: context close failed: {e}")
                try:
                    if browser is not None:
                        await asyncio.wait_for(browser.close(), timeout=10)
                except (TimeoutError, Exception) as e:
                    logger.warning(f"{portal}: browser close failed: {e}")
                gc.collect()


class BaseScraper(ABC):
    portal: str = ""

    @abstractmethod
    def build_search_url(self, params: SearchParams) -> str: ...

    @abstractmethod
    async def scrape(
        self, page: Any, params: SearchParams, on_event: EventEmitter
    ) -> list[JobListingDTO]: ...
