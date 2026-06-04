from app.schemas import SearchParams
from app.scrapers import orchestrator


async def test_linkedin_context_seeded_from_ensure_session(monkeypatch):
    """run_portals must seed the LinkedIn browser context with the storage_state
    that linkedin_auth.ensure_session returns."""
    seen = {}

    async def fake_ensure():
        return "/tmp/linkedin_state.json"

    monkeypatch.setattr(orchestrator.linkedin_auth, "ensure_session", fake_ensure)

    class _Page:
        async def goto(self, *a, **k): ...

    class _CM:
        def __init__(self, portal, storage_state=None):
            seen["portal"] = portal
            seen["storage_state"] = storage_state

        async def __aenter__(self):
            return _Page()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(orchestrator, "browser_session", _CM)

    async def fake_scrape(self, page, params, on_event):
        return []

    monkeypatch.setattr(orchestrator.LinkedInScraper, "scrape", fake_scrape)

    async def on_event(_ev): ...

    await orchestrator.run_portals(["linkedin"], SearchParams(role_keywords=["x"]), on_event)
    assert seen["portal"] == "linkedin"
    assert seen["storage_state"] == "/tmp/linkedin_state.json"
