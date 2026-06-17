import uuid

from fastapi.testclient import TestClient


def _make_client(monkeypatch, db_engine) -> TestClient:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    import app.api.deps as deps_mod
    import app.db as db_mod
    import app.ws.search as ws_search_mod

    db_mod.engine = db_engine
    db_mod.SessionLocal = async_sessionmaker(db_engine, expire_on_commit=False)
    deps_mod.SessionLocal = db_mod.SessionLocal
    ws_search_mod.SessionLocal = db_mod.SessionLocal
    from app.main import app

    return TestClient(app)


def _register(client: TestClient, email: str = "user@example.com") -> None:
    r = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123", "name": "User"},
    )
    assert r.status_code == 201


def test_auth_cookie_session_flow(monkeypatch, db_engine):
    with _make_client(monkeypatch, db_engine) as client:
        assert client.get("/api/v1/auth/me").status_code == 401

        _register(client)
        me = client.get("/api/v1/auth/me")
        assert me.status_code == 200
        assert me.json()["email"] == "user@example.com"

        logout = client.post("/api/v1/auth/logout")
        assert logout.status_code == 204
        assert client.get("/api/v1/auth/me").status_code == 401

        login = client.post(
            "/api/v1/auth/login",
            json={"email": "user@example.com", "password": "password123"},
        )
        assert login.status_code == 200
        assert client.get("/api/v1/auth/me").status_code == 200


def test_conversation_crud_and_general_message(monkeypatch, db_engine):
    with _make_client(monkeypatch, db_engine) as client:
        assert client.get("/api/v1/conversations").status_code == 401
        _register(client)

        created = client.post("/api/v1/conversations", json={"title": "Laravel Bandung"})
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        uuid.UUID(conversation_id)

        listed = client.get("/api/v1/conversations")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == conversation_id

        message = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "hi"},
        )
        assert message.status_code == 200
        body = message.json()
        assert body["action"] == "general_chat"
        assert body["conversation_id"] == conversation_id
        assert body["query_id"] is None
        assert body["user_message"]["conversation_id"] == conversation_id
        assert body["assistant_message"]["conversation_id"] == conversation_id

        renamed = client.patch(
            f"/api/v1/conversations/{conversation_id}", json={"title": "Laravel Jakarta"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Laravel Jakarta"

        detail = client.get(f"/api/v1/conversations/{conversation_id}")
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["conversation"]["id"] == conversation_id
        assert len(detail_body["messages"]) == 2

        deleted = client.delete(f"/api/v1/conversations/{conversation_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404


def test_conversation_refinement_merges_previous_search_params(monkeypatch, db_engine):
    async def _noop_pipeline(*args, **kwargs):
        return None

    monkeypatch.setattr("app.api.conversations.search_service.run_pipeline", _noop_pipeline)

    with _make_client(monkeypatch, db_engine) as client:
        _register(client)

        created = client.post("/api/v1/conversations", json={"title": "Chat baru"})
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        uuid.UUID(conversation_id)

        first = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "cari kerjaan laravel di bandung"},
        )
        assert first.status_code == 200
        assert first.json()["action"] == "new_search"

        refined = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "tolong lokasinya ganti jakarta"},
        )
        assert refined.status_code == 200
        body = refined.json()
        assert body["action"] == "refine_search"
        assert body["conversation_id"] == conversation_id
        assert body["query_id"] is not None
        uuid.UUID(body["conversation_id"])
        uuid.UUID(body["query_id"])
        uuid.UUID(body["assistant_message"]["search_query_id"])

        params = body["assistant_message"]["metadata"]["params"]
        assert params["role_keywords"] == ["laravel"]
        assert params["location"] == ["Jakarta"]
