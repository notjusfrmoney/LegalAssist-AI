from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import chat as chat_routes
from app.core.config import Settings
from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse


def test_allowed_origins_parse_comma_separated_and_json_values() -> None:
    comma_settings = Settings(
        _env_file=None,
        allowed_origins_config=(
            "http://localhost:3000, http://127.0.0.1:3000, "
            "http://localhost:3001, http://127.0.0.1:3001"
        ),
    )
    json_settings = Settings(
        _env_file=None,
        allowed_origins_config=(
            '["http://localhost:3000", "http://127.0.0.1:3000", '
            '"http://localhost:3001", "http://127.0.0.1:3001"]'
        ),
    )

    expected = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
    assert comma_settings.allowed_origins == expected
    assert json_settings.allowed_origins == expected


def test_cors_preflight_allows_frontend_origins_and_chat_headers() -> None:
    client = TestClient(app)
    for origin in (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ):
        response = client.options(
            "/api/v1/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert "POST" in response.headers["access-control-allow-methods"]
        assert "OPTIONS" in response.headers["access-control-allow-methods"]
        assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_chat_post_is_reachable_with_frontend_origin(monkeypatch) -> None:
    async def fake_generate_chat_response(
        request: ChatRequest, settings: Settings
    ) -> ChatResponse:
        return ChatResponse(answer="Chat endpoint reached.")

    monkeypatch.setattr(
        chat_routes, "generate_chat_response", fake_generate_chat_response
    )
    response = TestClient(app).post(
        "/api/v1/chat",
        headers={"Origin": "http://localhost:3001"},
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3001"
    assert response.json()["answer"] == "Chat endpoint reached."
