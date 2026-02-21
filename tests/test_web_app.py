from fastapi.testclient import TestClient

from claimclaw.web_app import create_web_app


def test_web_app_health_endpoint() -> None:
    app = create_web_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_app_root_serves_html() -> None:
    app = create_web_app()
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "ClaimClaw Chat" in response.text


def test_chat_start_returns_session_and_missing_documents() -> None:
    app = create_web_app()
    client = TestClient(app)
    response = client.post("/api/chat/start", json={"thread_name": "Ravi Kumar"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["thread_name"] == "Ravi Kumar"
    assert len(payload["missing_documents"]) == 4
    assert "upload" in payload["assistant_message"].lower()


def test_chat_threads_list_and_open_thread_history() -> None:
    app = create_web_app()
    client = TestClient(app)

    start = client.post("/api/chat/start", json={"thread_name": "Anita Sharma"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    listing = client.get("/api/chat/threads")
    assert listing.status_code == 200
    threads = listing.json()["threads"]
    assert any(t["session_id"] == session_id and t["thread_name"] == "Anita Sharma" for t in threads)

    opened = client.get(f"/api/chat/thread/{session_id}")
    assert opened.status_code == 200
    payload = opened.json()
    assert payload["session_id"] == session_id
    assert payload["thread_name"] == "Anita Sharma"
    assert isinstance(payload["messages"], list)
    assert "age_days" in payload


def test_thread_note_update() -> None:
    app = create_web_app()
    client = TestClient(app)

    start = client.post("/api/chat/start", json={"thread_name": "Nitin Arora"})
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    note = client.post(
        f"/api/chat/thread/{session_id}/note",
        json={"note": "CKD rejection; waiting period dispute. Follow up Monday."},
    )
    assert note.status_code == 200
    payload = note.json()
    assert payload["thread_note"] == "CKD rejection; waiting period dispute. Follow up Monday."

    listing = client.get("/api/chat/threads")
    assert listing.status_code == 200
    threads = listing.json()["threads"]
    item = next(t for t in threads if t["session_id"] == session_id)
    assert item["thread_note"] == "CKD rejection; waiting period dispute. Follow up Monday."
