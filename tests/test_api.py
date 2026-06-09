from pathlib import Path

from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "Aircraft engine RUL prediction API is running" in data["message"]
    assert data["web_console"] == "/app"


def test_console_redirects_to_app():
    response = client.get("/console", follow_redirects=False)

    assert response.status_code in [307, 308]
    assert response.headers["location"] == "/app/"


def test_web_console_file_exists():
    web_file = Path("web/index.html")

    assert web_file.exists()
    assert web_file.is_file()