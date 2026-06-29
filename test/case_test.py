import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
import json

from main import fastapi_app
from database import get_session
from models import Case, Alert

# Setup in-memory SQLite for testing
sqlite_url = "sqlite:///:memory:"
engine = create_engine(
    sqlite_url, 
    connect_args={"check_same_thread": False}, 
    poolclass=StaticPool
)

@pytest.fixture(name="session")
def session_fixture():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session

    from auth import get_current_user
    def get_current_user_override():
        return {
            "user_id": 1,
            "role": "employee",
            "token": "mock-token"
        }

    fastapi_app.dependency_overrides[get_session] = get_session_override
    fastapi_app.dependency_overrides[get_current_user] = get_current_user_override
    client = TestClient(fastapi_app)
    yield client
    fastapi_app.dependency_overrides.clear()

class TestCases:

    # ---------------------------------------------------------
    # GET /api/cases
    # ---------------------------------------------------------
    def test_list_cases_empty(self, client: TestClient):
        response = client.get("/api/cases")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_cases_data_mapping(self, client: TestClient, session: Session):
        # Create a mock case
        case = Case(
            case_result_status="pending",
            context="This is a very long context string that should be truncated because it exceeds the 40 character limit.",
            urls_json=json.dumps(["https://example.com"]),
            pdf_paths_json=json.dumps(["document.pdf"]),
            image_paths_json=json.dumps(["photo.jpg"]),
            user_id=1
        )
        session.add(case)
        session.commit()

        response = client.get("/api/cases")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        
        # Check mapping
        assert data[0]["case_result_status"] == "active"  # "pending" mapped to "active"
        assert len(data[0]["name"]) == 43 # 40 chars + "..."
        assert data[0]["name"].endswith("...")
        assert data[0]["canResolve"] is False
        
        # Check vault mapping
        vault = data[0]["vault"]
        assert len(vault) == 3
        assert vault[0]["type"] == "url"
        assert vault[0]["url"] == "https://example.com"
        assert vault[1]["type"] == "pdf"
        assert vault[1]["name"] == "document.pdf"
        assert vault[2]["type"] == "image"
        assert vault[2]["name"] == "photo.jpg"

    def test_list_cases_can_resolve_true(self, client: TestClient, session: Session):
        case = Case(status="pending", user_id=1)
        session.add(case)
        session.commit()

        alert = Alert(
            case_id=case.id,
            title="Test Alert",
            summary="Test Summary",
            ai_reasoning="Test Reasoning",
            content="Test", 
            review_status="accepted"
        )
        session.add(alert)
        session.commit()

        response = client.get("/api/cases")
        assert response.status_code == 200
        data = response.json()
        assert data[0]["canResolve"] is True

    def test_list_cases_can_resolve_false(self, client: TestClient, session: Session):
        case = Case(status="pending", user_id=1)
        session.add(case)
        session.commit()

        # Alert exists but not accepted
        alert = Alert(
            case_id=case.id, 
            title="Test Alert",
            summary="Test Summary",
            ai_reasoning="Test Reasoning",
            content="Test", 
            review_status="pending"
        )
        session.add(alert)
        session.commit()

        response = client.get("/api/cases")
        assert response.status_code == 200
        data = response.json()
        assert data[0]["canResolve"] is False

    # ---------------------------------------------------------
    # POST /api/cases/{case_id}/reindex
    # ---------------------------------------------------------
    def test_reindex_case_success(self, client: TestClient, session: Session):
        case = Case(
            urls_json=json.dumps(["http://test.com"]),
            pdf_paths_json=json.dumps(["test.pdf"]),
            user_id=1
        )
        session.add(case)
        session.commit()

        response = client.post(f"/api/cases/{case.id}/reindex")
        assert response.status_code == 200
        assert response.json()["pdfs"] == 1
        assert response.json()["urls"] == 1

    def test_reindex_case_no_files(self, client: TestClient, session: Session):
        case = Case(user_id=1)
        session.add(case)
        session.commit()

        response = client.post(f"/api/cases/{case.id}/reindex")
        assert response.status_code == 200
        assert response.json()["total_files"] == 0

    def test_reindex_case_not_found(self, client: TestClient):
        response = client.post("/api/cases/999/reindex")
        assert response.status_code == 404

    # ---------------------------------------------------------
    # POST /api/cases/{case_id}/add-url
    # ---------------------------------------------------------
    def test_add_url_to_case_success(self, client: TestClient, session: Session, monkeypatch):
        # Mock background task queuing
        def mock_enqueue(*args, **kwargs):
            pass
        monkeypatch.setattr("ai.background.enqueue_research", mock_enqueue)

        case = Case(status="complete", user_id=1)
        session.add(case)
        session.commit()

        response = client.post(f"/api/cases/{case.id}/add-url", json={"url": "http://new.com"})
        assert response.status_code == 200

        session.refresh(case)
        assert "http://new.com" in case.urls_json
        assert case.status == "pending"

    def test_add_url_case_not_found(self, client: TestClient):
        response = client.post("/api/cases/999/add-url", json={"url": "http://new.com"})
        assert response.status_code == 404

    # ---------------------------------------------------------
    # POST /api/cases/{case_id}/add-context
    # ---------------------------------------------------------
    def test_add_context_to_case_success(self, client: TestClient, session: Session, monkeypatch):
        def mock_enqueue(*args, **kwargs):
            pass
        monkeypatch.setattr("ai.background.enqueue_research", mock_enqueue)

        case = Case(context="Old context", status="complete", user_id=1)
        session.add(case)
        session.commit()

        response = client.post(f"/api/cases/{case.id}/add-context", json={"context": "New notes"})
        assert response.status_code == 200

        session.refresh(case)
        assert case.context == "Old context\n\nNew notes"
        assert case.status == "pending"

    def test_add_context_case_not_found(self, client: TestClient):
        response = client.post("/api/cases/999/add-context", json={"context": "New notes"})
        assert response.status_code == 404

    # ---------------------------------------------------------
    # PATCH /api/cases/{case_id}/status
    # ---------------------------------------------------------
    def test_update_case_status_success(self, client: TestClient, session: Session):
        case = Case(case_result_status="pending", user_id=1)
        session.add(case)
        session.commit()

        alert = Alert(
            case_id=case.id,
            title="Test Alert",
            summary="Test Summary",
            ai_reasoning="Test Reasoning",
            content="Test", 
            review_status="accepted"
        )
        session.add(alert)
        session.commit()

        response = client.patch(
            f"/api/cases/{case.id}/status", 
            json={"case_result_status": "closed", "case_result_reason": "Resolved"}
        )
        assert response.status_code == 200

        session.refresh(case)
        assert case.case_result_status == "closed"
        assert case.case_result_reason == "Resolved"

    def test_update_case_status_unauthorized_no_alert(self, client: TestClient, session: Session):
        case = Case(case_result_status="pending", user_id=1)
        session.add(case)
        session.commit()

        # No accepted alerts!
        response = client.patch(
            f"/api/cases/{case.id}/status", 
            json={"case_result_status": "closed", "case_result_reason": "Resolved"}
        )
        assert response.status_code == 400

        session.refresh(case)
        assert case.case_result_status == "pending"

    def test_update_case_status_not_found(self, client: TestClient):
        response = client.patch(
            "/api/cases/999/status", 
            json={"case_result_status": "closed", "case_result_reason": "Resolved"}
        )
        assert response.status_code == 404
