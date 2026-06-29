import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from main import fastapi_app
from database import get_session
from models import Alert, Case

# Setup isolated in-memory SQLite database for alerts tests
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

class TestAlerts:

    # ---------------------------------------------------------
    # GET /api/alerts
    # ---------------------------------------------------------
    def test_get_alerts_list(self, client: TestClient, session: Session):
        # Create a mock Case and an Alert linked to it
        case = Case(context="This is a test case with details about contract law.", user_id=1)
        session.add(case)
        session.commit()

        alert = Alert(
            case_id=case.id,
            title="Contract Breach Detected",
            summary="Found relevant contract clauses.",
            ai_reasoning="Strategic match",
            severity="strategic",
            status="unread",
            review_status="pending"
        )
        session.add(alert)
        session.commit()

        # Query the alerts endpoint
        response = client.get("/api/alerts")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        
        # Check that details are properly mapped (with truncated case name)
        assert data[0]["title"] == "Contract Breach Detected"
        assert data[0]["case_id"] == case.id
        assert data[0]["case_name"] == "This is a test case with details about c..." # Truncated to 40 chars + "..."

    # ---------------------------------------------------------
    # PATCH /api/alerts/{alert_id}/read
    # ---------------------------------------------------------
    def test_mark_alert_read(self, client: TestClient, session: Session):
        alert = Alert(
            title="Notification Alert",
            summary="A regular system notification",
            status="unread"
        )
        session.add(alert)
        session.commit()

        # Mark read
        response = client.patch(f"/api/alerts/{alert.id}/read")
        assert response.status_code == 200
        assert response.json()["ok"] is True

        # Check DB status
        session.refresh(alert)
        assert alert.status == "read"

    # ---------------------------------------------------------
    # PATCH /api/alerts/{alert_id}/accept
    # ---------------------------------------------------------
    def test_accept_alert(self, client: TestClient, session: Session):
        alert = Alert(
            title="Notification Alert",
            summary="A regular system notification",
            review_status="pending"
        )
        session.add(alert)
        session.commit()

        # Accept alert
        response = client.patch(f"/api/alerts/{alert.id}/accept")
        assert response.status_code == 200
        assert response.json()["ok"] is True

        session.refresh(alert)
        assert alert.review_status == "accepted"

    # ---------------------------------------------------------
    # PATCH /api/alerts/{alert_id}/reject
    # ---------------------------------------------------------
    def test_reject_alert(self, client: TestClient, session: Session, monkeypatch):
        # Mock background task queuing so the test runs completely locally
        enqueued_cases = []
        def mock_enqueue(case_id):
            enqueued_cases.append(case_id)

        monkeypatch.setattr("ai.background.enqueue_research", mock_enqueue)

        # Create case and alert to reject
        case = Case(context="Original context", status="complete", user_id=1)
        session.add(case)
        session.commit()

        alert = Alert(
            case_id=case.id,
            title="Bad Alert",
            summary="Incorrect finding details",
            review_status="pending"
        )
        session.add(alert)
        session.commit()

        # Submit rejection feedback
        rejection_reason = "This is incorrect since the statute has expired."
        response = client.patch(
            f"/api/alerts/{alert.id}/reject",
            json={"reason": rejection_reason}
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

        # Verify DB updates on the alert
        session.refresh(alert)
        assert alert.review_status == "rejected"
        assert alert.rejection_reason == rejection_reason
        assert alert.status == "archived"

        # Verify DB updates on the parent case
        session.refresh(case)
        assert case.context == "Original context"  # Context remains untouched
        assert case.status == "pending"

        # Verify that background research task was queued with correct ID
        assert enqueued_cases == [case.id]

    # ---------------------------------------------------------
    # PATCH /api/alerts/archive-all
    # ---------------------------------------------------------
    def test_archive_all_alerts(self, client: TestClient, session: Session):
        # Create unread and read alerts
        alert1 = Alert(title="Alert 1", summary="One", status="unread")
        alert2 = Alert(title="Alert 2", summary="Two", status="unread")
        alert3 = Alert(title="Alert 3", summary="Three", status="read")

        session.add(alert1)
        session.add(alert2)
        session.add(alert3)
        session.commit()

        # Bulk archive
        response = client.patch("/api/alerts/archive-all")
        assert response.status_code == 200
        assert response.json()["archived_count"] == 2

        # Check DB to verify only unread alerts were archived
        session.refresh(alert1)
        session.refresh(alert2)
        session.refresh(alert3)

        assert alert1.status == "archived"
        assert alert2.status == "archived"
        assert alert3.status == "read" # Remained 'read', not archived
