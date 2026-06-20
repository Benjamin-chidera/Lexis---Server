import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from main import fastapi_app
from database import get_session
from models import User
from auth import hash_password

# Setup an isolated in-memory SQLite database for auth tests
sqlite_url = "sqlite:///:memory:"
engine = create_engine(
    sqlite_url, 
    connect_args={"check_same_thread": False}, 
    poolclass=StaticPool
)

@pytest.fixture(name="session")
def session_fixture():
    # Create database tables
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    # Clean up tables after each test runs
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(name="client")
def client_fixture(session: Session):
    # Override get_session dependency in FastAPI to use our test DB session
    def get_session_override():
        return session

    fastapi_app.dependency_overrides[get_session] = get_session_override
    client = TestClient(fastapi_app)
    yield client
    # Clean up dependency overrides
    fastapi_app.dependency_overrides.clear()

class TestAuth:

    # ---------------------------------------------------------
    # POST /api/auth/admin-setup (First-time Admin Setup)
    # ---------------------------------------------------------
    def test_admin_setup_success(self, client: TestClient, session: Session):
        response = client.post(
            "/api/auth/admin-setup",
            json={
                "corporate_email": "admin@company.com",
                "name": "Super Admin",
                "password": "adminpassword123",
                "confirm_password": "adminpassword123",
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "admin@company.com"
        assert "created" in data["message"].lower()

        # Try to run admin setup again (should fail with 409 Conflict)
        response_second = client.post(
            "/api/auth/admin-setup",
            json={
                "corporate_email": "another@company.com",
                "name": "Second Admin",
                "password": "adminpassword123",
                "confirm_password": "adminpassword123",
            }
        )
        assert response_second.status_code == 409

    # ---------------------------------------------------------
    # POST /api/auth/check-email
    # ---------------------------------------------------------
    def test_check_email_registered_user(self, client: TestClient, session: Session):
        # Insert a user into the mock database
        user = User(corporate_email="employee@company.com", name="Bob Employee", role="employee")
        session.add(user)
        session.commit()

        # Check a registered email
        response = client.post(
            "/api/auth/check-email",
            json={"corporate_email": "employee@company.com"}
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Bob Employee"
        assert response.json()["has_password"] is False

    def test_check_email_not_registered(self, client: TestClient):
        # Check an unregistered email
        response = client.post(
            "/api/auth/check-email",
            json={"corporate_email": "stranger@company.com"}
        )
        assert response.status_code == 404

    # ---------------------------------------------------------
    # POST /api/auth/login
    # ---------------------------------------------------------
    def test_login_success(self, client: TestClient, session: Session):
        # Insert a user who already has a password
        hashed = hash_password("securepassword123")
        user = User(
            corporate_email="jane@company.com",
            name="Jane Doe",
            password_hash=hashed,
            role="employee"
        )
        session.add(user)
        session.commit()

        # Login with correct password
        response = client.post(
            "/api/auth/login",
            json={"corporate_email": "jane@company.com", "password": "securepassword123"}
        )
        assert response.status_code == 200
        assert response.json()["email"] == "jane@company.com"
        assert "access_token" in response.cookies  # Verify that cookie is set in response

    def test_login_wrong_password(self, client: TestClient, session: Session):
        hashed = hash_password("securepassword123")
        user = User(
            corporate_email="jane@company.com",
            name="Jane Doe",
            password_hash=hashed,
            role="employee"
        )
        session.add(user)
        session.commit()

        # Login with wrong password
        response = client.post(
            "/api/auth/login",
            json={"corporate_email": "jane@company.com", "password": "wrongpassword"}
        )
        assert response.status_code == 401

    # ---------------------------------------------------------
    # POST /api/auth/set-password
    # ---------------------------------------------------------
    def test_set_password_success(self, client: TestClient, session: Session):
        # Register a new user without a password hash yet
        user = User(corporate_email="newbie@company.com", name="Newbie", role="employee")
        session.add(user)
        session.commit()

        # Set password
        response = client.post(
            "/api/auth/set-password",
            json={
                "corporate_email": "newbie@company.com",
                "password": "brandnewpassword123",
                "confirm_password": "brandnewpassword123",
            }
        )
        assert response.status_code == 200
        assert response.json()["email"] == "newbie@company.com"

        # Check that user database record now contains password hash
        session.refresh(user)
        assert user.password_hash is not None

    # ---------------------------------------------------------
    # Admin User Actions: Create, Deactivate, Activate Users
    # ---------------------------------------------------------
    def test_admin_user_management(self, client: TestClient, session: Session):
        # Create an admin user first
        admin = User(
            corporate_email="boss@company.com",
            name="Boss Admin",
            password_hash=hash_password("bosspassword"),
            role="admin"
        )
        session.add(admin)
        session.commit()

        # Log in as admin to retrieve cookies/session
        login_res = client.post(
            "/api/auth/login",
            json={"corporate_email": "boss@company.com", "password": "bosspassword"}
        )
        assert login_res.status_code == 200
        
        # Test creating a new user via admin endpoint
        create_res = client.post(
            "/api/admin/users",
            json={"corporate_email": "subordinate@company.com", "name": "Subordinate", "role": "employee"}
        )
        assert create_res.status_code == 201
        sub_id = create_res.json()["user_id"]

        # Test listing all users
        list_res = client.get("/api/admin/users")
        assert list_res.status_code == 200
        assert len(list_res.json()) == 2  # Boss Admin & Subordinate

        # Test deactivating the newly created user
        deactivate_res = client.patch(f"/api/admin/users/{sub_id}/deactivate")
        assert deactivate_res.status_code == 200

        # Verify deactivated status in database
        sub_user = session.get(User, sub_id)
        assert sub_user.is_active is False

        # Test activating the user back
        activate_res = client.patch(f"/api/admin/users/{sub_id}/activate")
        assert activate_res.status_code == 200
        
        session.refresh(sub_user)
        assert sub_user.is_active is True
