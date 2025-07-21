from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Diamond Query API"}

def test_query_diamond():
    headers = {"x-session-id": "test_session"}
    payload = {"query": "Find diamonds with clarity VS1"}
    response = client.post("/query", json=payload, headers=headers)
    
    # Assuming the DiamondFinder returns a valid response
    assert response.status_code == 200
    assert "x-session-id" in response.headers
    assert response.headers["x-session-id"] == "test_session"
    assert isinstance(response.json(), dict)  # Replace with specific checks based on expected output