import pytest
from flask_app.app import app  # import your Flask app

@pytest.fixture
def client():
    return app.test_client()

def test_predict_endpoint(client):
    data = {"comments": ["This is a great product!", "Not worth the money.", "It's okay."]}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)

def test_predict_with_timestamps_endpoint(client):
    data = {
        "comments": [
            {"text": "This is fantastic!", "timestamp": "2025-10-25 10:00:00"},
            {"text": "Could be better.", "timestamp": "2025-10-26 14:00:00"}
        ]
    }
    response = client.post("/predict_with_timestamps", json=data)
    assert response.status_code == 200
    assert all('sentiment' in item for item in response.get_json())
