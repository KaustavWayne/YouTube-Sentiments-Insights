import pytest
from flask_app.app import app  # adjust if your app is in a different folder

@pytest.fixture
def client():
    return app.test_client()

# ------------------------------
# Prediction endpoints
# ------------------------------
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

# ------------------------------
# Image endpoints
# ------------------------------
def test_generate_chart_endpoint(client):
    data = {"sentiment_counts": {"1": 5, "0": 3, "-1": 2}}
    response = client.post("/generate_chart", json=data)
    assert response.status_code == 200
    assert response.content_type == "image/png"

def test_generate_wordcloud_endpoint(client):
    data = {"comments": ["Love this!", "Not so great.", "Absolutely amazing!", "Horrible experience."]}
    response = client.post("/generate_wordcloud", json=data)
    assert response.status_code == 200
    assert response.content_type == "image/png"

def test_generate_trend_graph_endpoint(client):
    data = {
        "sentiment_data": [
            {"timestamp": "2025-10-01", "sentiment": 1},
            {"timestamp": "2025-10-02", "sentiment": 0},
            {"timestamp": "2025-10-03", "sentiment": -1}
        ]
    }
    response = client.post("/generate_trend_graph", json=data)
    assert response.status_code == 200
    assert response.content_type == "image/png"
