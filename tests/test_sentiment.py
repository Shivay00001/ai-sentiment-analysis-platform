from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_positive():
    r = client.post("/analyze", json={"text": "I absolutely love this product, it is amazing!"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert body["compound"] > 0.5


def test_negative():
    r = client.post("/analyze", json={"text": "This is terrible. The app keeps crashing and support was rude."})
    body = r.json()
    assert body["label"] == "negative"
    assert body["compound"] < -0.5


def test_neutral():
    r = client.post("/analyze", json={"text": "The meeting is scheduled for 3pm tomorrow in room 4."})
    body = r.json()
    assert body["label"] == "neutral"
    assert body["compound"] == 0.0


def test_negation_flips():
    r = client.post("/analyze", json={"text": "The movie was not good at all."})
    assert r.json()["label"] == "negative"


def test_deterministic_no_randomness():
    t = "The food was good but the service was awful."
    a = client.post("/analyze", json={"text": t}).json()
    b = client.post("/analyze", json={"text": t}).json()
    assert a == b


def test_batch():
    r = client.post("/analyze/batch", json={"texts": ["I love it", "I hate it", "It is a table"]})
    assert r.status_code == 200
    labels = [x["label"] for x in r.json()["results"]]
    assert labels == ["positive", "negative", "neutral"]


def test_empty_rejected():
    assert client.post("/analyze", json={"text": "   "}).status_code == 400
