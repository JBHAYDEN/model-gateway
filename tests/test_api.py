from fastapi.testclient import TestClient
from app.main import app
def test_health():
    with TestClient(app) as client: response=client.get("/health")
    assert response.status_code==200
def test_prediction():
    payload={"age":31,"sessions":8,"purchases":2,"averageSessionMinutes":12.5,"channel":"organic"}
    with TestClient(app) as client: response=client.post("/predict",json=payload)
    assert response.status_code==200
    assert response.json()["label"] in (0,1)
