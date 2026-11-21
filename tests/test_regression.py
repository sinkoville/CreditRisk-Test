from fastapi.testclient import TestClient

from app.main import app


def test_stabilite_prediction():
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    with TestClient(app) as client:
        response1 = client.post("/predict", json=payload)
        response2 = client.post("/predict", json=payload)

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()


def test_structure_reponse_prediction():
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert set(response.json().keys()) == {
        "probabilite_defaut",
        "decision"
    }


def test_stabilite_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")

    data = response.json()

    assert data["model_name"] == "CreditRisk-LogisticRegression"
    assert data["version"] == "1.0.0"
    assert data["algorithm"] == "Logistic Regression"
    assert data["loaded"] is True


def test_prediction_limites():
    payload = {
        "age": 70,
        "revenu_mensuel": 100000000,
        "montant_credit": 1000000000,
        "duree_mois": 60,
        "anciennete_activite": 70
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert 0 <= response.json()["probabilite_defaut"] <= 1
