from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["loaded"] is True
    assert response.json()["version"] == "1.0.0"


def test_prediction_valide():
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert 0 <= response.json()["probabilite_defaut"] <= 1
    assert response.json()["decision"] in [
        "RISQUE FAIBLE",
        "RISQUE MODERE",
        "RISQUE ELEVE"
    ]


def test_age_invalide():
    payload = {
        "age": 15,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["erreur"] == "Données invalides"


def test_revenu_negatif():
    payload = {
        "age": 35,
        "revenu_mensuel": -100,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["erreur"] == "Données invalides"


def test_montant_credit_negatif():
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": -1000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_duree_superieure_limite():
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 90,
        "anciennete_activite": 4
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_champ_obligatoire_manquant():
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
