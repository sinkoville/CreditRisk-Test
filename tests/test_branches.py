import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.prediction import CreditRiskModel, credit_model


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_modele_indisponible(client, monkeypatch):
    monkeypatch.setattr(credit_model, "model", None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service indisponible"


def test_prediction_modele_indisponible(client, monkeypatch):
    monkeypatch.setattr(credit_model, "model", None)

    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Modèle indisponible"


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.20, "RISQUE FAIBLE"),
        (0.40, "RISQUE MODERE"),
        (0.80, "RISQUE ELEVE"),
    ]
)
def test_branches_decision(client, monkeypatch, probability, expected):
    monkeypatch.setattr(
        credit_model,
        "predict",
        lambda features: probability
    )

    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["decision"] == expected


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_probabilite_invalide(client, monkeypatch, probability):
    monkeypatch.setattr(
        credit_model,
        "predict",
        lambda features: probability
    )

    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert "traceback" not in response.text.lower()


def test_erreur_pretraitement(client, monkeypatch):
    def simulate_error(data):
        raise ValueError("Erreur interne simulée")

    monkeypatch.setattr("app.main.preprocess_data", simulate_error)

    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de la prédiction"
    assert "Erreur interne simulée" not in response.text


def test_exception_http_conservee(client, monkeypatch):
    def simulate_http_error(data):
        raise HTTPException(status_code=418, detail="Erreur HTTP simulée")

    monkeypatch.setattr("app.main.preprocess_data", simulate_http_error)

    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 418
    assert response.json()["detail"] == "Erreur HTTP simulée"


def test_prediction_directe_modele_non_charge():
    model = CreditRiskModel()

    with pytest.raises(RuntimeError, match="Modèle non chargé"):
        model.predict(np.zeros((1, 5)))
