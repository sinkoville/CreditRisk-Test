import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_age_texte(client):
    payload = {
        "age": "trente",
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_valeur_infinie(client):
    payload = {
        "age": 35,
        "revenu_mensuel": "Infinity",
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_champ_supplementaire(client):
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4,
        "mot_de_passe": "secret"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_corps_json_invalide(client):
    response = client.post(
        "/predict",
        content="{json invalide",
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_probabilite_dans_intervalle(client):
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    probability = response.json()["probabilite_defaut"]

    assert 0 <= probability <= 1
