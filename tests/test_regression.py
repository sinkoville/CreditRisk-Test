
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.main import app
from app.prediction import CreditRiskModel


def test_stabilite_prediction():
    payload = {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4,
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
        "anciennete_activite": 4,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "probabilite_defaut",
        "decision",
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
        "anciennete_activite": 70,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert 0 <= response.json()["probabilite_defaut"] <= 1


def test_persistance_modele(tmp_path):
    features = np.array([[35, 250000, 1000000, 12, 4]])

    modele_initial = CreditRiskModel()
    modele_initial.load_model()

    prediction_initiale = modele_initial.predict(features)

    fichier = tmp_path / "creditrisk_model.joblib"
    modele_initial.save_model(fichier)

    modele_recharge = CreditRiskModel()
    modele_recharge.reload_model(fichier)

    prediction_rechargee = modele_recharge.predict(features)

    assert prediction_rechargee == prediction_initiale


def test_metriques_reference():
    rng = np.random.default_rng(42)

    X = rng.uniform(
        low=[18, 50000, 50000, 1, 0],
        high=[70, 2000000, 5000000, 60, 40],
        size=(1000, 5),
    )

    age, revenu, montant, duree, anciennete = X.T

    score = (
        0.000002 * montant
        + 0.015 * duree
        - 0.000001 * revenu
        - 0.025 * anciennete
        - 0.01 * age
    )

    probabilities = 1 / (1 + np.exp(-score))
    y = (probabilities > 0.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    assert np.isclose(recall, 0.9940828402366864)
    assert np.isclose(f1, 0.9911504424778761)

