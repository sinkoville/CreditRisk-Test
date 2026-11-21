import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.prediction import credit_model
from app.preprocessing import preprocess_data
from app.validation import CreditRequest


def test_integration_validation_pretraitement():
    data = CreditRequest(
        age=35,
        revenu_mensuel=250000,
        montant_credit=1000000,
        duree_mois=12,
        anciennete_activite=4
    )

    features = preprocess_data(data)

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, 5)

    expected = np.array([[35, 250000, 1000000, 12, 4]])
    np.testing.assert_array_equal(features, expected)


def test_integration_pretraitement_prediction():
    data = CreditRequest(
        age=35,
        revenu_mensuel=250000,
        montant_credit=1000000,
        duree_mois=12,
        anciennete_activite=4
    )

    with TestClient(app):
        features = preprocess_data(data)
        probability = credit_model.predict(features)

    assert isinstance(probability, float)
    assert 0 <= probability <= 1


def test_reproductibilite_prediction():
    data = CreditRequest(
        age=35,
        revenu_mensuel=250000,
        montant_credit=1000000,
        duree_mois=12,
        anciennete_activite=4
    )

    with TestClient(app):
        features = preprocess_data(data)

        result_1 = credit_model.predict(features)
        result_2 = credit_model.predict(features)

    assert result_1 == result_2
