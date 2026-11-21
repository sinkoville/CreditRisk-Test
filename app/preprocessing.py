import numpy as np

from app.validation import CreditRequest


def preprocess_data(data: CreditRequest) -> np.ndarray:
    """
    Prépare les données du demandeur
    pour le modèle de Machine Learning.
    """

    features = [
        data.age,
        data.revenu_mensuel,
        data.montant_credit,
        data.duree_mois,
        data.anciennete_activite,
    ]

    return np.array(features, dtype=float).reshape(1, -1)
