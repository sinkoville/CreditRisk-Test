import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CreditRiskModel:

    def __init__(self):
        self.model = None
        self.version = "1.0.0"
        self.model_name = "CreditRisk-LogisticRegression"

    def load_model(self):
        """Entraîne le modèle sur des données synthétiques."""

        rng = np.random.default_rng(42)

        X = rng.uniform(
            low=[18, 50000, 50000, 1, 0],
            high=[70, 2000000, 5000000, 60, 40],
            size=(1000, 5)
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

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42))
        ])

        self.model.fit(X, y)

    def predict(self, features):
        """Retourne la probabilité de défaut."""

        if self.model is None:
            raise RuntimeError("Modèle non chargé")

        probability = self.model.predict_proba(features)[0][1]

        return float(probability)


credit_model = CreditRiskModel()
