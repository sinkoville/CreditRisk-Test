from locust import HttpUser, between, task


class CreditRiskUser(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def predict_credit(self):
        payload = {
            "age": 35,
            "revenu_mensuel": 250000,
            "montant_credit": 1000000,
            "duree_mois": 12,
            "anciennete_activite": 4
        }

        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            else:
                try:
                    data = response.json()
                    probability = data.get("probabilite_defaut")

                    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                        response.failure("Probabilité invalide")
                except ValueError:
                    response.failure("Réponse JSON invalide")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
