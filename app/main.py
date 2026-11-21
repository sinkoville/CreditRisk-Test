from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.prediction import credit_model
from app.preprocessing import preprocess_data
from app.validation import CreditRequest

app = FastAPI(
    title="CreditRisk API",
    description="API de prédiction du risque de défaut de crédit",
    version="1.0.0"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    messages = {
        "greater_than_equal": "La valeur doit être supérieure ou égale à {limit}.",
        "greater_than": "La valeur doit être strictement supérieure à {limit}.",
        "less_than_equal": "La valeur doit être inférieure ou égale à {limit}.",
        "less_than": "La valeur doit être strictement inférieure à {limit}.",
        "missing": "Ce champ est obligatoire.",
        "int_type": "Ce champ doit être un nombre entier.",
        "float_type": "Ce champ doit être un nombre.",
        "extra_forbidden": "Ce champ n'est pas autorisé."
    }

    errors = []

    for error in exc.errors():
        error_type = error["type"]
        message = messages.get(error_type, "Valeur invalide.")

        if "{limit}" in message:
            limit = error.get("ctx", {}).get(
                "ge",
                error.get("ctx", {}).get(
                    "gt",
                    error.get("ctx", {}).get(
                        "le",
                        error.get("ctx", {}).get("lt", "")
                    )
                )
            )
            message = message.format(limit=limit)

        errors.append({
            "champ": ".".join(str(x) for x in error["loc"][1:]),
            "message": message
        })

    return JSONResponse(
        status_code=422,
        content={"erreur": "Données invalides", "details": errors}
    )
@app.on_event("startup")
def startup_event():
    credit_model.load_model()


@app.get("/health")
def health_check():

    if credit_model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Service indisponible"
        )

    return {
        "status": "healthy",
        "service": "CreditRisk API"
    }


@app.post("/predict")
def predict_credit(data: CreditRequest):

    if credit_model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle indisponible"
        )

    try:
        features = preprocess_data(data)
        probability = credit_model.predict(features)

        if not 0 <= probability <= 1:
            raise RuntimeError("Probabilité invalide")

        if probability < 0.30:
            decision = "RISQUE FAIBLE"
        elif probability < 0.60:
            decision = "RISQUE MODERE"
        else:
            decision = "RISQUE ELEVE"

        return {
            "probabilite_defaut": probability,
            "decision": decision
        }

    except HTTPException:
        raise

    except Exception:  # noqa: BLE001 - masque les erreurs internes sans exposer de trace
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Erreur interne lors de la prédiction"
            }
        )


@app.get("/model-info")
def model_info():

    return {
        "model_name": credit_model.model_name,
        "version": credit_model.version,
        "algorithm": "Logistic Regression",
        "loaded": credit_model.model is not None
    }
