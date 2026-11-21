from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class CreditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: Annotated[int, Field(strict=True, ge=18, le=70)]

    revenu_mensuel: Annotated[
        float,
        Field(strict=True, gt=0, le=100_000_000, allow_inf_nan=False)
    ]

    montant_credit: Annotated[
        float,
        Field(strict=True, gt=0, le=1_000_000_000, allow_inf_nan=False)
    ]

    duree_mois: Annotated[int, Field(strict=True, ge=1, le=60)]

    anciennete_activite: Annotated[
        float,
        Field(strict=True, ge=0, le=70, allow_inf_nan=False)
    ]
