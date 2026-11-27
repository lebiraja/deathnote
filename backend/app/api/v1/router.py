"""Routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from app.ml.model import Model, get_model
from app.schemas.prediction import Prediction, Profile, WhatIfRequest, WhatIfResponse
from app.services import prediction
from app.services.report import build_report

router = APIRouter()
Loaded = Annotated[Model, Depends(get_model)]


@router.get("/health")
def health(model: Loaded) -> dict[str, str]:
    return {"status": "healthy", "model_version": model.version}


@router.post("/predictions/predict")
def predict(profile: Profile, model: Loaded) -> Prediction:
    return prediction.predict(model, profile)


@router.post("/predictions/what-if")
def what_if(request: WhatIfRequest, model: Loaded) -> WhatIfResponse:
    try:
        return prediction.what_if(model, request.profile, request.changes)
    except prediction.InvalidChange as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/report", response_class=Response)
def report(profile: Profile, model: Loaded) -> Response:
    pdf = build_report(profile, prediction.predict(model, profile))
    return Response(
        pdf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="life-estimate.pdf"'},
    )
