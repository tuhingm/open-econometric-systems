import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from oes_api.catalog import list_models
from oes_api.schemas import OLSRequest
from oes_api.estimators.ols import fit_ols

app = FastAPI(
    title="Open Econometric Systems API",
    version="0.1.0",
    description="Free, browser-accessible econometric estimation service.",
)

origins = [x.strip() for x in os.getenv("OES_ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "oes-api", "version": "0.1.0"}

@app.get("/api/v1/models")
def models():
    return {"models": list_models()}

@app.post("/api/v1/estimate/ols")
def estimate_ols(request: OLSRequest):
    try:
        n = len(request.data.y)
        if len(request.data.x) != n:
            raise ValueError("y and x must have the same number of observations.")
        p = len(request.data.x[0])
        if p == 0 or any(len(row) != p for row in request.data.x):
            raise ValueError("x must be a non-empty rectangular matrix.")
        return fit_ols(
            request.data.y,
            request.data.x,
            variable_names=request.variable_names,
            covariance=request.covariance,
            confidence_level=request.confidence_level,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
