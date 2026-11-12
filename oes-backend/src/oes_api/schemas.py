from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class OLSData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    y: list[float] = Field(min_length=3)
    x: list[list[float]] = Field(min_length=3)

class OLSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: OLSData
    variable_names: list[str] | None = None
    covariance: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"] = "nonrobust"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
