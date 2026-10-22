from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
from campus_contract import LABELS


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra='forbid')
    left: Unit
    top: Unit
    right: Unit
    bottom: Unit

    @model_validator(mode='after')
    def positive_area(self):
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError('bbox must have positive area')
        return self


class Detection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    class_id: int = Field(ge=0, lt=len(LABELS), strict=True)
    label: str
    confidence: Unit
    bbox: BoundingBox

    @model_validator(mode='after')
    def matching_label(self):
        # Explicit v1 compatibility alias; v2 requires the canonical trash_can.
        if self.label != LABELS[self.class_id] and not (self.class_id == 10 and self.label == 'trash_bin'):
            raise ValueError('Model labels do not match MagicVision')
        return self


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: str = Field(min_length=1, max_length=64)
    processing_time_ms: float = Field(ge=0, allow_inf_nan=False)
    image_width: int = Field(gt=0, le=1280, strict=True)
    image_height: int = Field(gt=0, le=1280, strict=True)
    detections: list[Detection] = Field(max_length=300)
    model_version: str | None = Field(default=None, min_length=1, max_length=128)
