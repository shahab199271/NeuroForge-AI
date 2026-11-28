# NeuroForge AI
# Maintainer: Shahab Baloochi

"""Local FastAPI inference interface with lazy model loading."""
import os
from functools import lru_cache
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import BASE_MODEL, load_model, predict

app = FastAPI(title="NeuroForge AI", version="0.1.0")
_inference_lock = Lock()


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class SentimentResponse(BaseModel):
    sentiment: str
    raw_output: str


@lru_cache(maxsize=1)
def get_runtime():
    adapter = os.environ.get("NEUROFORGE_ADAPTER")
    if not adapter:
        raise RuntimeError("NEUROFORGE_ADAPTER must point to a trained local adapter")
    return load_model(adapter, os.environ.get("NEUROFORGE_BASE_MODEL", BASE_MODEL))


@app.get("/health")
def health():
    return {
        "status": "configured" if os.environ.get("NEUROFORGE_ADAPTER") else "missing_adapter"
    }


@app.post("/analyze", response_model=SentimentResponse)
def analyze(request: SentimentRequest):
    try:
        tokenizer, model = get_runtime()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Model unavailable; check adapter configuration"
        ) from exc
    try:
        # Serialization prevents simultaneous model.generate() calls on one GPU.
        with _inference_lock:
            predicted, raw = predict(request.text, tokenizer, model)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model inference failed") from exc
    if predicted is None:
        raise HTTPException(status_code=422, detail="Model returned an invalid sentiment label")
    return SentimentResponse(sentiment=predicted, raw_output=raw)
