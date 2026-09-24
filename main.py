"""Real sentiment analysis API.

VADER-style lexicon analyzer (sentiment.py): a curated valence lexicon plus
the documented VADER heuristics — negations, intensifiers/dampeners,
punctuation and ALL-CAPS emphasis, "but"-clause reweighting, compound
normalization. Deterministic: identical input always yields identical output.
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment import analyze

app = FastAPI(title="Sentiment Analysis Platform", version="1.0.0")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=10000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., max_length=100)


@app.get("/health")
def health():
    return {"status": "ok", "method": "vader-style lexicon"}


@app.post("/analyze")
def analyze_one(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    result = analyze(req.text)
    return {"text": req.text, **result}


@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(400, "texts must not be empty")
    return {"results": [{"text": t, **analyze(t)} for t in req.texts]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8002)))
