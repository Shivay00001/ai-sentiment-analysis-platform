from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from src.core.model import sentiment_model
from typing import List, Optional
import uuid

app = FastAPI(title="AI Sentiment Analysis API")

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchSentimentRequest(BaseModel):
    texts: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_text(request: SentimentRequest):
    try:
        result = sentiment_model.analyze(request.text)
        return SentimentResponse(
            text=request.text,
            sentiment=result["label"],
            confidence=result["score"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-analyze")
async def batch_analyze(request: BatchSentimentRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    # In a real system, we'd hand this to a Celery worker
    return {"job_id": job_id, "status": "queued"}
