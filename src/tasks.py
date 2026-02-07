from celery import Celery
from src.core.config import settings
from src.core.model import sentiment_model

celery_app = Celery("tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

@celery_app.task
def process_batch_sentiment(texts: list):
    results = []
    for text in texts:
        results.append(sentiment_model.analyze(text))
    return results
