from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from src.core.config import settings
import torch

class SentimentModel:
    def __init__(self):
        self.model_name = settings.MODEL_PATH
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def load(self):
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "sentiment-analysis", 
            model=self.model_name, 
            tokenizer=self.model_name,
            device=device
        )

    def analyze(self, text: str):
        if not self.pipeline:
            self.load()
        
        # RoBERTa returns LABEL_0, LABEL_1, LABEL_2
        # Mapping: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        result = self.pipeline(text[:settings.MAX_TEXT_LENGTH])[0]
        
        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }
        
        return {
            "label": label_map.get(result["label"], result["label"]),
            "score": round(result["score"], 4),
            "original_label": result["label"]
        }

sentiment_model = SentimentModel()
