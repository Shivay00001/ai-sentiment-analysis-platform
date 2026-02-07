from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Sentiment Platform"
    MODEL_PATH: str = "cardiffnlp/twitter-roberta-base-sentiment"
    REDIS_URL: str = "redis://localhost:6379/0"
    LOG_LEVEL: str = "INFO"
    MAX_TEXT_LENGTH: int = 512
    
    class Config:
        env_file = ".env"

try:
    settings = Settings()
except Exception:
    import os
    class MockSettings:
        PROJECT_NAME = "AI Sentiment Platform"
        MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment"
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        LOG_LEVEL = "INFO"
        MAX_TEXT_LENGTH = 512
    settings = MockSettings()
