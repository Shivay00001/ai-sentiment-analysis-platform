# AI Sentiment Analysis Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-4.37-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade NLP platform** for real-time and batch sentiment analysis. Powered by state-of-the-art transformer models (BERT, RoBERTa) and optimized for high-throughput enterprise use cases.

## 🚀 Features

- **Hybrid Analysis**: Choice of lightweight VADER/TextBlob or heavy-duty Transformer models.
- **Async Processing**: Celery-backed batch processing for large datasets.
- **RESTful API**: Fast and well-documented API endpoints.
- **Observability**: Prometheus metrics and structured logging.
- **Caching**: Redis integration for frequent query results.
- **Scalability**: Designed for horizontal scaling with Docker.

## 📁 Project Structure

```
ai-sentiment-analysis-platform/
├── src/
│   ├── api/          # FastAPI routes and middleware
│   ├── core/         # Model loading, Processing logic
│   ├── models/       # NLP model abstractions
│   ├── tasks/        # Celery worker tasks
│   └── main.py       # Application entrypoint
├── tests/            # Unit and integration tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🛠️ Quick Start

```bash
# Clone
git clone https://github.com/Shivay00001/ai-sentiment-analysis-platform.git

# Run with Docker
docker-compose up --build
```

## 📄 License

MIT License
