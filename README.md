# ai-sentiment-analysis-platform

Real sentiment analysis API. **Honest approach:** a VADER-style lexicon
analyzer implemented from scratch (`sentiment.py`) — a curated valence
lexicon plus the documented VADER heuristics: negations ("not good" flips),
intensifiers/dampeners ("very", "slightly"), punctuation and ALL-CAPS
emphasis, "but"-clause reweighting, and compound normalization. No ML model,
no randomness: identical input always gives identical output.

## What it does

- `POST /analyze` — `{"text": "..."}` → `{"label", "compound", "pos", "neu", "neg"}`
- `POST /analyze/batch` — up to 100 texts at once
- `GET /health`

Labels: `positive` (compound ≥ 0.05), `negative` (compound ≤ −0.05),
`neutral` otherwise.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --port 8002
```

```bash
curl -X POST http://localhost:8002/analyze -H 'Content-Type: application/json' \
  -d '{"text":"I absolutely love this, it works perfectly!"}'
# {"label":"positive","compound":0.86,...}
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers positive/negative/neutral sentences, negation flip, determinism,
batch, and empty-input rejection. No API key needed — fully offline.

## Built by VisionQuantech

Part of the [VisionQuantech](https://visionquantech.com/) AI automation studio. See [AI automation services](https://visionquantech.com/ai-automation-services) and the comparison of [AI chatbot vs live chat for small business](https://visionquantech.com/blog/ai-chatbot-vs-live-chat-small-business).
