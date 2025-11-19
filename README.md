
# MLOps Pipeline Template

**Objective:** End-to-end template with training, FastAPI inference, tests, Dockerfile, and CI via GitHub Actions.

## Features
- `src/train.py` trains and saves a model artifact
- `app/main.py` exposes a prediction API (`/predict`)
- `tests/` includes unit tests
- GitHub Actions workflow runs lint & tests on PR
- Dockerfile for containerized deployment

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn app.main:app --reload
# POST: {{"features": [[0.1, 0.2]]}}
```
