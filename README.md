# LOI call scoring

A demo pipeline for scoring conversational interest with an LLM, evaluating the scores against outcomes, and training a smaller Qwen model with QLoRA. It supports local transcription and review. Use synthetic conversations and independently created labels for demos.

## Layout

- `scoring/src/` contains ingestion, transcription, scoring, evaluation and export.
- `scoring/kaggle/` contains training and batch inference scripts.
- `scoring/app/` and `frontend/` provide local review interfaces.
- `allocation-briefing/` contains offline backtest methods for compatible local data.
- `docs/RUNBOOK.md` describes local setup.

Install dependencies with `pip install -r scoring/requirements.txt -r requirements.txt`. Copy `.env.example` to `.env` and configure only the providers you intend to use. Run tests with `cd scoring && python -m pytest tests`. Keep call data, prompts derived from private material, and exports outside the repository.
