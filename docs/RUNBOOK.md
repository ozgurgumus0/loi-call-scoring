# Local scoring workflow

1. Create a virtual environment, install the scoring requirements, and configure your chosen provider in a local `.env`.
2. Use synthetic transcripts or authorized local recordings. Configure their locations using the paths in `.env.example`.
3. Run the local scorer or review UI. Check its structured outputs before using them for evaluation.
4. For a local model, configure Ollama or train a Qwen adapter using synthetic labels. Keep the input, model files and reports outside the repo.
