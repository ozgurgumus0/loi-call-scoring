# Qwen training on Kaggle

Use synthetic transcripts and teacher labels to prepare compatible JSONL inputs. Attach the training, validation and test files to a GPU notebook and run `train_qwen_loi.py`. Evaluate predictions on held-out synthetic transcripts with `eval_only.py`. Keep datasets, scores and exported model weights outside the repository.
