# Model Checkpoints Directory

Stores PyTorch trained weights:

- `best_model.pt` — Optimal model checkpoint selected based on minimum validation loss.
- `smoke_test_model.pt` — Self-contained verification checkpoint generated during smoke testing.

To train a model on configured datasets:

```bash
python src/train.py --epochs 25 --batch-size 32
```

