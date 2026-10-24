# SC4002_G40_Project

Stage 1 baseline scaffolding for MASSIVE 1.0 (en-US):

- `src/data_loader.py`: dataset parsing, deterministic vocabulary building, GloVe loading, and DataLoader contracts.
- `src/model.py`: joint BiLSTM + attention model with intent/slot multitask heads and <=20M parameter assertion.
- `src/metrics.py`: intent accuracy and exact-match slot micro-F1 evaluation.
- `train.py`: end-to-end training loop with checkpointing and reload consistency verification.
- `params.json`: baseline hyperparameter configuration.

Run tests:

```bash
python -m unittest tests/test_stage1_baseline.py -v
```
