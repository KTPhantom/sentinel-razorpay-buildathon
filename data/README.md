# data/

The Kaggle dataset isn't committed here due to size — you need to download it manually.

**Download:**
1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv` (~150 MB)
3. Drop it here as `data/creditcard.csv`

The file should have 284,807 rows, 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`), and a ~0.17% fraud rate. Running `python scripts/verify_setup.py` will confirm the schema is correct.

The synthetic dataset (`synthetic_transactions.csv`) is generated on demand by `src/data/synthetic.py` — it's not stored here. Run `python scripts/generate_synthetic_data.py` if you want a saved copy.
