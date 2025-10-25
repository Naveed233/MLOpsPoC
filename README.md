# Real-Estate Price Estimation Engine (MVP)

Lean appraisal engine for JP residential properties.
- Data prep, feature engineering, time-aware split
- Baseline Linear + XGBoost with MLflow tracking
- FastAPI inference, Dockerized for ECS
- Monitoring & retraining (scripts)

## Quick start
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) python src/data_prep.py   # next step
