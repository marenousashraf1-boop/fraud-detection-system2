# AI-Powered Flexible Financial Fraud Detection System

Upload **any** transactions CSV — even if its columns are named differently (`Amount`, `Transaction_Value`, `Txn_Amnt`…) —
and the system understands the file, maps the columns automatically, scores every transaction with a trained ML model, and
returns a report (`Normal` / `Suspicious` + fraud probability) on screen and as a downloadable CSV.

## Project files

| File / folder | Purpose |
|---|---|
| `index.html`, `main.js`, `style.css` | Frontend (single page): upload, threshold, results dashboard |
| `app.py` | Flask backend: serves the frontend, JSON API (`/predict`, `/api/predict`, `/download/<id>`, `/health`) |
| `feature_mapper.py` | Data understanding + automatic column mapping (synonyms, fuzzy matching, value validation) |
| `pipeline.py` | Feature engineering (same as training), prediction, summary statistics |
| `train_model.py` | Retrain Random Forest + XGBoost on PaySim; keeps the best by F1 |
| `fraud_detection_model.pkl` | Saved XGBoost model package |
| `Fraud_Detection_System.ipynb` | Original exploration / training notebook |
| `Fraud-Detection-System.pptx`, `Documentation.docx` | Presentation and written report |
| `Procfile`, `requirements.txt` | Deployment start command and dependencies |
| `sample_data/`, `generate_sample_data.py` | 3 synthetic test files with 3 different column layouts |
| `tests/` | 25 automated tests (mapper, pipeline, web app) |
| `deploy/` | AWS EC2 deployment: setup script, systemd service, nginx config, step-by-step guide (Arabic) |
| `static/images/` | Figures used by the web page |
| `check_setup.py` | Self-check for your machine: library versions, model loading, sample run |

## Run locally

```bash
pip install -r requirements.txt
python check_setup.py        # optional: checks everything and prints the exact error if something is wrong
python app.py                # open http://localhost:5000
```
Production: `gunicorn app:app` (see `Procfile`, and `deploy/AWS_DEPLOY_AR.md` for AWS EC2 with nginx).
Tests: `pip install pytest && pytest -q`.

## API

```bash
curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" \
  -d '{"type":"TRANSFER","amount":900000,"oldbalanceOrg":900000,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'
```

## Standard schema
`Customer_ID, Time_Step, Type, Amount, Old_Balance_Orig, New_Balance_Orig, Dest_ID, Old_Balance_Dest, New_Balance_Dest, Is_Fraud (optional)`.
Required: `Type`, `Amount`, `Old_Balance_Orig`, `New_Balance_Orig`. Optional fields get defaults with a warning.

## Honest limitations
* The model was trained on **PaySim, a synthetic dataset**; near-perfect scores (F1 0.964 XGBoost / 0.997 Random Forest) do not
  guarantee real-world accuracy. The 3 sample files are synthetic too: they verify the mapping and pipeline, not the model.
* The model needs the sender balance columns (before/after).
* In the notebook XGBoost was chosen by a near-tied PR-AUC although Random Forest has better precision/F1; run
  `python train_model.py --data <PaySim.csv>` to choose by F1.
* Single-instance deployment; the load balancer / Auto Scaling / S3 plan is a proposal, not implemented.
* The `.pkl` is a Python pickle: only load model files you trust.
