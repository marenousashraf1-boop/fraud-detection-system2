"""
app.py — Flask backend for the AI-Powered Flexible Financial Fraud Detection System.

The frontend (index.html + main.js + style.css) is served by this same app, and talks to it
through the JSON API below.

  GET  /                       web interface
  POST /predict                multipart form: file=<csv>, threshold=<0-1>   -> JSON results
  GET  /download/<id>          results CSV (output_results.csv)
  GET  /samples/<name>         the 3 synthetic sample files (for demo)
  POST /api/predict            JSON body: one transaction or a list (max 1000)
  GET  /health                 health check

Run locally:   python app.py            -> http://localhost:5000
Production:    gunicorn app:app         (see Procfile and deploy/)
"""
import io
import json
import logging
import os
import re
import time
import uuid

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory
from pandas.errors import EmptyDataError, ParserError

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("[WARNING] flask_cors is not installed! CORS is disabled.")

from pipeline import PipelineError, load_model, run_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RESULTS = "/tmp/results" if os.environ.get("VERCEL") else os.path.join(BASE, "results")
RESULTS_DIR = os.environ.get("RESULTS_DIR", _DEFAULT_RESULTS)
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
KEEP_SECONDS = 24 * 3600
SHOW_ERRORS = os.environ.get("SHOW_ERRORS", "1") == "1"      # SHOW_ERRORS=0 hides technical details
ID_RE = re.compile(r"^[0-9a-f]{32}$")
SAMPLES = {"sample_1_paysim_format.csv", "sample_2_bank_format.csv", "sample_3_messy_names.csv"}

# Only the three frontend files are served from the project root (never app.py / the model file).
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
if CORS_AVAILABLE:
    CORS(app)          # lets you test by double-clicking index.html while the backend runs
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fraud-app")


# ----------------------------- helpers ------------------------------------- #
def read_csv_safely(file_storage) -> pd.DataFrame:
    name = (file_storage.filename or "").strip()
    if not name:
        raise PipelineError("No file selected.")
    if not name.lower().endswith(".csv"):
        raise PipelineError("Unsupported file type. Please upload a .csv file.")
    raw = file_storage.read()
    if not raw or not raw.strip():
        raise PipelineError("The uploaded file is empty.")
    head = raw[:4000].decode("utf-8", errors="ignore").split("\n")[0]
    sep = max([",", ";", "\t", "|"], key=head.count)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
            break
        except UnicodeDecodeError:
            continue
        except (EmptyDataError, ParserError) as e:
            raise PipelineError(f"Could not parse the CSV file ({e}).") from e
    else:
        raise PipelineError("Could not decode the file. Please save it as UTF-8 CSV.")
    if df.empty:
        raise PipelineError("The CSV has a header but no data rows.")
    return df


def cleanup_old_results():
    now = time.time()
    for f in os.listdir(RESULTS_DIR):
        p = os.path.join(RESULTS_DIR, f)
        try:
            if f.endswith(".csv") and now - os.path.getmtime(p) > KEEP_SECONDS:
                os.remove(p)
        except OSError:
            pass


def parse_threshold(value, default=0.5) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as e:
        raise PipelineError("Threshold must be a number between 0 and 1.") from e


def err(msg, code):
    return jsonify({"error": msg}), code


# ------------------------------ frontend ----------------------------------- #
@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/main.js")
def main_js():
    return send_from_directory(BASE, "main.js")


@app.get("/style.css")
def style_css():
    return send_from_directory(BASE, "style.css")


@app.get("/samples/<name>")
def sample(name):
    if name not in SAMPLES:
        return err("Unknown sample file.", 404)
    return send_from_directory(os.path.join(BASE, "sample_data"), name, mimetype="text/csv")


# --------------------------------- API ------------------------------------- #
@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            raise PipelineError("No file part in the request.")
        threshold = parse_threshold(request.form.get("threshold"))
        upload = request.files["file"]
        df = read_csv_safely(upload)
        t0 = time.time()
        result = run_pipeline(df, threshold=threshold)
        log.info("Scored %d rows in %.1fs", len(df), time.time() - t0)

        cleanup_old_results()
        rid = uuid.uuid4().hex
        result.results_df.to_csv(os.path.join(RESULTS_DIR, f"{rid}.csv"), index=False)

        s = result.summary
        out = result.results_df
        top = out[out[s["prediction_column"]] == "Suspicious"] \
            .sort_values(s["probability_column"], ascending=False).head(20)
        hist = np.histogram(out[s["probability_column"]], bins=10, range=(0, 1))[0].tolist()
        return jsonify({
            "filename": upload.filename, "download_id": rid, "summary": s, "hist": hist,
            "mapping": result.mapping_details, "unmapped": [str(c) for c in result.unmapped_columns],
            "warnings": result.warnings, "report": result.data_report,
            "top": {"columns": [str(c) for c in top.columns],
                    "rows": json.loads(top.to_json(orient="values"))},
        })
    except PipelineError as e:
        return err(str(e), 400)
    except Exception as e:                                  # unexpected: full traceback goes to the log
        log.exception("Unexpected error while processing upload")
        detail = f" ({type(e).__name__}: {str(e)[:200]})" if SHOW_ERRORS else ""
        return err("Something went wrong while processing the file." + detail, 500)


@app.get("/download/<rid>")
def download(rid):
    if not ID_RE.match(rid):
        return err("Invalid download link.", 400)
    path = os.path.join(RESULTS_DIR, f"{rid}.csv")
    if not os.path.exists(path):
        return err("This report has expired. Please upload the file again.", 404)
    return send_file(path, as_attachment=True, download_name="output_results.csv", mimetype="text/csv")


@app.post("/api/predict")
def api_predict():
    """Body: one transaction {"type": "TRANSFER", "amount": ...} or a list of them (max 1000)."""
    try:
        data = request.get_json(silent=True)
        if not data:
            raise PipelineError("Send a JSON object or a list of objects.")
        rows = data if isinstance(data, list) else [data]
        if len(rows) > 1000 or not all(isinstance(r, dict) for r in rows):
            raise PipelineError("Send at most 1000 JSON objects.")
        res = run_pipeline(pd.DataFrame(rows), threshold=parse_threshold(request.args.get("threshold")))
        cols = [res.summary["prediction_column"], res.summary["probability_column"]]
        out = res.results_df[cols].rename(columns={cols[0]: "prediction", cols[1]: "fraud_probability"})
        return jsonify({"results": out.to_dict(orient="records"),
                        "summary": {k: res.summary[k] for k in ("total_transactions", "suspicious_count", "model_name")},
                        "warnings": res.warnings})
    except PipelineError as e:
        return err(str(e), 400)
    except Exception:
        log.exception("API error")
        return err("Internal error.", 500)


@app.get("/health")
def health():
    try:
        return jsonify({"status": "ok", "model": load_model().get("model_name")})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 503


@app.errorhandler(413)
def too_large(_):
    return err(f"File too large. Maximum size is {MAX_MB} MB.", 413)


if __name__ == "__main__":
    print("[INFO] Starting Flask API on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
