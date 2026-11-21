"""Run this ONCE on your own PC, in the folder that has your retrained model, before deploying.

It (1) checks that fraud_detection_model.pkl loads and does not need xgboost, and
(2) pins scikit-learn in requirements.txt to the version installed here, because a pickled
scikit-learn model must be loaded by the same scikit-learn version on the server.
"""
import re, sys
import joblib, sklearn

pkg = joblib.load("fraud_detection_model.pkl")
name = pkg.get("model_name", type(pkg["model"]).__name__)
print("Model in file:", name, "|", type(pkg["model"]).__module__)
if type(pkg["model"]).__module__.startswith("xgboost"):
    sys.exit("This is an XGBoost model. Retrain first:  python train_model.py --data sample_data/sample_1_paysim_format.csv")

req = open("requirements.txt").read()
req = re.sub(r"^scikit-learn.*$", f"scikit-learn=={sklearn.__version__}", req, flags=re.M)
open("requirements.txt", "w").write(req)
print("requirements.txt now pins scikit-learn ==", sklearn.__version__)
print("OK - now:  git add -A  &&  git commit -m deploy  &&  git push")
