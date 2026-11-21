"""
check_setup.py — run this FIRST if anything fails:   python check_setup.py
Checks Python/library versions, loads the model, and scores the sample files end-to-end.
If something is wrong it prints the exact error (copy the whole output and send it).
"""
import glob
import os
import platform
import sys
import traceback

print("Python  :", sys.version.split()[0], "|", platform.platform())
ok = True
for mod in ["pandas", "numpy", "sklearn", "xgboost", "joblib", "rapidfuzz", "flask"]:
    try:
        m = __import__(mod)
        print(f"{mod:9}:", getattr(m, "__version__", "?"))
    except Exception as e:
        ok = False
        print(f"{mod:9}: MISSING -> {type(e).__name__}: {e}   (fix: pip install -r requirements.txt)")

if ok:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        import pandas as pd
        from pipeline import load_model, run_pipeline
        pkg = load_model()
        print("Model   : loaded ->", pkg.get("model_name"), f"({len(pkg['feature_columns'])} features)")
        for f in sorted(glob.glob(os.path.join(here, "sample_data", "*.csv"))):
            r = run_pipeline(pd.read_csv(f))
            print(f"  OK {os.path.basename(f):32} rows={r.summary['total_transactions']} "
                  f"suspicious={r.summary['suspicious_count']}")
        print("\nALL GOOD. Start the app with:  python app.py")
    except Exception:
        print("\nPROBLEM FOUND (send me everything below):\n")
        traceback.print_exc()
