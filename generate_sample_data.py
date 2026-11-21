"""
generate_sample_data.py
Creates SYNTHETIC test files (PaySim-like) in 3 different column layouts, to test the
mapper's flexibility. These are NOT real data and NOT a validation of the model.
    python generate_sample_data.py
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


def make_base(n=3000, fraud_rate=0.02, seed=7):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"], n,
                     p=[0.34, 0.35, 0.22, 0.08, 0.01])
    step = rng.integers(1, 744, n)
    old_o = rng.lognormal(9, 1.6, n).round(2)
    amount = np.minimum(rng.lognormal(7.5, 1.3, n), old_o * rng.uniform(0.05, 0.9, n)).round(2)
    amount = np.where(typ == "CASH_IN", rng.lognormal(8, 1.3, n).round(2), amount)
    new_o = np.where(typ == "CASH_IN", old_o + amount, old_o - amount).round(2)
    old_d = np.where(np.isin(typ, ["PAYMENT", "DEBIT"]), 0, rng.lognormal(9.5, 1.5, n)).round(2)
    new_d = np.where(typ == "PAYMENT", 0, np.where(typ == "CASH_IN", np.maximum(old_d - amount, 0),
                                                  old_d + amount)).round(2)
    fraud = np.zeros(n, dtype=int)
    idx = rng.choice(n, int(n * fraud_rate), replace=False)
    fraud[idx] = 1
    for i in idx:                                            # "drain the account" pattern
        typ[i] = rng.choice(["TRANSFER", "CASH_OUT"])
        old_o[i] = round(float(rng.lognormal(11.5, 1.0)), 2)
        amount[i], new_o[i] = old_o[i], 0.0
        old_d[i] = 0.0 if rng.random() < 0.7 else round(float(rng.lognormal(10, 1)), 2)
        new_d[i] = 0.0 if typ[i] == "CASH_OUT" and old_d[i] == 0 else round(old_d[i] + amount[i], 2)
    return pd.DataFrame({
        "step": step, "type": typ, "amount": amount,
        "nameOrig": [f"C{rng.integers(10**8, 10**9)}" for _ in range(n)],
        "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
        "nameDest": [("M" if t in ("PAYMENT", "DEBIT") else "C") + str(rng.integers(10**8, 10**9)) for t in typ],
        "oldbalanceDest": old_d, "newbalanceDest": new_d, "isFraud": fraud,
    })


def main():
    os.makedirs(OUT, exist_ok=True)
    base = make_base()

    # 1) original PaySim layout
    base.assign(isFlaggedFraud=0).to_csv(f"{OUT}/sample_1_paysim_format.csv", index=False)

    # 2) "bank" layout: different names, real timestamps, Fraud/Normal labels
    t0 = pd.Timestamp("2025-01-01")
    b = pd.DataFrame({
        "Transaction_ID": [f"TX{i:06d}" for i in range(len(base))],
        "Transaction_Time": [(t0 + pd.Timedelta(hours=int(s))).strftime("%Y-%m-%d %H:%M:%S") for s in base.step],
        "Transaction_Type": base.type,
        "Transaction_Value": base.amount,
        "Sender_Account": base.nameOrig,
        "Sender_Balance_Before": base.oldbalanceOrg,
        "Sender_Balance_After": base.newbalanceOrig,
        "Receiver_Account": base.nameDest,
        "Receiver_Balance_Before": base.oldbalanceDest,
        "Receiver_Balance_After": base.newbalanceDest,
        "Fraud_Label": np.where(base.isFraud == 1, "Fraud", "Normal"),
    })
    b.to_csv(f"{OUT}/sample_2_bank_format.csv", index=False)

    # 3) messy layout: abbreviations, typos, free-text types, no label column
    m = pd.DataFrame({
        "Cust_ID": base.nameOrig,
        "Txn_Amnt": base.amount,
        "TxnType": base.type.map({"CASH_OUT": "cash out", "CASH_IN": "Deposit", "TRANSFER": "Wire",
                                  "PAYMENT": "Payment", "DEBIT": "Debit"}),
        "Acct_Bal_Pre": base.oldbalanceOrg,
        "Acct_Bal_Post": base.newbalanceOrig,
        "Beneficiary": base.nameDest,
        "Beneficiary_Bal_Pre": base.oldbalanceDest,
        "Beneficiary_Bal_Post": base.newbalanceDest,
        "Time": base.step,
        "Branch_Code": np.random.default_rng(1).integers(100, 999, len(base)),   # irrelevant column
    })
    m.to_csv(f"{OUT}/sample_3_messy_names.csv", index=False)
    print("Created 3 sample files in", OUT, "| fraud rows:", int(base.isFraud.sum()), "of", len(base))


if __name__ == "__main__":
    main()
