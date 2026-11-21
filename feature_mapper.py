"""
feature_mapper.py
=================
Data Understanding + Automatic Feature Mapping.

Takes ANY transactions CSV (whatever its column names are) and converts it to a
standard schema the model understands.

Steps:
  1. analyze_data()  -> data-understanding report (dtypes, missing %, uniques...)
  2. map_columns()   -> synonym dictionary + fuzzy matching (rapidfuzz) +
                        side/phase awareness (origin vs destination, old vs new)
                        + content validation (is the column really money / a date?)
  3. standardize()   -> DataFrame with the standard column names, clean types.

Standard schema
---------------
Customer_ID, Time_Step, Type, Amount, Old_Balance_Orig, New_Balance_Orig,
Dest_ID, Old_Balance_Dest, New_Balance_Dest, Is_Fraud (optional label)
"""
from __future__ import annotations

import re
import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# --------------------------------------------------------------------------- #
# Standard schema
# --------------------------------------------------------------------------- #
STANDARD_FIELDS = [
    "Customer_ID", "Time_Step", "Type", "Amount",
    "Old_Balance_Orig", "New_Balance_Orig",
    "Dest_ID", "Old_Balance_Dest", "New_Balance_Dest", "Is_Fraud",
]
# Without these four the model cannot score anything.
REQUIRED = ["Type", "Amount", "Old_Balance_Orig", "New_Balance_Orig"]
# Optional fields get a default (and a warning) when absent.
DEFAULTS = {"Time_Step": 0, "Old_Balance_Dest": 0.0, "New_Balance_Dest": 0.0}

NUMERIC_FIELDS = ["Amount", "Old_Balance_Orig", "New_Balance_Orig",
                  "Old_Balance_Dest", "New_Balance_Dest"]

# Synonym dictionary (names are compared after normalisation: lowercase, no symbols)
_RAW_SYNONYMS = {
    "Customer_ID": ["nameorig", "customerid", "customer", "customername", "clientid", "client",
                    "senderid", "sender", "sendername", "senderaccount", "originid", "origin",
                    "originator", "accountid", "account", "accountnumber", "accountno", "custid",
                    "fromaccount", "from", "sourceaccount", "source", "payer", "payerid", "userid"],
    "Time_Step": ["step", "time", "timestamp", "datetime", "date", "transactiontime",
                  "transactiondate", "txntime", "txndate", "transactiondatetime", "hour",
                  "trxtime", "trxdate", "eventtime", "createdat"],
    "Type": ["type", "transactiontype", "txntype", "trxtype", "transtype",
             "transactioncategory", "category", "kind", "paymenttype", "operation",
             "operationtype", "txtype", "transactionkind", "channel"],
    "Amount": ["amount", "transactionamount", "transactionvalue", "transferamount", "txnamount",
               "txnamt", "txnvalue", "amt", "value", "sum", "transactionamt", "trxamount",
               "paymentamount", "transamount", "amountusd", "amountegp"],
    "Old_Balance_Orig": ["oldbalanceorg", "oldbalanceorig", "oldbalance", "oldbalanceorigin",
                         "balancebefore", "senderbalancebefore", "senderoldbalance",
                         "originoldbalance", "initialbalance", "openingbalance", "prebalance",
                         "accountbalancebefore", "balancebeforetransaction", "sourcebalancebefore",
                         "accbalpre", "origbalancebefore", "balanceorigbefore"],
    "New_Balance_Orig": ["newbalanceorig", "newbalanceorg", "newbalance", "newbalanceorigin",
                         "balanceafter", "senderbalanceafter", "sendernewbalance",
                         "originnewbalance", "finalbalance", "closingbalance", "postbalance",
                         "accountbalanceafter", "balanceaftertransaction", "sourcebalanceafter",
                         "accbalpost", "origbalanceafter", "balanceorigafter"],
    "Dest_ID": ["namedest", "destination", "destinationid", "destid", "receiver", "receiverid",
                "receivername", "recipient", "recipientid", "beneficiary", "beneficiaryid",
                "toaccount", "to", "targetaccount", "target", "payee", "payeeid", "merchant",
                "merchantid", "destaccount", "destinationaccount"],
    "Old_Balance_Dest": ["oldbalancedest", "oldbalancedestination", "receiverbalancebefore",
                         "recipientbalancebefore", "destbalancebefore", "destinationbalancebefore",
                         "beneficiarybalancebefore", "receiveroldbalance", "destoldbalance",
                         "targetbalancebefore", "payeebalancebefore"],
    "New_Balance_Dest": ["newbalancedest", "newbalancedestination", "receiverbalanceafter",
                         "recipientbalanceafter", "destbalanceafter", "destinationbalanceafter",
                         "beneficiarybalanceafter", "receivernewbalance", "destnewbalance",
                         "targetbalanceafter", "payeebalanceafter"],
    "Is_Fraud": ["isfraud", "fraud", "label", "class", "target", "isfraudulent", "fraudulent",
                 "fraudflag", "fraudlabel", "isfraudtransaction", "groundtruth"],
}
SYNONYMS = {k: [re.sub(r"[^a-z0-9]", "", s.lower()) for s in v] for k, v in _RAW_SYNONYMS.items()}

# Which side / phase each field expects ("orig"/"dest", "old"/"new")
FIELD_TAGS = {
    "Customer_ID": {"side": "orig"},
    "Dest_ID": {"side": "dest"},
    "Old_Balance_Orig": {"side": "orig", "phase": "old"},
    "New_Balance_Orig": {"side": "orig", "phase": "new"},
    "Old_Balance_Dest": {"side": "dest", "phase": "old"},
    "New_Balance_Dest": {"side": "dest", "phase": "new"},
}

# Transaction-type vocabulary -> the 5 types the model was trained on
TYPE_MAP = {
    "CASHIN": "CASH_IN", "DEPOSIT": "CASH_IN", "CREDIT": "CASH_IN", "TOPUP": "CASH_IN",
    "CASHOUT": "CASH_OUT", "WITHDRAWAL": "CASH_OUT", "WITHDRAW": "CASH_OUT", "ATM": "CASH_OUT",
    "ATMWITHDRAWAL": "CASH_OUT",
    "DEBIT": "DEBIT",
    "PAYMENT": "PAYMENT", "PAY": "PAYMENT", "PURCHASE": "PAYMENT", "POS": "PAYMENT",
    "BILLPAYMENT": "PAYMENT",
    "TRANSFER": "TRANSFER", "TRANSF": "TRANSFER", "WIRE": "TRANSFER", "WIRETRANSFER": "TRANSFER",
    "P2P": "TRANSFER", "TRF": "TRANSFER", "BANKTRANSFER": "TRANSFER",
}
KNOWN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


# --------------------------------------------------------------------------- #
# Result objects
# --------------------------------------------------------------------------- #
class MappingError(ValueError):
    """Raised when the file cannot be mapped to the standard schema."""


@dataclass
class MappingResult:
    mapping: dict = field(default_factory=dict)      # standard field -> source column
    details: list = field(default_factory=list)      # [{field, column, score, method}]
    unmapped_columns: list = field(default_factory=list)
    missing_required: list = field(default_factory=list)
    missing_optional: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required


# --------------------------------------------------------------------------- #
# 1) Data understanding
# --------------------------------------------------------------------------- #
def analyze_data(df: pd.DataFrame) -> dict:
    """Data-understanding report: shape, dtype family, missing %, uniques."""
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        else:
            kind = "text"
        info = {
            "column": str(c), "kind": kind, "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2) if n else 0.0,
            "unique": int(s.nunique(dropna=True)),
        }
        if kind == "numeric" and s.notna().any():
            info.update(min=float(s.min()), max=float(s.max()), mean=float(s.mean()))
        cols.append(info)
    return {
        "rows": int(n), "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()) if n else 0,
        "column_info": cols,
        "total_missing_cells": int(df.isna().sum().sum()),
    }


# --------------------------------------------------------------------------- #
# 2) Name matching helpers
# --------------------------------------------------------------------------- #
def _tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))     # camelCase -> words
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def _key(name: str) -> str:
    return "".join(_tokens(name))


def _tags(name: str) -> dict:
    """Detect side (orig/dest) and phase (old/new) hints in a column name."""
    toks, key = _tokens(name), _key(name)
    side = phase = None
    if any(w in key for w in ("dest", "receiv", "recip", "benef", "payee", "target", "merchant")) \
            or any(t in ("to", "dst") for t in toks):
        side = "dest"
    elif any(w in key for w in ("orig", "sender", "source", "payer")) \
            or any(t in ("from", "src") for t in toks):
        side = "orig"
    if any(w in key for w in ("old", "before", "prev", "initial", "opening")) \
            or any(t in ("pre", "start") for t in toks):
        phase = "old"
    elif any(w in key for w in ("new", "after", "final", "closing")) \
            or any(t in ("post", "end") for t in toks):
        phase = "new"
    return {"side": side, "phase": phase}


def normalize_type(value) -> str:
    """'cash out' / 'CashOut' / 'Wire' ... -> one of the model's known types."""
    if pd.isna(value):
        return "UNKNOWN"
    k = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return TYPE_MAP.get(k, str(value).strip().upper().replace(" ", "_"))


def _name_score(field_name: str, col: str) -> float:
    key = _key(col)
    if not key:
        return 0.0
    syns = SYNONYMS[field_name]
    if key in syns:
        score = 100.0
    else:
        score = 0.0
        for s in syns:
            if len(s) < 4:            # very short synonyms ('to', 'amt') only match exactly
                continue
            r = fuzz.ratio(key, s)
            if min(len(key), len(s)) >= 4:
                r = max(r, 0.9 * fuzz.partial_ratio(key, s))
            score = max(score, r)
    # side / phase awareness
    want, got = FIELD_TAGS.get(field_name, {}), _tags(col)
    for tag in ("side", "phase"):
        if want.get(tag):
            if got[tag] and got[tag] != want[tag]:
                score -= 35                      # contradicts (e.g. a 'dest' column for an origin field)
            elif got[tag] == want[tag]:
                score += 8
    if want.get("side") == "dest" and not got["side"]:
        score -= 20                              # a plain 'balance' column belongs to the sender
    return max(0.0, min(100.0, score))


# --------------------------------------------------------------------------- #
# Content validation
# --------------------------------------------------------------------------- #
def _sample(series: pd.Series, n: int = 5000) -> pd.Series:
    s = series.dropna()
    return s.sample(n, random_state=0) if len(s) > n else s


def _to_dt(s: pd.Series) -> pd.Series:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def _clean_number(v: str):
    """'1,234.50' / '1.234,50' / '123,45' / '$ 99' -> float (or NaN)."""
    v = re.sub(r"[^\d,.\-]", "", v)
    if not v or v in ("-", ".", ","):
        return np.nan
    if "," in v and "." in v:                       # the LAST separator is the decimal one
        dec = "," if v.rfind(",") > v.rfind(".") else "."
        v = v.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in v:
        v = v.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", v) else v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return np.nan


def _num(s: pd.Series) -> pd.Series:
    """pd.to_numeric that also understands thousand separators, decimal commas and currency signs."""
    num = pd.to_numeric(s, errors="coerce")
    if pd.api.types.is_numeric_dtype(s):
        return num
    todo = num.isna() & s.notna()
    if todo.any():
        num[todo] = s[todo].astype(str).map(_clean_number)
    return num


def _binary_labels(series: pd.Series):
    """Convert a label column (0/1, True/False, Fraud/Normal...) to 0/1, or None if not binary."""
    pos = {"1", "true", "yes", "y", "fraud", "fraudulent", "suspicious", "t"}
    neg = {"0", "false", "no", "n", "normal", "legit", "legitimate", "genuine", "f", "not fraud"}
    txt = series.astype(str).str.strip().str.lower()
    out = txt.map(lambda v: 1 if v in pos else (0 if v in neg else np.nan))
    if out[series.notna()].isna().mean() > 0.01:
        num = pd.to_numeric(series, errors="coerce")
        if num.notna().any() and set(num.dropna().unique()) <= {0, 1}:
            return num.fillna(0).astype(int)
        return None
    return out.fillna(0).astype(int)


def _validate(field_name: str, series: pd.Series) -> tuple[bool, str]:
    s = _sample(series)
    if s.empty:
        return False, "column is empty"
    if field_name in NUMERIC_FIELDS:
        num = _num(s)
        if num.notna().mean() < 0.9:
            return False, "not numeric"
        if field_name == "Amount" and (num.dropna() >= 0).mean() < 0.95:
            return False, "amounts are not mostly non-negative"
        return True, "numeric"
    if field_name == "Type":
        if pd.api.types.is_numeric_dtype(s) or s.nunique() > 50:
            return False, "not a low-cardinality text column"
        return True, "categorical"
    if field_name == "Time_Step":
        if pd.to_numeric(s, errors="coerce").notna().mean() >= 0.9:
            return True, "numeric"
        if _to_dt(s).notna().mean() >= 0.9:
            return True, "datetime"
        return False, "not numeric or datetime"
    if field_name == "Is_Fraud":
        if series.nunique() <= 3 and _binary_labels(series) is not None:
            return True, "binary label"
        return False, "not a binary label"
    return True, "ok"                             # IDs: anything goes


# --------------------------------------------------------------------------- #
# 2) Mapping
# --------------------------------------------------------------------------- #
def map_columns(df: pd.DataFrame, overrides: dict | None = None,
                min_score: float = 70.0) -> MappingResult:
    """Find which source column plays which role. `overrides={'Amount': 'my_col'}` forces a choice."""
    res = MappingResult()
    cols = list(df.columns)
    used: set = set()

    for f, c in (overrides or {}).items():             # manual overrides win
        if f not in STANDARD_FIELDS:
            raise MappingError(f"Unknown standard field in overrides: {f}")
        if c not in cols:
            raise MappingError(f"Override column '{c}' not found in file.")
        res.mapping[f] = c
        used.add(c)
        res.details.append({"field": f, "column": c, "score": 100, "method": "manual override"})

    # score every (field, column) pair, validate content, then assign greedily by best score
    candidates, rejected = [], []
    for f in STANDARD_FIELDS:
        if f in res.mapping:
            continue
        for c in cols:
            if c in used:
                continue
            sc = _name_score(f, c)
            if sc >= min_score:
                ok, why = _validate(f, df[c])
                if ok:
                    candidates.append((sc, f, c))
                elif sc >= 90:
                    rejected.append((f, c, why))
    for sc, f, c in sorted(candidates, key=lambda t: (-t[0], STANDARD_FIELDS.index(t[1]))):
        if f in res.mapping or c in used:
            continue
        res.mapping[f] = c
        used.add(c)
        res.details.append({"field": f, "column": c, "score": round(sc, 1),
                            "method": "exact name" if sc >= 100 else "fuzzy name"})

    for f, c, why in rejected:                   # only worth mentioning if the field stayed unmapped
        if f not in res.mapping:
            res.warnings.append(f"Column '{c}' looks like {f} by name but was rejected: {why}.")

    # content-based fallbacks (the name was useless, but the values give it away)
    if "Type" not in res.mapping:
        for c in cols:
            if c in used or pd.api.types.is_numeric_dtype(df[c]):
                continue
            s = _sample(df[c])
            if 0 < s.nunique() <= 30 and s.map(normalize_type).isin(KNOWN_TYPES).mean() >= 0.6:
                res.mapping["Type"] = c
                used.add(c)
                res.details.append({"field": "Type", "column": c, "score": 60,
                                    "method": "content (values look like transaction types)"})
                break
    if "Time_Step" not in res.mapping:
        for c in cols:
            if c in used or pd.api.types.is_numeric_dtype(df[c]):
                continue
            s = _sample(df[c])
            if len(s) and s.astype(str).str.contains(r"\d{1,4}[-/:]\d{1,2}", regex=True).mean() > 0.9 \
                    and _to_dt(s).notna().mean() >= 0.9:
                res.mapping["Time_Step"] = c
                used.add(c)
                res.details.append({"field": "Time_Step", "column": c, "score": 60,
                                    "method": "content (values look like dates)"})
                break

    res.unmapped_columns = [c for c in cols if c not in used]
    res.missing_required = [f for f in REQUIRED if f not in res.mapping]
    res.missing_optional = [f for f in STANDARD_FIELDS if f not in REQUIRED and f not in res.mapping]
    for f in res.missing_optional:
        if f in DEFAULTS:
            res.warnings.append(f"Optional field {f} not found: using default value {DEFAULTS[f]}.")
    for d in res.details:
        if d["score"] < 85:
            res.warnings.append(f"Low-confidence mapping: '{d['column']}' -> {d['field']} "
                                f"({d['score']}%). Please double-check.")
    return res


# --------------------------------------------------------------------------- #
# 3) Standardise
# --------------------------------------------------------------------------- #
def _to_time_step(s: pd.Series, warnings: list) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.9:
        warnings.append("Time column is numeric: treated as 'step' (1 step = 1 hour).")
        return num.fillna(0)
    ts = _to_dt(s)
    if ts.notna().mean() < 0.9:
        warnings.append("Time column could not be parsed: Time_Step set to 0.")
        return pd.Series(0, index=s.index)
    base = ts.min().normalize()
    step = (ts.dt.normalize() - base).dt.days * 24 + ts.dt.hour      # step % 24 == hour of day
    if ts.isna().any():
        warnings.append(f"{int(ts.isna().sum())} unparseable timestamps set to step 0.")
    return step.fillna(0)


def standardize(df: pd.DataFrame, result: MappingResult) -> tuple[pd.DataFrame, list]:
    """Return (DataFrame with standard column names, list of warnings)."""
    if not result.ok:
        raise MappingError("Could not identify required column(s): "
                           + ", ".join(result.missing_required)
                           + ". Columns found in file: " + ", ".join(map(str, df.columns)))
    warnings = list(result.warnings)
    out = pd.DataFrame(index=df.index)
    for f in STANDARD_FIELDS:
        src = result.mapping.get(f)
        if src is None:
            if f in DEFAULTS:
                out[f] = DEFAULTS[f]
            elif f in ("Customer_ID", "Dest_ID"):
                out[f] = ""
            continue
        s = df[src]
        if f in NUMERIC_FIELDS:
            num = _num(s)
            bad = int(num.isna().sum())
            if bad:
                warnings.append(f"{bad} missing/invalid values in '{src}' ({f}) were set to 0.")
            out[f] = num.fillna(0.0)
        elif f == "Type":
            out[f] = s.map(normalize_type)
            unknown = sorted(set(out[f]) - set(KNOWN_TYPES))
            if unknown:
                warnings.append("Unrecognised transaction types (no type flag set): "
                                + ", ".join(unknown[:10]))
        elif f == "Time_Step":
            out[f] = _to_time_step(s, warnings)
        elif f == "Is_Fraud":
            lab = _binary_labels(s)
            if lab is not None:
                out[f] = lab
        else:
            out[f] = s.astype(str)
    return out, warnings


def understand_and_standardize(df: pd.DataFrame, overrides: dict | None = None):
    """One call: returns (standard_df, MappingResult, data_report, warnings)."""
    report = analyze_data(df)
    res = map_columns(df, overrides)
    std, warns = standardize(df, res)
    return std, res, report, warns
