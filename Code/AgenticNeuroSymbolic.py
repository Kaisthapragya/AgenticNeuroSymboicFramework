# ==========================================
# NEURO-SYMBOLIC AGENTIC PIPELINE 
# Dataset: SHARPIC / Manchester CS Coordinated Diabetes Study ZIP (T1D-UOM style folders)
# Features: glucose + basal + bolus + nutrition + activity + sleep/stress + BMI
# Neural: LSTM + Transformer (choose one)
# Symbolic: upgraded (patient-adaptive thresholds + persistence + IOB/COB + nocturnal + dawn + severity score)
# Fusion: safety override for CRITICAL symbolic + (neural AND symbolic score)
# ==========================================

import os, zipfile, re, warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# 0. UPLOAD ZIP (Colab)
# -----------------------------
try:
    from google.colab import files
    print("Upload the dataset ZIP file when prompted...")
    _ = files.upload()
except Exception:
    print("Not running in Colab upload mode. Ensure ZIP is present in current directory.")

# -----------------------------
# 1. LOCATE ZIP + EXTRACT
# -----------------------------
zip_files = [f for f in os.listdir('.') if f.lower().endswith('.zip')]
if not zip_files:
    raise FileNotFoundError("No ZIP found. Upload or place it in current directory.")

ZIP_PATH = sorted(zip_files, key=lambda x: os.path.getmtime(x))[-1]
EXTRACT_PATH = "t1d_uom_extracted"
print(f"\nUsing ZIP: {ZIP_PATH}")

os.makedirs(EXTRACT_PATH, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_PATH)
print("ZIP extracted into:", EXTRACT_PATH)

# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        return pd.read_csv(path, encoding_errors="ignore")

def list_csvs(root):
    out = []
    for r, _, fs in os.walk(root):
        if "__MACOSX" in r:
            continue
        for f in fs:
            if f.lower().endswith(".csv") and not f.startswith("._"):
                out.append(os.path.join(r, f))
    return out

def pid_from_name(path):
    m = re.search(r'(\d{4})', os.path.basename(path))
    return m.group(1) if m else "P001"

def to_dt(x): return pd.to_datetime(x, errors="coerce")
def to_num(x): return pd.to_numeric(x, errors="coerce")
def bucket_5min(ts): return ts.dt.floor("5min")

def rolling_time_sum(events_df, value_col, window="2H"):
    """
    events_df must have columns: patient_id, timestamp, value_col
    returns a vector aligned to events_df rows with rolling time-based sum per patient.
    """
    if len(events_df) == 0:
        return np.array([], dtype=np.float32)
    tmp = events_df.sort_values(["patient_id", "timestamp"]).copy()
    tmp = tmp.dropna(subset=["timestamp"])
    tmp = tmp.set_index("timestamp")
    s = tmp.groupby("patient_id")[value_col].rolling(window=window).sum()
    s = s.reset_index(level=0, drop=True)
    return s.values

def asof_merge_per_patient(left, right):
    """
    Bulletproof merge_asof to avoid 'keys must be sorted' errors:
    merges within each patient separately.
    right must have patient_id + timestamp + feature columns.
    """
    if right is None or len(right) == 0:
        return left

    left = left.dropna(subset=["timestamp"]).copy()
    right = right.dropna(subset=["timestamp"]).copy()

    out_parts = []
    for pid, L in left.groupby("patient_id", sort=False):
        R = right[right["patient_id"] == pid]
        L = L.sort_values("timestamp").reset_index(drop=True)
        R = R.sort_values("timestamp").reset_index(drop=True)
        if len(R) == 0:
            out_parts.append(L)
            continue

        merged = pd.merge_asof(
            L,
            R.drop(columns=["patient_id"]),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True
        )
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return out

# -----------------------------
# 2. FIND CSV FILES
# -----------------------------
csv_files = list_csvs(EXTRACT_PATH)
if not csv_files:
    raise FileNotFoundError("No valid CSV files found.")
print("\nCSV files found (valid):", len(csv_files))

# -----------------------------
# 3. LOAD GLUCOSE (bg_ts, value)
# -----------------------------
glucose_files = [p for p in csv_files if ("glucose" in p.lower() or "bg" in p.lower())]
gl_list = []
for p in glucose_files:
    d = safe_read_csv(p)
    cols = {c.lower(): c for c in d.columns}
    if "bg_ts" in cols and "value" in cols:
        d = d.rename(columns={cols["bg_ts"]: "timestamp", cols["value"]: "glucose"})
    else:
        continue

    if "participant_id" in cols:
        d = d.rename(columns={cols["participant_id"]: "patient_id"})
    else:
        d["patient_id"] = pid_from_name(p)

    d["timestamp"] = to_dt(d["timestamp"])
    d["glucose"] = to_num(d["glucose"])
    d = d.dropna(subset=["timestamp", "glucose"])
    gl_list.append(d[["patient_id", "timestamp", "glucose"]])

if not gl_list:
    raise ValueError("Could not load glucose data (need bg_ts and value).")

glucose_df = pd.concat(gl_list, ignore_index=True).sort_values(["patient_id", "timestamp"])
print("Glucose rows:", len(glucose_df), "patients:", glucose_df["patient_id"].nunique())

# mmol/L -> mg/dL check
med = float(glucose_df["glucose"].median())
if med < 30:
    glucose_df["glucose"] = glucose_df["glucose"] * 18.0
    print(f"Converted glucose mmol/L -> mg/dL (median was {med:.2f}).")
else:
    print(f"Glucose appears mg/dL (median {med:.2f}).")

# Base 5-min grid
base = glucose_df.copy()
base["timestamp"] = bucket_5min(base["timestamp"])
base = base.groupby(["patient_id", "timestamp"], as_index=False)["glucose"].mean()
base = base.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

# -----------------------------
# 4. LOAD MULTIMODAL SOURCES (best-effort, auto if columns exist)
# -----------------------------
def load_event_table(file_match_fn, time_candidates, value_cols_map, agg="sum"):
    """
    file_match_fn: lambda path -> bool
    time_candidates: list of possible time col names (lowercase)
    value_cols_map: dict logical_name -> list possible col names (lowercase)
    agg: 'sum' or 'mean'
    returns DF: patient_id, timestamp, <logical cols...>
    """
    files = [p for p in csv_files if file_match_fn(p)]
    out = []
    for p in files:
        d = safe_read_csv(p)
        cols = {c.lower(): c for c in d.columns}

        # time
        time_col = None
        for t in time_candidates:
            if t in cols:
                time_col = cols[t]
                break
        if time_col is None:
            continue

        d = d.rename(columns={time_col: "timestamp"})
        d["patient_id"] = pid_from_name(p)

        d["timestamp"] = bucket_5min(to_dt(d["timestamp"]))
        if d["timestamp"].isna().all():
            continue

        # values
        for logical, cands in value_cols_map.items():
            found = None
            for c in cands:
                if c in cols:
                    found = cols[c]
                    break
            if found is None:
                d[logical] = 0.0
            else:
                d = d.rename(columns={found: logical})
                d[logical] = to_num(d[logical]).fillna(0.0)

        d = d.dropna(subset=["timestamp"])
        group_cols = ["patient_id", "timestamp"]
        val_cols = list(value_cols_map.keys())

        if agg == "mean":
            d2 = d.groupby(group_cols, as_index=False)[val_cols].mean()
        else:
            d2 = d.groupby(group_cols, as_index=False)[val_cols].sum()

        out.append(d2)

    if not out:
        return pd.DataFrame(columns=["patient_id", "timestamp"] + list(value_cols_map.keys()))
    return pd.concat(out, ignore_index=True).sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

# Basal
basal_df = load_event_table(
    file_match_fn=lambda p: ("basal" in p.lower()),
    time_candidates=["basal_ts", "timestamp", "time", "ts"],
    value_cols_map={"basal_dose": ["basal_dose", "dose", "basal"]},
    agg="mean"
)

# Bolus
bolus_df = load_event_table(
    file_match_fn=lambda p: ("bolus" in p.lower()),
    time_candidates=["bolus_ts", "timestamp", "time", "ts"],
    value_cols_map={"bolus_dose": ["bolus_dose", "dose", "bolus"]},
    agg="sum"
)
if len(bolus_df) > 0:
    bolus_df["bolus_2h"] = rolling_time_sum(bolus_df, "bolus_dose", "2H")
    bolus_df["bolus_4h"] = rolling_time_sum(bolus_df, "bolus_dose", "4H")
else:
    bolus_df["bolus_2h"] = []
    bolus_df["bolus_4h"] = []

# Nutrition
nut_df = load_event_table(
    file_match_fn=lambda p: ("nutrition" in p.lower() or "meal" in p.lower()),
    time_candidates=["meal_ts", "timestamp", "time", "ts"],
    value_cols_map={
        "carbs_g": ["carbs_g", "carbs", "carbohydrates"],
        "prot_g": ["prot_g", "protein_g", "protein"],
        "fat_g": ["fat_g", "fat"],
        "fibre_g": ["fibre_g", "fiber_g", "fiber"]
    },
    agg="sum"
)
if len(nut_df) > 0:
    nut_df["carbs_2h"] = rolling_time_sum(nut_df, "carbs_g", "2H")
    nut_df["carbs_4h"] = rolling_time_sum(nut_df, "carbs_g", "4H")
else:
    nut_df["carbs_2h"] = []
    nut_df["carbs_4h"] = []

# Activity
act_df = load_event_table(
    file_match_fn=lambda p: ("activity" in p.lower()),
    time_candidates=["activity_ts", "timestamp", "time", "ts"],
    value_cols_map={
        "step_count": ["step_count", "steps"],
        "active_kcal": ["active_kcal", "calories", "kcal"],
        "met": ["met"],
        "duration_s": ["duration_s", "duration"],
        "motion_intensity_mean": ["motion_intensity_mean", "intensity_mean"],
        "motion_intensity_max": ["motion_intensity_max", "intensity_max"]
    },
    agg="sum"
)
# met and intensity are better as mean/max; keep as is (still ok)

# Sleep / Stress
sleep_df = load_event_table(
    file_match_fn=lambda p: ("sleep" in p.lower() or "stress" in p.lower()),
    time_candidates=["sleep_ts", "timestamp", "time", "ts"],
    value_cols_map={
        "heart_rate": ["heart_rate", "hr"],
        "resting_heart_rate": ["resting_heart_rate", "rhr"],
        "stress_level_value": ["stress_level_value", "stress"],
        "sleep_level": ["sleep_level", "sleep_stage", "stage"]
    },
    agg="mean"
)

# BMI (static)
bmi_df = pd.DataFrame(columns=["patient_id", "bmi", "weight_kg", "height_m"])
bmi_path = next((p for p in csv_files if os.path.basename(p).lower() in ["uombmi.csv", "bmi.csv"]), None)
if bmi_path:
    b = safe_read_csv(bmi_path)
    cols = {c.lower(): c for c in b.columns}
    if "participant_id" in cols:
        b = b.rename(columns={cols["participant_id"]: "patient_id"})
    elif "patient_id" not in b.columns:
        b["patient_id"] = None

    for name, cand in [("bmi", ["bmi"]),
                       ("weight_kg", ["weight_kg", "weight"]),
                       ("height_m", ["height_m", "height"])]:
        found = None
        for c in cand:
            if c in cols:
                found = cols[c]
                break
        if found is not None:
            b = b.rename(columns={found: name})
            b[name] = to_num(b[name])
        else:
            b[name] = np.nan

    bmi_df = b[["patient_id", "bmi", "weight_kg", "height_m"]].dropna(subset=["patient_id"]).copy()
    bmi_df["patient_id"] = bmi_df["patient_id"].astype(str)

# -----------------------------
# 5. MERGE MULTIMODAL ONTO BASE
# -----------------------------
df = base.copy()
df = asof_merge_per_patient(df, basal_df)
df = asof_merge_per_patient(df, bolus_df)
df = asof_merge_per_patient(df, nut_df)
df = asof_merge_per_patient(df, act_df)
df = asof_merge_per_patient(df, sleep_df)

if len(bmi_df) > 0:
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.merge(bmi_df, on="patient_id", how="left")
else:
    df["bmi"] = 0.0
    df["weight_kg"] = 0.0
    df["height_m"] = 0.0

# numeric fill
for c in df.columns:
    if c not in ["patient_id", "timestamp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.fillna(0.0).sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

print("\nUnified DF:", df.shape, "patients:", df["patient_id"].nunique())
print("Columns:", list(df.columns))

# -----------------------------
# 6. GLUCOSE DYNAMICS FEATURES
# -----------------------------
WINDOW = 6  # 30 minutes
df["rolling_mean"] = (
    df.groupby("patient_id")["glucose"]
      .rolling(WINDOW).mean()
      .reset_index(level=0, drop=True)
).fillna(0.0)

df["rolling_std"] = (
    df.groupby("patient_id")["glucose"]
      .rolling(WINDOW).std()
      .reset_index(level=0, drop=True)
).fillna(0.0)

df["glucose_lag"] = df.groupby("patient_id")["glucose"].shift(WINDOW)
df["delta_30m"] = (df["glucose"] - df["glucose_lag"]).fillna(0.0)
df["trend"] = (df["rolling_mean"] - df["glucose"]).fillna(0.0)

# -----------------------------
# 7. LABEL
# -----------------------------
df["risk_label"] = ((df["glucose"] > 180) | (df["glucose"] < 70)).astype(int)

# -----------------------------
# 8. FEATURE COLUMNS (multimodal)
# -----------------------------
def ensure_col(name, default=0.0):
    if name not in df.columns:
        df[name] = default

for name in [
    "basal_dose",
    "bolus_dose","bolus_2h","bolus_4h",
    "carbs_g","carbs_2h","carbs_4h","prot_g","fat_g","fibre_g",
    "step_count","active_kcal","met","duration_s","motion_intensity_mean","motion_intensity_max",
    "heart_rate","resting_heart_rate","stress_level_value","sleep_level",
    "bmi","weight_kg","height_m",
]:
    ensure_col(name, 0.0)

feat_cols = [
    "glucose","rolling_mean","rolling_std","delta_30m",
    "basal_dose","bolus_dose","bolus_2h","bolus_4h",
    "carbs_g","carbs_2h","carbs_4h","prot_g","fat_g","fibre_g",
    "step_count","active_kcal","met","duration_s","motion_intensity_mean","motion_intensity_max",
    "heart_rate","resting_heart_rate","stress_level_value","sleep_level",
    "bmi","weight_kg","height_m"
]

X = df[feat_cols].values.astype(np.float32)
y = df["risk_label"].values.astype(np.int32)

# z-score
X_mean = X.mean(axis=0, keepdims=True)
X_std  = X.std(axis=0, keepdims=True) + 1e-6
Xz = (X - X_mean) / X_std

# -----------------------------
# 9. BUILD SEQUENCES PER PATIENT + SAVE TARGET ROW INDICES
# -----------------------------
SEQ_LEN = 24  # 2 hours at 5-min resolution
STEP = 1

def build_sequences_with_indices(df_sorted, X_z, y, seq_len=24, step=1):
    Xs, ys, row_idxs = [], [], []
    for pid, grp in df_sorted.groupby("patient_id", sort=False):
        idx = grp.index.to_numpy()
        for i in range(seq_len, len(idx), step):
            Xs.append(X_z[idx[i-seq_len:i]])
            ys.append(y[idx[i]])
            row_idxs.append(idx[i])   # IMPORTANT: row in df that this seq predicts
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32), np.array(row_idxs, dtype=np.int32)

X_seq, y_seq, seq_row_idx = build_sequences_with_indices(df, Xz, y, SEQ_LEN, STEP)
print("\nSequence tensor:", X_seq.shape, "Labels:", y_seq.shape)

strat = y_seq if len(np.unique(y_seq)) > 1 else None
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_seq, y_seq, seq_row_idx, test_size=0.2, random_state=42, stratify=strat
)

# -----------------------------
# 10A. LSTM MODEL
# -----------------------------
def build_lstm_model(seq_len, n_features):
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="LSTM_RiskModel")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# 10B. TRANSFORMER MODEL
# -----------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.dp1 = layers.Dropout(rate)
        self.dp2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.dp1(attn, training=training)
        out1 = self.ln1(x + attn)
        ffn = self.ffn(out1)
        ffn = self.dp2(ffn, training=training)
        return self.ln2(out1 + ffn)

def build_transformer_model(seq_len, n_features, embed_dim=32, num_heads=4, ff_dim=64):
    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.Dense(embed_dim)(inp)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="Transformer_RiskModel")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

lstm_model = build_lstm_model(SEQ_LEN, X_seq.shape[-1])
trans_model = build_transformer_model(SEQ_LEN, X_seq.shape[-1])

print("\nTraining LSTM...")
hist_lstm = lstm_model.fit(
    X_train, y_train, validation_split=0.2,
    epochs=12, batch_size=64, callbacks=[early], verbose=1
)

print("\nTraining Transformer...")
hist_trans = trans_model.fit(
    X_train, y_train, validation_split=0.2,
    epochs=12, batch_size=64, callbacks=[early], verbose=1
)

# -----------------------------
# 11. EVALUATE
# -----------------------------
def eval_model(model, X_te, y_te, name):
    probs = model.predict(X_te, verbose=0).reshape(-1)
    pred = (probs >= 0.5).astype(int)
    print(f"\n--- {name} Results ---")
    print("Confusion Matrix:\n", confusion_matrix(y_te, pred, labels=[0, 1]))
    print("\nClassification Report:\n", classification_report(y_te, pred, digits=3, zero_division=0))
    return probs, pred

_ = eval_model(lstm_model, X_test, y_test, "LSTM")
_ = eval_model(trans_model, X_test, y_test, "Transformer")

USE_MODEL = "lstm"  # "lstm" or "transformer"
chosen_model = lstm_model if USE_MODEL == "lstm" else trans_model
print("\nUsing for agent decisions:", chosen_model.name)

# -----------------------------
# 12. NEURAL PREDICTIONS -> ALIGN TO DF USING seq_row_idx (FIXES YOUR SHAPE ERROR)
# -----------------------------
df["neural_risk"] = 0
all_probs = chosen_model.predict(X_seq, verbose=0).reshape(-1)
all_pred = (all_probs >= 0.5).astype(int)

# seq_row_idx length == all_pred length, so assignment is safe:
df.loc[seq_row_idx, "neural_risk"] = all_pred

# -----------------------------
# 13. UPGRADED SYMBOLIC FEATURES (adaptive + persistence + IOB/COB + context)
# -----------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# patient robust baselines
stats = df.groupby("patient_id")["glucose"].agg(
    g_median="median",
    g_q10=lambda s: s.quantile(0.10),
    g_q90=lambda s: s.quantile(0.90),
    g_std="std"
).reset_index()
df = df.merge(stats, on="patient_id", how="left")

# adaptive thresholds
df["hypo_adapt"] = np.maximum(70.0, df["g_q10"] - 5.0)
df["hyper_adapt"] = np.minimum(180.0, df["g_q90"] + 5.0)

# time context
df["hour"] = df["timestamp"].dt.hour
df["is_night"] = df["hour"].between(0, 5)
df["is_morning"] = df["hour"].between(5, 9)

# persistence over last 60 minutes (12 * 5min)
PERSIST_STEPS = 12
df["hyper_sustained_60m"] = (
    df.groupby("patient_id")["glucose"]
      .rolling(PERSIST_STEPS).apply(lambda x: np.mean(x > 180.0), raw=True)
      .reset_index(level=0, drop=True)
).fillna(0.0)

df["hypo_sustained_60m"] = (
    df.groupby("patient_id")["glucose"]
      .rolling(PERSIST_STEPS).apply(lambda x: np.mean(x < 70.0), raw=True)
      .reset_index(level=0, drop=True)
).fillna(0.0)

# IOB / COB proxies
eps = 1e-6
df["iob_proxy"] = df["bolus_4h"] * (0.5 + 0.5 * (df["bolus_2h"] / (df["bolus_4h"] + eps)))
df["iob_proxy"] = df["iob_proxy"].fillna(0.0)

df["cob_proxy"] = df["carbs_4h"] * (0.4 + 0.6 * (df["carbs_2h"] / (df["carbs_4h"] + eps)))
df["cob_proxy"] = df["cob_proxy"].fillna(0.0)

# meal mismatch + dawn
df["carb_insulin_mismatch"] = ((df["carbs_2h"] >= 30) & (df["bolus_2h"] <= 0.5)).astype(int)
df["dawn_signal"] = ((df["is_morning"]) & (df["delta_30m"] > 20) & (df["carbs_2h"] < 10)).astype(int)

# -----------------------------
# 14. UPGRADED SYMBOLIC REASONING (explanations + score + severity)
# -----------------------------
def symbolic_reasoning(row):
    expl = []
    score = 0
    severity = "LOW"

    g = float(row["glucose"])
    d30 = float(row["delta_30m"])
    carbs2 = float(row["carbs_2h"])
    bol2 = float(row["bolus_2h"])
    met = float(row["met"])
    stress = float(row["stress_level_value"])

    hypo_thr = float(row["hypo_adapt"]) if row["hypo_adapt"] > 0 else 70.0
    hyper_thr = float(row["hyper_adapt"]) if row["hyper_adapt"] > 0 else 180.0

    hyper_persist = float(row["hyper_sustained_60m"])
    hypo_persist = float(row["hypo_sustained_60m"])

    iob = float(row["iob_proxy"])
    cob = float(row["cob_proxy"])

    is_night = bool(row["is_night"])
    dawn = int(row["dawn_signal"]) == 1
    mismatch = int(row["carb_insulin_mismatch"]) == 1

    # HARD SAFETY
    if g < 54:
        expl.append("SEVERE HYPO: glucose < 54 mg/dL (urgent).")
        score += 4
        severity = "CRITICAL"
    if g > 250:
        expl.append("SEVERE HYPER: glucose > 250 mg/dL (urgent).")
        score += 3
        severity = "CRITICAL"

    # adaptive threshold rules
    if g < hypo_thr:
        expl.append(f"Hypoglycemia: glucose < adaptive threshold ({hypo_thr:.1f}).")
        score += 2
    if g > hyper_thr:
        expl.append(f"Hyperglycemia: glucose > adaptive threshold ({hyper_thr:.1f}).")
        score += 2

    # rate-of-change rules
    if d30 <= -30:
        expl.append("Rapid fall: >30 mg/dL drop in 30 minutes.")
        score += 2
    elif d30 <= -20 and g < 90:
        expl.append("Falling trend near low range (impending hypo risk).")
        score += 1

    if d30 >= 30:
        expl.append("Rapid rise: >30 mg/dL rise in 30 minutes.")
        score += 2
    elif d30 >= 20 and g > 160:
        expl.append("Rising trend near high range (impending hyper risk).")
        score += 1

    # persistence rules
    if hyper_persist >= 0.75:
        expl.append("Sustained hyperglycemia: most of last 60 minutes > 180.")
        score += 2
    if hypo_persist >= 0.50:
        expl.append("Sustained hypoglycemia: significant portion of last 60 minutes < 70.")
        score += 2

    # causal rules
    if mismatch and g > 140 and d30 > 0:
        expl.append("Meal mismatch: carbs last 2h with low/absent bolus (missed/insufficient insulin).")
        score += 2

    if cob >= 40 and g > 140 and d30 > 0:
        expl.append("Carbs-on-board likely driving rise (high COB proxy).")
        score += 1

    if iob > 1.0 and g < 100 and d30 < 0:
        expl.append("Insulin-on-board likely driving drop (high IOB proxy).")
        score += 2

    if (met > 3.5 or float(row["active_kcal"]) > 50) and g < 100 and d30 < 0:
        expl.append("Activity likely contributing to drop (exercise + falling glucose).")
        score += 1

    if stress >= 7 and g > 140 and d30 > 0:
        expl.append("Stress-associated rise (high stress + rising glucose).")
        score += 1

    # context rules
    if is_night and g < 90 and d30 < 0:
        expl.append("Nocturnal risk: falling glucose during night hours (extra safety concern).")
        score += 2

    if dawn and g > 140 and d30 > 0:
        expl.append("Possible dawn phenomenon: morning rise without recent carbs.")
        score += 1

    # severity from score (unless critical)
    if severity != "CRITICAL":
        if score >= 5:
            severity = "HIGH"
        elif score >= 3:
            severity = "MEDIUM"
        else:
            severity = "LOW"

    return expl, score, severity

sym = df.apply(lambda r: symbolic_reasoning(r), axis=1)
df["symbolic_expl"] = sym.apply(lambda x: x[0])
df["symbolic_score"] = sym.apply(lambda x: x[1]).astype(int)
df["symbolic_severity"] = sym.apply(lambda x: x[2])

df["symbolic_hit"] = (df["symbolic_score"] >= 2).astype(int)

# -----------------------------
# 15. FUSION (SAFETY OVERRIDE + NEURAL AGREEMENT)
# -----------------------------
df["trigger_alert"] = (
    (df["symbolic_severity"] == "CRITICAL") |
    ((df["neural_risk"] == 1) & (df["symbolic_score"] >= 3))
).astype(int)

# -----------------------------
# 15B. FINAL NEURO-SYMBOLIC EVALUATION ON TEST SET
# -----------------------------
final_test_pred = df.loc[idx_test, "trigger_alert"].values.astype(int)
final_test_true = df.loc[idx_test, "risk_label"].values.astype(int)

print("\n--- Final Neuro-Symbolic Results (Test Set) ---")
print("Confusion Matrix:\n", confusion_matrix(final_test_true, final_test_pred, labels=[0, 1]))
print("\nClassification Report:\n", classification_report(final_test_true, final_test_pred, digits=3, zero_division=0))


# -----------------------------
# 16. SAMPLE OUTPUTS + SAVE CSVs
# -----------------------------
print("\n--- Agent Decisions (Sample 15) ---")
sample = df.sample(15, random_state=7)

rows = []
for _, r in sample.iterrows():
    decision = "TRIGGER_ALERT" if (int(r["trigger_alert"]) == 1) else "NO_ACTION"
    out = {
        "patient_id": r["patient_id"],
        "timestamp": r["timestamp"],
        "glucose": float(r["glucose"]),
        "neural_risk": int(r["neural_risk"]),
        "symbolic_score": int(r["symbolic_score"]),
        "symbolic_severity": r["symbolic_severity"],
        "decision": decision,
        "explanation": r["symbolic_expl"],
    }
    rows.append(out)
    print(out)

pd.DataFrame(rows).to_csv("agentic_neuro_symbolic_results_sample.csv", index=False)
print("\nSaved: agentic_neuro_symbolic_results_sample.csv")

print("\n--- Summary ---")
print("Total records:", len(df))
print("Symbolic hits:", int(df["symbolic_hit"].sum()))
print("Neural risk=1:", int((df["neural_risk"] == 1).sum()))
print("Alerts triggered:", int((df["trigger_alert"] == 1).sum()))
print("No action:", int((df["trigger_alert"] == 0).sum()))

alert_cols = [
    "patient_id","timestamp","glucose","rolling_mean","rolling_std","trend",
    "basal_dose","bolus_2h","carbs_2h","met","stress_level_value","sleep_level","bmi",
    "neural_risk","symbolic_score","symbolic_severity"
]
df.loc[df["trigger_alert"] == 1, alert_cols].to_csv("agentic_neuro_symbolic_alerts.csv", index=False)
print("Saved: agentic_neuro_symbolic_alerts.csv (all alerts)")


# ==========================================
# 17. FINAL ADVANCED AGENT (STATEFUL + ADAPTIVE)
# ==========================================

class FinalHybridAgent:
    def __init__(self):
        self.history = []
        self.patient_state = {}

    # -----------------------------
    # UTILITY FUNCTION (IMPROVED)
    # -----------------------------
    def compute_utility(self, row):
        g = float(row["glucose"])
        score = float(row["symbolic_score"])
        neural = int(row["neural_risk"])

        carbs = float(row["carbs_2h"])
        insulin = float(row["bolus_2h"])
        activity = float(row["met"]) + float(row["active_kcal"]) / 100.0
        stress = float(row["stress_level_value"])
        iob = float(row["iob_proxy"])
        cob = float(row["cob_proxy"])

        # glucose risk
        if g < 70:
            glucose_risk = (70 - g) / 70
        elif g > 180:
            glucose_risk = (g - 180) / 180
        else:
            glucose_risk = 0

        # imbalance
        imbalance = 0
        if carbs > 20 and insulin < 0.5:
            imbalance = 1
        elif insulin > 1.0 and carbs < 10:
            imbalance = 0.8

        # context risks
        activity_risk = 0.7 if (activity > 3 and g < 110) else 0
        stress_risk = 0.6 if (stress > 6 and g > 140) else 0
        iob_risk = 0.7 if (iob > 1.0 and g < 110) else 0
        cob_risk = 0.7 if (cob > 40 and g > 140) else 0

        symbolic_risk = score / 6.0
        neural_risk = float(neural)

        utility = (
            0.30 * glucose_risk +
            0.20 * symbolic_risk +
            0.15 * neural_risk +
            0.10 * imbalance +
            0.10 * activity_risk +
            0.05 * stress_risk +
            0.05 * iob_risk +
            0.05 * cob_risk
        )

        return min(1.0, utility)

    # -----------------------------
    # TEMPORAL MEMORY
    # -----------------------------
    def update_state(self, pid, utility):
        if pid not in self.patient_state:
            self.patient_state[pid] = []

        self.patient_state[pid].append(utility)

        if len(self.patient_state[pid]) > 10:
            self.patient_state[pid].pop(0)

    def get_trend(self, pid):
        hist = self.patient_state.get(pid, [])
        if len(hist) < 3:
            return 0
        return hist[-1] - hist[0]

    # -----------------------------
    # DECISION LOGIC (SMARTER)
    # -----------------------------
    def apply_rules(self, row, utility):
        g = float(row["glucose"])
        severity = row["symbolic_severity"]
        pid = row["patient_id"]

        trend = self.get_trend(pid)

        # HARD SAFETY
        if severity == "CRITICAL" or g < 54 or g > 300:
            return "EMERGENCY"

        # ESCALATION (if getting worse)
        if utility > 0.6 and trend > 0.1:
            return "ALERT"

        # STABLE REDUCTION (avoid spam alerts)
        if utility < 0.4 and trend <= 0:
            return "SAFE"

        # NORMAL THRESHOLDS
        if utility >= 0.7:
            return "ALERT"
        elif utility >= 0.4:
            return "WATCH"
        else:
            return "SAFE"

    # -----------------------------
    # ACTION POLICY
    # -----------------------------
    def act(self, row, decision):
        g = float(row["glucose"])
        carbs = float(row["carbs_2h"])
        activity = float(row["met"])

        if decision == "EMERGENCY":
            return "🚨 Immediate intervention required"

        if decision == "ALERT":
            if g > 180 and carbs > 20:
                return "💉 Insulin correction recommended"
            elif g < 80:
                return "🍬 Take fast-acting carbs"
            elif activity > 3:
                return "⚠️ Reduce activity"
            else:
                return "⚠️ Monitor closely"

        if decision == "WATCH":
            return "👀 Observe trend"

        return "✅ Stable"

    # -----------------------------
    # EXPLANATION
    # -----------------------------
    def explain(self, row, utility, trend):
        base = "; ".join(row["symbolic_expl"]) if row["symbolic_expl"] else "Normal"
        return f"{base} | Utility={utility:.2f} | Trend={trend:.2f}"

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, row):
        pid = row["patient_id"]

        utility = self.compute_utility(row)
        self.update_state(pid, utility)

        trend = self.get_trend(pid)

        decision = self.apply_rules(row, utility)
        action = self.act(row, decision)
        explanation = self.explain(row, utility, trend)

        return decision, action, explanation, utility


# ==========================================
# 18. RUN FINAL AGENT
# ==========================================
agent = FinalHybridAgent()

decisions, actions, explanations, utilities = [], [], [], []

for _, row in df.iterrows():
    d, a, e, u = agent.step(row)
    decisions.append(d)
    actions.append(a)
    explanations.append(e)
    utilities.append(u)

df["agent_decision"] = decisions
df["agent_action"] = actions
df["agent_explanation"] = explanations
df["agent_utility"] = utilities


# ==========================================
# 19. SUMMARY
# ==========================================
print("\n--- FINAL AGENT OUTPUT ---")
print(df["agent_decision"].value_counts())

alignment = np.mean(
    (df["agent_decision"].isin(["ALERT", "EMERGENCY"])) ==
    (df["trigger_alert"] == 1)
)

print("\nAlignment with Neuro-Symbolic:", alignment)


print("\n--- Sample Output ---")
print(df.sample(10)[[
    "glucose","agent_utility","agent_decision","agent_action"
]])

# -----------------------------
# 20. PLOTS 
# -----------------------------

plt.figure(figsize=(7,5))
plt.scatter(df["glucose"], df["agent_utility"], alpha=0.3)

plt.axvline(70, linestyle="--", label="Hypo")
plt.axvline(180, linestyle="--", label="Hyper")

plt.title("Agent Utility vs Glucose")
plt.xlabel("Glucose (mg/dL)")
plt.ylabel("Utility (Risk Score)")
plt.legend()
plt.tight_layout()
plt.show()



# Choose most interesting patient (max emergencies)
sample_pid = df.groupby("patient_id")["agent_decision"] \
               .apply(lambda x: (x=="EMERGENCY").sum()) \
               .idxmax()

# Filter that patient's data
p = df[df["patient_id"] == sample_pid]

plt.figure(figsize=(12,5))
plt.plot(p["timestamp"], p["glucose"], label="Glucose")

alerts = p[p["agent_decision"].isin(["ALERT","EMERGENCY"])]
plt.scatter(alerts["timestamp"], alerts["glucose"], 
            color="red", marker="x", label="Alerts")

plt.axhline(70, linestyle="--")
plt.axhline(180, linestyle="--")

plt.title(f"Patient {sample_pid} - Glucose & Agent Decisions")
plt.xlabel("Time")
plt.ylabel("Glucose")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
df["agent_decision"].value_counts().plot(kind="bar")

plt.title("Agent Decision Distribution")
plt.xlabel("Decision")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



import seaborn as sns

cm = confusion_matrix(final_test_true, final_test_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Final Neuro-Symbolic Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
