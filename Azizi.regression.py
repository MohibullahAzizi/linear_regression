#############
# Azizi     #
# regresion #
#############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)

#####################
# 1. Load Dataset  ##
#####################
df_raw = pd.read_csv("./NASA_JPL_Dataset.csv")
print(f"[INFO] Dataset loaded: {df_raw.shape}")
df = df_raw.copy()

# ################################################
# 2. EDA (optional sampling for large datasets) ##
# ################################################
if df.shape[0] > 50000:
    df_eda = df.sample(3000, random_state=42)
    print("[INFO] Large dataset detected — using 3,000-row sample for EDA.")
else:
    df_eda = df.copy()

num_cols = df_eda.select_dtypes(include=[np.number]).columns
if len(num_cols) > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(df_eda[num_cols].corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

if {"H","diameter"}.issubset(df_eda.columns):
    sns.scatterplot(x="H", y="diameter", data=df_eda, alpha=0.6)
    plt.title("Absolute Magnitude (H) vs Diameter [km]")
    plt.show()

# #########################
# 3. Data Preprocessing  ##
# #########################
if "diameter" not in df.columns:
    raise ValueError("Dataset must contain a 'diameter' column as target.")

train_df = df.dropna(subset=["diameter"]).copy()
X = train_df.drop(columns=["diameter"])
y = train_df["diameter"]

#######################################
# Fill numeric NaNs with column mean ##
#######################################
for c in X.select_dtypes(include=[np.number]).columns:
    X[c] = X[c].fillna(X[c].mean())

######################################
# Encode binary columns (PHA, NEO)  ##
######################################
for c in ["PHA", "NEO"]:
    if c in X.columns and X[c].dtype == object:
        X[c] = X[c].str.strip().str.upper().map({"Y":1, "YES":1, "N":0, "NO":0})
    if c in X.columns and X[c].dtype == bool:
        X[c] = X[c].astype(int)

##########################
# Drop name/id columns  ##
##########################
if "name_full" in X.columns:
    X = X.drop(columns=["name_full"])
#########################################
# One-hot encode categorical variables ##
#########################################
for c in X.select_dtypes(include=["object"]).columns:
    if X[c].nunique() < 50:
        X = pd.get_dummies(X, columns=[c], drop_first=True)
    else:
        X = X.drop(columns=[c])
######################################
# Drop constant or NaN-only columns ##
######################################
bad_cols = [c for c in X.columns if X[c].nunique() <= 1 or X[c].isna().all()]
if bad_cols:
    print("[INFO] Dropping constant/NaN columns:", bad_cols)
    X = X.drop(columns=bad_cols)

# #########################
# 4. Split & Scale Data  ##
# #########################
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train_raw)
X_test_scaled = x_scaler.transform(X_test_raw)
y_train_scaled = y_scaler.fit_transform(y_train_raw.values.reshape(-1,1)).flatten()
y_test_scaled = y_scaler.transform(y_test_raw.values.reshape(-1,1)).flatten()

def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]

X_train = add_bias(X_train_scaled)
X_test = add_bias(X_test_scaled)
y_train, y_test = y_train_scaled, y_test_scaled

print(f"[INFO] Training data: {X_train.shape}, Test data: {X_test.shape}")

# ############################################
# 5. SGD Training Function (From Scratch)   ##
##############################################
def train_sgd(
    X_train, y_train, X_val, y_val,
    epochs=1200, batch_size=64, lr_init=0.0005,
    momentum=0.9, weight_decay=1e-3, patience=60,
    decay_rate=0.001, grad_clip=10
):
    n, d = X_train.shape
    w = np.zeros(d)
    v = np.zeros(d)
    best_w, best_val = w.copy(), np.inf
    no_improve = 0
    train_curve, val_curve = [], []

    for epoch in range(epochs):
        lr = lr_init / (1 + decay_rate * epoch)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb, yb = X_train[idx[start:end]], y_train[idx[start:end]]
            err = xb @ w - yb
            grad = (2/xb.shape[0]) * xb.T @ err + 2*weight_decay*w
            norm = np.linalg.norm(grad)
            if norm > grad_clip:
                grad *= grad_clip / norm
            v = momentum*v + lr*grad
            w -= v
        tr_mse = np.mean((X_train@w - y_train)**2)
        va_mse = np.mean((X_val@w - y_val)**2)
        train_curve.append(tr_mse)
        val_curve.append(va_mse)
        if va_mse < best_val - 1e-8:
            best_val, best_w = va_mse, w.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break
    return best_w, train_curve, val_curve

# ###################
# 6. Train Model   ##
# ###################
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
w_best, tr_curve, va_curve = train_sgd(X_tr, y_tr, X_val, y_val)

plt.figure(figsize=(6,4))
plt.plot(tr_curve, label="Train MSE")
plt.plot(va_curve, label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Validation Loss (SGD)")
plt.legend()
plt.grid(True)
plt.show()

# #######################
# 7. Manual Evaluation ##
# #######################
def inverse_scale(pred_scaled):
    return y_scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

def mae_manual(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse_manual(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

y_pred_train = inverse_scale(X_train @ w_best)
y_pred_test  = inverse_scale(X_test @ w_best)
y_train_km   = inverse_scale(y_train)
y_test_km    = inverse_scale(y_test)

r2_train = r2_score_manual(y_train_km, y_pred_train)
r2_test  = r2_score_manual(y_test_km,  y_pred_test)
mae_test = mae_manual(y_test_km, y_pred_test)
mse_test = mse_manual(y_test_km, y_pred_test)

print("\n========== MODEL EVALUATION (Manual Metrics) ==========")
print(f"Train R² : {r2_train:.4f}")
print(f"Test  R² : {r2_test:.4f}")
print(f"MAE  (km): {mae_test:.4f}")
print(f"MSE  (km²): {mse_test:.4f}")

if r2_test >= 0.8:
    print("\n✅ Excellent — meets project target (R² ≥ 0.8).")
else:
    print("\n⚠️ Model can be tuned further for higher accuracy.")

# ############################################################
# 8. Retrain on Full Dataset and Predict Missing Diameters  ##
# ############################################################
print("\n[INFO] Re-training final model on all known diameters...")
X_full_scaled = x_scaler.fit_transform(X)
y_full_scaled = y_scaler.fit_transform(y.values.reshape(-1,1)).flatten()
X_full_bias = add_bias(X_full_scaled)

w_final, _, _ = train_sgd(X_full_bias, y_full_scaled, X_full_bias, y_full_scaled)
print("[INFO] Final model ready.")

mask_missing = df_raw["diameter"].isna()
num_missing = mask_missing.sum()
print(f"[INFO] Predicting for {num_missing} asteroids with missing diameters...")

X_missing = df_raw.drop(columns=["diameter"])
for c in X_missing.select_dtypes(include=[np.number]).columns:
    X_missing[c] = X_missing[c].fillna(X_missing[c].mean())
for c in ["PHA","NEO"]:
    if c in X_missing.columns and X_missing[c].dtype == object:
        X_missing[c] = X_missing[c].str.strip().str.upper().map({"Y":1,"YES":1,"N":0,"NO":0})
    if c in X_missing.columns and X_missing[c].dtype == bool:
        X_missing[c] = X_missing[c].astype(int)
if "name_full" in X_missing.columns:
    X_missing = X_missing.drop(columns=["name_full"])
for c in X_missing.select_dtypes(include=["object"]).columns:
    if X_missing[c].nunique() < 50:
        X_missing = pd.get_dummies(X_missing, columns=[c], drop_first=True)
    else:
        X_missing = X_missing.drop(columns=[c])
X_missing = X_missing.reindex(columns=X.columns, fill_value=0)
X_missing_scaled = x_scaler.transform(X_missing)
X_missing_bias = add_bias(X_missing_scaled)

y_missing_scaled = X_missing_bias @ w_final
y_missing_pred_km = y_scaler.inverse_transform(y_missing_scaled.reshape(-1,1)).flatten()

df_raw["diameter_predicted"] = df_raw["diameter"]
if num_missing > 0:
    df_raw.loc[mask_missing, "diameter_predicted"] = y_missing_pred_km[mask_missing]
else:
    df_raw["diameter_predicted"] = y_missing_pred_km
    print("[INFO] No missing diameters detected — column filled with predictions.")

df_raw.to_csv("asteroids_with_predicted_diameters.csv", index=False)
print("[✅] Saved 'asteroids_with_predicted_diameters.csv'.")

sns.histplot(df_raw["diameter_predicted"], bins=60, color="royalblue")
plt.title("Predicted Diameter Distribution [km]")
plt.xlabel("Diameter (km)")
plt.show()

print("\n🌍 All asteroid diameters estimated successfully.")
