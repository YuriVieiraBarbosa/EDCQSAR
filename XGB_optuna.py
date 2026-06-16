import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import optuna

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

# =====================================================
# CONFIG
# =====================================================

FILE = "/home/yuri/Projetos/EDCQSAR/dados/ERa_clear.csv"
OUTPUT_DIR = "/home/yuri/Projetos/EDCQSAR/XGB"

TARGET = "pChEMBL Value"
SMILES = "Smiles"

N_BITS = 2048
RADIUS = 2
RANDOM_STATE = 42
N_TRIALS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# DATASET
# =====================================================

df = pd.read_csv(FILE)

print("\n===================================")
print("DATASET")
print("===================================")
print("Moléculas:", len(df))

# =====================================================
# MORGAN
# =====================================================

print("\n===================================")
print("GERANDO ECFP4")
print("===================================")

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=N_BITS
)

fps = []

for smi in df[SMILES]:
    mol = Chem.MolFromSmiles(str(smi))
    fp = generator.GetFingerprint(mol)
    fps.append(np.array(fp))

X_fp = np.vstack(fps)

print("Fingerprint matrix:", X_fp.shape)

# =====================================================
# DESCRITORES
# =====================================================

descriptor_cols = [
    "molecular_weight",
    "logP",
    "hydrogen_bond_acceptors",
    "hydrogen_bond_donors",
    "stereo_centers",
    "tpsa",
    "HydrationFreeEnergy_FreeSolv",
    "Solubility_AqSolDB"
]

X_desc = df[descriptor_cols].values

print("Descritores:", X_desc.shape)

# =====================================================
# CONCATENA
# =====================================================

X = np.hstack([X_fp, X_desc])
y = df[TARGET].values

print("X final:", X.shape)

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\n===================================")
print("RANDOM SPLIT")
print("===================================")

print("Treino:", len(X_train))
print("Teste :", len(X_test))

# =====================================================
# OPTUNA
# =====================================================

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist"
    }

    model = XGBRegressor(**params)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    return scores.mean()

print("\n===================================")
print("RODANDO OPTUNA")
print("===================================")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n===================================")
print("MELHOR TRIAL")
print("===================================")
print(f"Melhor R² CV = {study.best_value:.4f}")

print("\nMelhores parâmetros:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

# =====================================================
# MODELO FINAL
# =====================================================

best_params = study.best_params.copy()
best_params.update({
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist"
})

model = XGBRegressor(**best_params)

print("\nTreinando XGBoost com melhores parâmetros...")
model.fit(X_train, y_train)

# =====================================================
# TESTE
# =====================================================

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n===================================")
print("TESTE EXTERNO")
print("===================================")

print(f"R² = {r2:.4f}")
print(f"MAE = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")

# =====================================================
# CROSS VALIDATION FINAL
# =====================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="r2",
    n_jobs=-1
)

print("\n===================================")
print("5-FOLD CV")
print("===================================")

print(f"R² médio = {scores.mean():.4f}")
print(f"R² std = {scores.std():.4f}")

# =====================================================
# IMPORTÂNCIA
# =====================================================

feature_names = [f"Bit_{i}" for i in range(N_BITS)] + descriptor_cols

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n===================================")
print("TOP 30 FEATURES")
print("===================================")
print(importance.head(30).to_string(index=False))

importance.to_csv(
    os.path.join(OUTPUT_DIR, "XGB_hybrid_importance.csv"),
    index=False
)

# =====================================================
# RESULTADOS DO OPTUNA
# =====================================================

best_params_df = pd.DataFrame([study.best_params])
best_params_df["best_cv_r2"] = study.best_value
best_params_df["test_r2"] = r2
best_params_df["test_mae"] = mae
best_params_df["test_rmse"] = rmse

best_params_df.to_csv(
    os.path.join(OUTPUT_DIR, "optuna_xgb_best_params.csv"),
    index=False
)

trials_df = study.trials_dataframe()
trials_df.to_csv(
    os.path.join(OUTPUT_DIR, "optuna_xgb_trials.csv"),
    index=False
)

# =====================================================
# REAL VS PREDITO
# =====================================================

plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred, alpha=0.6)

mn = min(y_test.min(), pred.min())
mx = max(y_test.max(), pred.max())

plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("pChEMBL Experimental")
plt.ylabel("pChEMBL Predito")
plt.title(f"XGBoost Híbrido + Optuna (R²={r2:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "XGB_hybrid_real_vs_predito.png"), dpi=300)
plt.close()

# =====================================================
# HISTOGRAMA
# =====================================================

errors = pred - y_test

plt.figure(figsize=(7, 5))
plt.hist(errors, bins=40)
plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.title("Distribuição dos erros")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "XGB_hybrid_histograma_erros.png"), dpi=300)
plt.close()

# =====================================================
# RESÍDUOS
# =====================================================

plt.figure(figsize=(7, 5))
plt.scatter(y_test, errors, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("pChEMBL Experimental")
plt.ylabel("Erro")
plt.title("Resíduos")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "XGB_hybrid_residuos.png"), dpi=300)
plt.close()

# =====================================================
# IMPORTÂNCIA - TOP 20
# =====================================================

top_imp = importance.head(20).iloc[::-1]

plt.figure(figsize=(9, 7))
plt.barh(top_imp["Feature"], top_imp["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 features mais importantes")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "XGB_hybrid_top20_importance.png"), dpi=300)
plt.close()

# =====================================================
# GIF - EVOLUÇÃO DO FIT
# =====================================================

print("\n===================================")
print("GERANDO GIF DA EVOLUÇÃO DO MODELO")
print("===================================")

frames_dir = os.path.join(OUTPUT_DIR, "frames_xgb_gif")
os.makedirs(frames_dir, exist_ok=True)

frame_paths = []

n_estimators_final = int(best_params["n_estimators"])
frame_indices = np.unique(
    np.linspace(1, n_estimators_final, num=min(40, n_estimators_final), dtype=int)
)

mn_all = min(y_test.min(), pred.min())
mx_all = max(y_test.max(), pred.max())

for step in frame_indices:
    pred_step = model.predict(X_test, iteration_range=(0, step))
    r2_step = r2_score(y_test, pred_step)
    rmse_step = np.sqrt(mean_squared_error(y_test, pred_step))
    err_step = pred_step - y_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axes[0].scatter(y_test, pred_step, alpha=0.6)
    axes[0].plot([mn_all, mx_all], [mn_all, mx_all], "--")
    axes[0].set_xlabel("pChEMBL Experimental")
    axes[0].set_ylabel("pChEMBL Predito")
    axes[0].set_title(
        f"Fit do modelo\nÁrvores: {step}/{n_estimators_final} | R²={r2_step:.3f}"
    )

    axes[1].hist(err_step, bins=30)
    axes[1].set_xlabel("Erro")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title(f"Distribuição dos erros\nRMSE={rmse_step:.3f}")

    fig.suptitle("Evolução do aprendizado do XGBoost", fontsize=14)
    fig.tight_layout()

    frame_path = os.path.join(frames_dir, f"frame_{step:04d}.png")
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)

    frame_paths.append(frame_path)

gif_frames = [imageio.imread(fp) for fp in frame_paths]
imageio.mimsave(
    os.path.join(OUTPUT_DIR, "XGB_hybrid_learning.gif"),
    gif_frames,
    duration=0.35
)

print("GIF gerado com sucesso.")

print("\n===================================")
print("ARQUIVOS GERADOS")
print("===================================")

print("optuna_xgb_best_params.csv")
print("optuna_xgb_trials.csv")
print("XGB_hybrid_importance.csv")
print("XGB_hybrid_top20_importance.png")
print("XGB_hybrid_real_vs_predito.png")
print("XGB_hybrid_histograma_erros.png")
print("XGB_hybrid_residuos.png")
print("XGB_hybrid_learning.gif")