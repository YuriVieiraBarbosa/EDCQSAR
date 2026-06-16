import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# =====================================================
# CONFIG
# =====================================================

FILE = "/home/yuri/Projetos/EDCQSAR/dados/ERa_clear.csv"

TARGET = "pChEMBL Value"
SMILES = "Smiles"

N_BITS = 2048
RADIUS = 2
RANDOM_STATE = 42
N_TRIALS = 100

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
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)

    params = {
        "kernel": kernel,
        "C": C,
        "epsilon": epsilon
    }

    if kernel in ["rbf", "poly", "sigmoid"]:
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(**params))
    ])

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

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(**best_params))
])

print("\nTreinando SVR com melhores parâmetros...")
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
# RESULTADOS DO OPTUNA
# =====================================================

best_params_df = pd.DataFrame([study.best_params])
best_params_df["best_cv_r2"] = study.best_value
best_params_df["test_r2"] = r2
best_params_df["test_mae"] = mae
best_params_df["test_rmse"] = rmse

best_params_df.to_csv(
    "optuna_svm_best_params.csv",
    index=False
)

trials_df = study.trials_dataframe()
trials_df.to_csv(
    "optuna_svm_trials.csv",
    index=False
)

# =====================================================
# REAL VS PREDITO
# =====================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    pred,
    alpha=0.6
)

mn = min(y_test.min(), pred.min())
mx = max(y_test.max(), pred.max())

plt.plot(
    [mn, mx],
    [mn, mx],
    "--"
)

plt.xlabel("pChEMBL Experimental")
plt.ylabel("pChEMBL Predito")
plt.title(f"SVR + Optuna (R²={r2:.3f})")

plt.tight_layout()

plt.savefig(
    "SVM_optuna_real_vs_predito.png",
    dpi=300
)

plt.close()

# =====================================================
# HISTOGRAMA
# =====================================================

errors = pred - y_test

plt.figure(figsize=(7, 5))

plt.hist(
    errors,
    bins=40
)

plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.title("Distribuição dos erros")

plt.tight_layout()

plt.savefig(
    "SVM_optuna_histograma_erros.png",
    dpi=300
)

plt.close()

# =====================================================
# RESÍDUOS
# =====================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    errors,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("pChEMBL Experimental")
plt.ylabel("Erro")
plt.title("Resíduos")

plt.tight_layout()

plt.savefig(
    "SVM_optuna_residuos.png",
    dpi=300
)

plt.close()

# =====================================================
# GIF - EVOLUÇÃO DO FIT COM C
# =====================================================

print("\n===================================")
print("GERANDO GIF DO SVM")
print("===================================")

frames_dir = "frames_svm_optuna_gif"
os.makedirs(frames_dir, exist_ok=True)

gif_frames = []

c_values = np.logspace(-2, 2, 12)

best_kernel = best_params["kernel"]
best_epsilon = best_params["epsilon"]
best_gamma = best_params.get("gamma", "scale")
best_degree = best_params.get("degree", 3)

mn_all = min(y_test.min(), pred.min())
mx_all = max(y_test.max(), pred.max())

for c in c_values:
    params_c = {
        "kernel": best_kernel,
        "C": float(c),
        "epsilon": best_epsilon
    }

    if best_kernel in ["rbf", "poly", "sigmoid"]:
        params_c["gamma"] = best_gamma

    if best_kernel == "poly":
        params_c["degree"] = best_degree

    model_c = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(**params_c))
    ])

    model_c.fit(X_train, y_train)
    pred_c = model_c.predict(X_test)

    r2_c = r2_score(y_test, pred_c)
    rmse_c = np.sqrt(mean_squared_error(y_test, pred_c))
    err_c = pred_c - y_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axes[0].scatter(y_test, pred_c, alpha=0.6)
    axes[0].plot([mn_all, mx_all], [mn_all, mx_all], "--")
    axes[0].set_xlabel("pChEMBL Experimental")
    axes[0].set_ylabel("pChEMBL Predito")
    axes[0].set_title(f"SVR\nC={c:.3f} | R²={r2_c:.3f}")

    axes[1].hist(err_c, bins=30)
    axes[1].set_xlabel("Erro")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title(f"Distribuição dos erros\nRMSE={rmse_c:.3f}")

    fig.suptitle("Evolução do ajuste do SVR com Optuna", fontsize=14)
    fig.tight_layout()

    frame_path = os.path.join(frames_dir, f"frame_{c:.3f}.png")
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)

    gif_frames.append(imageio.imread(frame_path))

imageio.mimsave("SVM_optuna_learning.gif", gif_frames, duration=0.45)

print("GIF gerado: SVM_optuna_learning.gif")

print("\n===================================")
print("ARQUIVOS GERADOS")
print("===================================")

print("optuna_svm_best_params.csv")
print("optuna_svm_trials.csv")
print("SVM_optuna_real_vs_predito.png")
print("SVM_optuna_histograma_erros.png")
print("SVM_optuna_residuos.png")
print("SVM_optuna_learning.gif")