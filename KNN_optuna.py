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
from sklearn.neighbors import KNeighborsRegressor

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
    n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])

    if metric == "minkowski":
        p = trial.suggest_int("p", 1, 3)
    elif metric == "euclidean":
        p = 2
    else:
        p = 1

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=-1
        ))
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

best_metric = best_params["metric"]
if best_metric == "euclidean":
    best_p = 2
elif best_metric == "manhattan":
    best_p = 1
else:
    best_p = best_params.get("p", 2)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        metric=best_metric,
        p=best_p,
        n_jobs=-1
    ))
])

print("\nTreinando KNN com melhores parâmetros...")
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

best_params_df = pd.DataFrame([{
    "n_neighbors": best_params["n_neighbors"],
    "weights": best_params["weights"],
    "metric": best_metric,
    "p": best_p,
    "best_cv_r2": study.best_value,
    "test_r2": r2,
    "test_mae": mae,
    "test_rmse": rmse
}])

best_params_df.to_csv(
    "optuna_knn_best_params.csv",
    index=False
)

trials_df = study.trials_dataframe()
trials_df.to_csv(
    "optuna_knn_trials.csv",
    index=False
)

# =====================================================
# VARREDURA DE K COM MELHORES CONFIGS FIXAS
# =====================================================

print("\n===================================")
print("VARREDURA DE K")
print("===================================")

k_values = list(range(1, 31))
k_results = []

for k in k_values:
    model_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=k,
            weights=best_params["weights"],
            metric=best_metric,
            p=best_p,
            n_jobs=-1
        ))
    ])

    cv_scores_k = cross_val_score(
        model_k,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    k_results.append({
        "k": k,
        "cv_r2_mean": cv_scores_k.mean(),
        "cv_r2_std": cv_scores_k.std()
    })

k_results_df = pd.DataFrame(k_results)
k_results_df.to_csv("KNN_optuna_k_scan.csv", index=False)

print(k_results_df.to_string(index=False))

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
plt.title(
    f"KNN + Optuna (R²={r2:.3f}, k={best_params['n_neighbors']})"
)

plt.tight_layout()

plt.savefig(
    "KNN_optuna_real_vs_predito.png",
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
    "KNN_optuna_histograma_erros.png",
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
    "KNN_optuna_residuos.png",
    dpi=300
)

plt.close()

# =====================================================
# GRÁFICO CV POR K
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_results_df["k"],
    k_results_df["cv_r2_mean"],
    marker="o"
)

plt.fill_between(
    k_results_df["k"],
    k_results_df["cv_r2_mean"] - k_results_df["cv_r2_std"],
    k_results_df["cv_r2_mean"] + k_results_df["cv_r2_std"],
    alpha=0.2
)

plt.axvline(best_params["n_neighbors"], linestyle="--")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("R² médio em CV")
plt.title("Desempenho do KNN em função de k")

plt.tight_layout()
plt.savefig("KNN_optuna_k_scan.png", dpi=300)
plt.close()

# =====================================================
# GIF - EVOLUÇÃO DO FIT COM K
# =====================================================

print("\n===================================")
print("GERANDO GIF DA EVOLUÇÃO DO KNN")
print("===================================")

frames_dir = "frames_knn_optuna_gif"
os.makedirs(frames_dir, exist_ok=True)

gif_frames = []

mn_all = min(y_test.min(), pred.min())
mx_all = max(y_test.max(), pred.max())

for k in k_values:
    model_k = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=k,
            weights=best_params["weights"],
            metric=best_metric,
            p=best_p,
            n_jobs=-1
        ))
    ])

    model_k.fit(X_train, y_train)
    pred_k = model_k.predict(X_test)

    r2_k = r2_score(y_test, pred_k)
    rmse_k = np.sqrt(mean_squared_error(y_test, pred_k))
    err_k = pred_k - y_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, pred_k, alpha=0.6)
    axes[0].plot([mn_all, mx_all], [mn_all, mx_all], "--")
    axes[0].set_xlabel("pChEMBL Experimental")
    axes[0].set_ylabel("pChEMBL Predito")
    axes[0].set_title(f"Fit do modelo\nk={k} | R²={r2_k:.3f}")

    axes[1].hist(err_k, bins=30)
    axes[1].set_xlabel("Erro")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title(f"Distribuição dos erros\nRMSE={rmse_k:.3f}")

    fig.suptitle("Evolução do ajuste do KNN com Optuna", fontsize=14)
    plt.tight_layout()

    frame_path = os.path.join(frames_dir, f"frame_{k:03d}.png")
    plt.savefig(frame_path, dpi=150, bbox_inches="tight")
    plt.close()

    gif_frames.append(imageio.imread(frame_path))

imageio.mimsave("KNN_optuna_learning.gif", gif_frames, duration=0.45)

print("GIF gerado: KNN_optuna_learning.gif")

print("\n===================================")
print("ARQUIVOS GERADOS")
print("===================================")

print("optuna_knn_best_params.csv")
print("optuna_knn_trials.csv")
print("KNN_optuna_k_scan.csv")
print("KNN_optuna_k_scan.png")
print("KNN_optuna_real_vs_predito.png")
print("KNN_optuna_histograma_erros.png")
print("KNN_optuna_residuos.png")
print("KNN_optuna_learning.gif")