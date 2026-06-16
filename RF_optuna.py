import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados"
OUT_DIR = BASE_DIR / "resultados"
OUT_DIR.mkdir(exist_ok=True)

FILE = DATA_DIR / "ERa_clear.csv"

TARGET = "pChEMBL Value"
SMILES = "Smiles"

N_BITS = 2048
RADIUS = 2
N_TRIALS = 50
RANDOM_STATE = 42

DESCRIPTOR_COLS = [
    "molecular_weight",
    "logP",
    "hydrogen_bond_acceptors",
    "hydrogen_bond_donors",
    "stereo_centers",
    "tpsa",
    "HydrationFreeEnergy_FreeSolv",
    "Solubility_AqSolDB",
]

# =====================================================
# DATASET
# =====================================================
if not FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {FILE}\n"
        f"Coloque o CSV em: {DATA_DIR}"
    )

df = pd.read_csv(FILE)

print("\n===================================")
print("DATASET")
print("===================================")
print("Arquivo:", FILE)
print("Moléculas:", len(df))

if TARGET not in df.columns:
    raise ValueError(f"Coluna alvo '{TARGET}' não encontrada.")
if SMILES not in df.columns:
    raise ValueError(f"Coluna SMILES '{SMILES}' não encontrada.")

missing_desc = [col for col in DESCRIPTOR_COLS if col not in df.columns]
if missing_desc:
    raise ValueError(
        "As seguintes colunas de descritores não foram encontradas no CSV:\n"
        + ", ".join(missing_desc)
    )

df = df.copy()
df = df[df[TARGET].notna()].copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].notna()].copy()
df = df[df[SMILES].notna()].copy()

for col in DESCRIPTOR_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=DESCRIPTOR_COLS).reset_index(drop=True)

# =====================================================
# VALIDAR SMILES
# =====================================================
valid_rows = []

for idx, smi in enumerate(df[SMILES]):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is not None:
        valid_rows.append(idx)

df = df.iloc[valid_rows].reset_index(drop=True)

print("Moléculas válidas:", len(df))

# =====================================================
# MORGAN FINGERPRINTS
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
X_desc = df[DESCRIPTOR_COLS].values

X = np.hstack([X_fp, X_desc])
y = df[TARGET].values

print("X final:", X.shape)

# =====================================================
# OPTUNA
# =====================================================
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 30)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    return scores.mean()

print("\n===================================")
print("OTIMIZAÇÃO COM OPTUNA")
print("===================================")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)

study.optimize(objective, n_trials=N_TRIALS)

print("\n===================================")
print("MELHORES HIPERPARÂMETROS")
print("===================================")
print("Best trial:", study.best_trial.number)
print("Best R² CV:", study.best_value)

for k, v in study.best_params.items():
    print(f"{k}: {v}")

# =====================================================
# SALVAR RESULTADOS
# =====================================================
trials_df = study.trials_dataframe()
trials_file = OUT_DIR / "rf_optuna_trials.csv"
trials_df.to_csv(trials_file, index=False)

best_params_file = OUT_DIR / "rf_optuna_best_params.txt"
with open(best_params_file, "w", encoding="utf-8") as f:
    f.write(f"Best trial: {study.best_trial.number}\n")
    f.write(f"Best R2 CV: {study.best_value:.6f}\n\n")
    for k, v in study.best_params.items():
        f.write(f"{k}: {v}\n")

print("\n===================================")
print("ARQUIVOS GERADOS")
print("===================================")
print(trials_file)
print(best_params_file)