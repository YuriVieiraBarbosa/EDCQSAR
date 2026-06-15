import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
print("Fingerprint matrix:", X_fp.shape)

# =====================================================
# DESCRITORES
# =====================================================
X_desc = df[DESCRIPTOR_COLS].values
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
    random_state=42
)

print("\n===================================")
print("RANDOM SPLIT")
print("===================================")
print("Treino:", len(X_train))
print("Teste :", len(X_test))

# =====================================================
# RANDOM FOREST
# =====================================================
model = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("\nTreinando Random Forest...")
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
# CROSS VALIDATION
# =====================================================
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
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
# IMPORTÂNCIA DAS FEATURES
# =====================================================
feature_names = [f"Bit_{i}" for i in range(N_BITS)] + DESCRIPTOR_COLS

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n===================================")
print("TOP 30 FEATURES")
print("===================================")
print(importance.head(30).to_string(index=False))

importance_file = OUT_DIR / "RF_hybrid_importance.csv"
importance.to_csv(importance_file, index=False)

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
plt.title(f"RF Híbrido (R²={r2:.3f})")
plt.tight_layout()

real_vs_pred_file = OUT_DIR / "RF_hybrid_real_vs_predito.png"
plt.savefig(real_vs_pred_file, dpi=300)
plt.close()

# =====================================================
# HISTOGRAMA DOS ERROS
# =====================================================
errors = pred - y_test

plt.figure(figsize=(7, 5))
plt.hist(errors, bins=40)
plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.title("Distribuição dos erros")
plt.tight_layout()

hist_file = OUT_DIR / "RF_hybrid_histograma_erros.png"
plt.savefig(hist_file, dpi=300)
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

resid_file = OUT_DIR / "RF_hybrid_residuos.png"
plt.savefig(resid_file, dpi=300)
plt.close()

print("\n===================================")
print("ARQUIVOS GERADOS")
print("===================================")
print(importance_file)
print(real_vs_pred_file)
print(hist_file)
print(resid_file)