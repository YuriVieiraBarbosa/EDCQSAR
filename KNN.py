import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# CONFIG
# =====================================================

FILE = "/home/curso/Desktop/yurivib/ERa_clear.csv"

TARGET = "pChEMBL Value"
SMILES = "Smiles"

N_BITS = 2048
RADIUS = 2

# =====================================================
# DATASET
# =====================================================

df = pd.read_csv(FILE)

print("\n===================================")
print("DATASET")
print("===================================")

print("Moléculas:", len(df))

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

    mol = Chem.MolFromSmiles(smi)

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
    random_state=42
)

print("\n===================================")
print("RANDOM SPLIT")
print("===================================")

print("Treino:", len(X_train))
print("Teste :", len(X_test))

# =====================================================
# KNN
# =====================================================

model = Pipeline([

    ("scaler", StandardScaler()),

    ("knn", KNeighborsRegressor(

        n_neighbors=5,
        weights="distance"

    ))

])

print("\nTreinando KNN...")

model.fit(
    X_train,
    y_train
)

# =====================================================
# TESTE
# =====================================================

pred = model.predict(X_test)

r2 = r2_score(y_test, pred)

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

print("\n===================================")
print("TESTE EXTERNO")
print("===================================")

print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")

# =====================================================
# CV
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
print(f"R² std   = {scores.std():.4f}")

# =====================================================
# REAL VS PREDITO
# =====================================================

plt.figure(figsize=(7,7))

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
    f"KNN Híbrido (R²={r2:.3f})"
)

plt.tight_layout()

plt.savefig(
    "KNN_hybrid_real_vs_predito.png",
    dpi=300
)

plt.close()

# =====================================================
# HISTOGRAMA DOS ERROS
# =====================================================

errors = pred - y_test

plt.figure(figsize=(7,5))

plt.hist(
    errors,
    bins=40
)

plt.xlabel("Erro")
plt.ylabel("Frequência")

plt.title(
    "Distribuição dos erros"
)

plt.tight_layout()

plt.savefig(
    "KNN_hybrid_histograma_erros.png",
    dpi=300
)

plt.close()

# =====================================================
# RESÍDUOS
# =====================================================

plt.figure(figsize=(7,5))

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

plt.title(
    "Resíduos"
)

plt.tight_layout()

plt.savefig(
    "KNN_hybrid_residuos.png",
    dpi=300
)

plt.close()

print("\n===================================")
print("ARQUIVOS GERADOS")
print("===================================")

print("KNN_hybrid_real_vs_predito.png")
print("KNN_hybrid_histograma_erros.png")
print("KNN_hybrid_residuos.png")