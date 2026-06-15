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

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor

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
# XGBOOST
# =====================================================

model = XGBRegressor(

    n_estimators=500,
    max_depth=8,

    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,
    n_jobs=-1

)

print("\nTreinando XGBoost...")

model.fit(
    X_train,
    y_train
)

# =====================================================
# TESTE
# =====================================================

pred = model.predict(X_test)

r2 = r2_score(
    y_test,
    pred
)

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
print(f"R² std   = {scores.std():.4f}")

# =====================================================
# IMPORTÂNCIA
# =====================================================

feature_names = (
    [f"Bit_{i}" for i in range(N_BITS)]
    + descriptor_cols
)

importance = pd.DataFrame({

    "Feature": feature_names,
    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===================================")
print("TOP 30 FEATURES")
print("===================================")

print(
    importance.head(30).to_string(index=False)
)

importance.to_csv(
    "XGB_hybrid_importance.csv",
    index=False
)

# =====================================================
# SHAP
# =====================================================

import shap

print("\n===================================")
print("CALCULANDO SHAP")
print("===================================")

sample_size = min(500, len(X_test))

idx = np.random.choice(
    len(X_test),
    sample_size,
    replace=False
)

X_shap = X_test[idx]

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_shap)

# =====================================================
# SHAP SUMMARY
# =====================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "XGB_hybrid_SHAP_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# SHAP BAR
# =====================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "XGB_hybrid_SHAP_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# SHAP CSV
# =====================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({

    "Feature": feature_names,
    "MeanAbsSHAP": mean_abs_shap

})

shap_df = shap_df.sort_values(
    by="MeanAbsSHAP",
    ascending=False
)

shap_df.to_csv(
    "XGB_hybrid_SHAP_importance.csv",
    index=False
)

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
    f"XGBoost Híbrido (R²={r2:.3f})"
)

plt.tight_layout()

plt.savefig(
    "XGB_hybrid_real_vs_predito.png",
    dpi=300
)

plt.close()

# =====================================================
# HISTOGRAMA
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
    "XGB_hybrid_histograma_erros.png",
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
    "XGB_hybrid_residuos.png",
    dpi=300
)

plt.close()

print("\n===================================")
print("ARQUIVOS GERADOS")
print("===================================")

print("XGB_hybrid_importance.csv")
print("XGB_hybrid_SHAP_importance.csv")
print("XGB_hybrid_SHAP_summary.png")
print("XGB_hybrid_SHAP_bar.png")
print("XGB_hybrid_real_vs_predito.png")
print("XGB_hybrid_histograma_erros.png")
print("XGB_hybrid_residuos.png")