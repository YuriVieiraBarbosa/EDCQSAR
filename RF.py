import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ======================
# CONFIG
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dados"
OUT_DIR = BASE_DIR / "resultados"
OUT_DIR.mkdir(exist_ok=True)

ARQUIVO = DATA_DIR / "ERa_admet_preds.csv"
TARGET_COL = "pChEMBL Value"
SMILES_COL = "Smiles"
THRESHOLD_PCHEMBL = 7.0  # ativo se pChEMBL >= 7.0
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ======================
# LOAD
# ======================
if not ARQUIVO.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO}\n"
        f"Coloque o CSV em: {DATA_DIR}"
    )

df = pd.read_csv(ARQUIVO)
print("Shape original:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada.")
if SMILES_COL not in df.columns:
    raise ValueError(f"Coluna SMILES '{SMILES_COL}' não encontrada.")

# limpeza básica
df = df.copy()
df = df[df[TARGET_COL].notna()].copy()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df[df[TARGET_COL].notna()].copy()
df = df[df[SMILES_COL].notna()].copy()

print("Shape após limpeza:", df.shape)

# ======================
# CRIAR CLASSE BINÁRIA
# 1 = ativo / liga
# 0 = inativo / não liga
# ======================
df["classe"] = (df[TARGET_COL] >= THRESHOLD_PCHEMBL).astype(int)

n_total = len(df)
n_ativos = int(df["classe"].sum())
n_inativos = n_total - n_ativos

print(f"\nThreshold pChEMBL: {THRESHOLD_PCHEMBL}")
print(f"Ativos (1): {n_ativos} ({100*n_ativos/n_total:.1f}%)")
print(f"Inativos (0): {n_inativos} ({100*n_inativos/n_total:.1f}%)")

# ======================
# SMILES -> molécula RDKit
# ======================
df["mol"] = df[SMILES_COL].apply(lambda x: Chem.MolFromSmiles(str(x)))
df = df[df["mol"].notna()].copy().reset_index(drop=True)

print("Shape após remover SMILES inválidos:", df.shape)

# ======================
# DESCRITORES RDKit
# ======================
def calc_rdkit_descriptors(mol):
    try:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "ExactMolWt": Descriptors.ExactMolWt(mol),
            "MolLogP": Crippen.MolLogP(mol),
            "MolMR": Crippen.MolMR(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "NumHAcceptors": Lipinski.NumHAcceptors(mol),
            "NumHDonors": Lipinski.NumHDonors(mol),
            "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
            "RingCount": Lipinski.RingCount(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
            "NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
            "NHOHCount": Lipinski.NHOHCount(mol),
            "NOCount": Lipinski.NOCount(mol),
            "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
            "BalabanJ": Descriptors.BalabanJ(mol),
            "BertzCT": Descriptors.BertzCT(mol),
            "Chi0": Descriptors.Chi0(mol),
            "Chi1": Descriptors.Chi1(mol),
            "Chi2n": Descriptors.Chi2n(mol),
            "Chi3n": Descriptors.Chi3n(mol),
            "Chi4n": Descriptors.Chi4n(mol),
            "HallKierAlpha": Descriptors.HallKierAlpha(mol),
            "Kappa1": Descriptors.Kappa1(mol),
            "Kappa2": Descriptors.Kappa2(mol),
            "Kappa3": Descriptors.Kappa3(mol),
            "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
            "MinPartialCharge": Descriptors.MinPartialCharge(mol),
            "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(mol),
            "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge(mol),
            "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        }
    except Exception:
        return None

desc_series = df["mol"].apply(calc_rdkit_descriptors)
desc_df = pd.DataFrame(desc_series.tolist())

valid_mask = desc_df.notna().all(axis=1)
df = df.loc[valid_mask].copy().reset_index(drop=True)
desc_df = desc_df.loc[valid_mask].copy().reset_index(drop=True)

print("Shape dos descritores:", desc_df.shape)

# ======================
# MATRIZ X / y
# ======================
X = desc_df.copy()
y = df["classe"].copy()

# ======================
# SPLIT ESTRATIFICADO
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain: {X_train.shape[0]} moléculas")
print(f"Test : {X_test.shape[0]} moléculas")

# ======================
# PIPELINE RF
# ======================
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# ======================
# CROSS-VALIDATION
# ======================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(
    pipe,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("\n===== CROSS-VALIDATION (5 folds) =====")
print(f"Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
print(f"Recall   : {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
print(f"F1       : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
print(f"ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")

# ======================
# TREINO FINAL
# ======================
pipe.fit(X_train, y_train)

# ======================
# TESTE
# ======================
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)

print("\n===== RESULTADOS NO TESTE =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\n===== MATRIZ DE CONFUSÃO =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# ======================
# FEATURE IMPORTANCE
# ======================
rf_model = pipe.named_steps["model"]
fi = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n===== TOP 20 DESCRITORES =====")
print(fi.head(20).to_string(index=False))

arquivo_fi = OUT_DIR / "rf_classification_feature_importance.csv"
fi.to_csv(arquivo_fi, index=False)

# ======================
# SALVAR PREDIÇÕES
# ======================
preds = df.loc[X_test.index, [SMILES_COL, TARGET_COL]].copy()
preds["classe_real"] = y_test.values
preds["classe_predita"] = y_pred
preds["prob_ativo"] = y_proba

arquivo_preds = OUT_DIR / "rf_classification_predictions.csv"
preds.to_csv(arquivo_preds, index=False)

print("\nArquivos salvos:")
print(arquivo_fi)
print(arquivo_preds)