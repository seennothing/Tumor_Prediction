"""
Survival Prediction (Binary Classification).
Modality: Multi-modal Integration (Protein + RNA).
Architecture: Modality-Specific Variance Filter -> SelectKBest -> Random Forest.
Validation: StratifiedKFold.
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "task3_model.pkl"

def _load_and_prefix(filepath: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t", index_col=0).T
    df.columns = [f"{prefix}_{col}" for col in df.columns]
    df.index.name = "case_id"
    return df

def _apply_variance_filter(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return df[df.var(skipna=True).nlargest(top_k).index]

def load_data():
    luad_p = _load_and_prefix(TRAIN_DIR / "LUAD_trainingset_protein_expression_tumor.tsv", "prot")
    lscc_p = _load_and_prefix(TRAIN_DIR / "LSCC_trainingset_protein_expression_tumor.tsv", "prot")
    luad_r = _load_and_prefix(TRAIN_DIR / "LUAD_trainingset_rna_expression_tumor.tsv", "rna")
    lscc_r = _load_and_prefix(TRAIN_DIR / "LSCC_trainingset_rna_expression_tumor.tsv", "rna")
    
    prot_common = list(set(luad_p.columns) & set(lscc_p.columns))
    rna_common = list(set(luad_r.columns) & set(lscc_r.columns))
    
    X_prot = _apply_variance_filter(pd.concat([luad_p[prot_common], lscc_p[prot_common]]), 1000)
    X_rna = _apply_variance_filter(pd.concat([luad_r[rna_common], lscc_r[rna_common]]), 1000)
    X_multi = pd.concat([X_prot, X_rna], axis=1)
    
    luad_surv = pd.read_csv(TRAIN_DIR / "LUAD_trainingset_overall_survival.tsv", sep="\t").set_index("case_id")
    lscc_surv = pd.read_csv(TRAIN_DIR / "LSCC_trainingset_overall_survival.tsv", sep="\t").set_index("case_id")
    y_df = pd.concat([luad_surv, lscc_surv])
    
    common_patients = list(set(X_multi.index) & set(y_df.index))
    return X_multi.loc[common_patients], y_df.loc[common_patients, "OS_event"].values

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=100)),
        ('clf', RandomForestClassifier(max_depth=4, n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1))
    ])

def train():
    X, y = load_data()
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
    print(f"[Task 3] CV ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    
    pipeline.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'features': X.columns.tolist()}, MODEL_PATH)

def predict():
    saved_model = joblib.load(MODEL_PATH)
    pipeline, train_features = saved_model['pipeline'], saved_model['features']
    
    # 1. Dynamically locate test files
    prot_files = list(TEST_DIR.glob("*protein_expression_tumor*.tsv"))
    rna_files = list(TEST_DIR.glob("*rna_expression_tumor*.tsv"))
    
    if not prot_files or not rna_files:
        print("ERROR: Missing paired protein or RNA test files.")
        return
        
    # 2. Load all available modality data independently
    df_p_list = [_load_and_prefix(f, "prot") for f in prot_files]
    df_r_list = [_load_and_prefix(f, "rna") for f in rna_files]
    
    df_p = pd.concat(df_p_list) if df_p_list else pd.DataFrame()
    df_r = pd.concat(df_r_list) if df_r_list else pd.DataFrame()
    
    # 3. Inner join on patient IDs (case_id) to guarantee paired multi-modal data
    common_patients = list(set(df_p.index) & set(df_r.index))
    if not common_patients:
        print("ERROR: No overlapping patients found between test protein and RNA datasets.")
        return
        
    X_test_raw = pd.concat([df_p.loc[common_patients], df_r.loc[common_patients]], axis=1)
    
    # 4. Strict dimensional alignment to training features
    X_test = X_test_raw.reindex(columns=train_features)
    
    # 5. Inference
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    results = pd.DataFrame({
        'sample_id': X_test.index,
        'prediction': ['Death' if p == 1 else 'Survival' for p in preds],
        'probability_death': probs.round(4)
    })
    
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "task3_predictions.csv"
    results.to_csv(out_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    args = parser.parse_args()
    train() if args.mode == 'train' else predict()