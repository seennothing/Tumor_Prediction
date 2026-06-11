"""
Survival Prediction (Binary Classification).
Modality: Multi-modal Integration (Protein + RNA).
Architecture: Modality-Specific PCA via ColumnTransformer -> Regularized Logistic Regression.
Validation: StratifiedKFold with GridSearchCV.
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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
    # 1. Biological Noise Filter: Drop the bottom 50% of genes with lowest mean expression
    means = df.mean(axis=0, skipna=True)
    valid_genes = means[means > means.median()].index
    df_filtered = df[valid_genes]
    
    # 2. Variance Filter: Select the top_k most highly variable genes among the highly expressed ones
    top_genes = df_filtered.var(skipna=True).nlargest(top_k).index
    return df_filtered[top_genes]

def load_data():
    luad_p = _load_and_prefix(TRAIN_DIR / "LUAD_trainingset_protein_expression_tumor.tsv", "prot")
    lscc_p = _load_and_prefix(TRAIN_DIR / "LSCC_trainingset_protein_expression_tumor.tsv", "prot")
    luad_r = _load_and_prefix(TRAIN_DIR / "LUAD_trainingset_rna_expression_tumor.tsv", "rna")
    lscc_r = _load_and_prefix(TRAIN_DIR / "LSCC_trainingset_rna_expression_tumor.tsv", "rna")
    
    prot_common = list(set(luad_p.columns) & set(lscc_p.columns))
    rna_common = list(set(luad_r.columns) & set(lscc_r.columns))
    
    X_prot = _apply_variance_filter(pd.concat([luad_p[prot_common], lscc_p[prot_common]]), 1500)
    X_rna = _apply_variance_filter(pd.concat([luad_r[rna_common], lscc_r[rna_common]]), 1500)
    X_multi = pd.concat([X_prot, X_rna], axis=1)
    
    luad_surv = pd.read_csv(TRAIN_DIR / "LUAD_trainingset_overall_survival.tsv", sep="\t").set_index("case_id")
    lscc_surv = pd.read_csv(TRAIN_DIR / "LSCC_trainingset_overall_survival.tsv", sep="\t").set_index("case_id")
    y_df = pd.concat([luad_surv, lscc_surv])
    
    common_patients = list(set(X_multi.index) & set(y_df.index))
    return X_multi.loc[common_patients], y_df.loc[common_patients, "OS_event"].values

def build_pipeline() -> Pipeline:
    # Modality-specific dimensionality reduction branches
    prot_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42))
    ])
    
    rna_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42))
    ])
    
    # Dynamically route features to their respective pipelines based on prefixes
    preprocessor = ColumnTransformer([
        ('prot', prot_pipe, make_column_selector(pattern='^prot_')),
        ('rna', rna_pipe, make_column_selector(pattern='^rna_'))
    ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

def train():
    X, y = load_data()
    base_pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Hyperparameter Grid: Optimizing PCA dimensions and L2 Regularization (C)
    param_grid = {
        'preprocessor__prot__pca__n_components': [15, 30, 50],
        'preprocessor__rna__pca__n_components': [15, 30, 50],
        'clf__C': [0.01, 0.1, 1.0]
    }
    
    print("[Task 3] Executing Modality-Specific GridSearchCV...")
    grid = GridSearchCV(base_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X, y)
    
    # Extract the standard deviation of the best model for volatility tracking
    best_idx = grid.best_index_
    best_mean = grid.cv_results_['mean_test_score'][best_idx]
    best_std = grid.cv_results_['std_test_score'][best_idx]
    
    print(f"[Task 3] Best CV ROC-AUC: {best_mean:.4f} ± {best_std:.4f}")
    print(f"[Task 3] Best Hyperparameters: {grid.best_params_}")
    
    best_pipeline = grid.best_estimator_
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'pipeline': best_pipeline, 'features': X.columns.tolist()}, MODEL_PATH)

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
