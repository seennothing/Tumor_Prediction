"""
Tumor vs. Normal Adjacent Tissue (NAT) Classification.
Modality: Protein Expression.
Architecture: PCA Dimensionality Reduction -> Logistic Regression.
Validation: GroupKFold (Patient ID).
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TEST_DIR = BASE_DIR / "data" / "test"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "task1_model.pkl"

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50, random_state=42)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

def train():
    data = joblib.load(PROCESSED_DIR / "task1_data.pkl")
    X, y, groups = data["X"], data["y"], data["groups"]
    
    pipeline = build_pipeline()
    cv = GroupKFold(n_splits=5)
    
    cv_results = cross_validate(pipeline, X, y, groups=groups, cv=cv, scoring=['accuracy', 'roc_auc'])
    print(f"[Task 1] CV ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    
    pipeline.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'features': X.columns.tolist()}, MODEL_PATH)

def predict():
    saved_model = joblib.load(MODEL_PATH)
    pipeline, train_features = saved_model['pipeline'], saved_model['features']
    
    test_files = list(TEST_DIR.glob("*protein_expression*.tsv"))
    if not test_files:
        return
        
    predictions_list = []
    for file_path in test_files:
        df = pd.read_csv(file_path, sep="\t", index_col=0).T
        X_test = df.reindex(columns=train_features)
        
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        
        predictions_list.append(pd.DataFrame({
            'sample_id': X_test.index,
            'source_file': file_path.name,
            'prediction': ['Tumor' if p == 1 else 'Normal' for p in preds],
            'probability_tumor': probs.round(4)
        }))
        
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.concat(predictions_list).to_csv(RESULTS_DIR / "task1_predictions.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    args = parser.parse_args()
    train() if args.mode == 'train' else predict()