"""
LUAD vs. LSCC Classification.
Modality: RNA Expression (Tumor only).
Architecture: SelectKBest (ANOVA) -> XGBoost.
Validation: StratifiedKFold.
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TEST_DIR = BASE_DIR / "data" / "test"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = MODEL_DIR / "task2_model.pkl"

def build_pipeline() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=100)),
        ('clf', XGBClassifier(max_depth=3, n_estimators=150, learning_rate=0.1, random_state=42, n_jobs=-1))
    ])

def train():
    data = joblib.load(PROCESSED_DIR / "task2_data.pkl")
    X, y = data["X"], data["y"]
    
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=['accuracy', 'roc_auc'])
    print(f"[Task 2] CV ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    
    pipeline.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'pipeline': pipeline, 'features': X.columns.tolist()}, MODEL_PATH)

def predict():
    saved_model = joblib.load(MODEL_PATH)
    pipeline, train_features = saved_model['pipeline'], saved_model['features']
    
    test_files = list(TEST_DIR.glob("*rna_expression_tumor*.tsv"))
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
            'prediction': ['LSCC' if p == 1 else 'LUAD' for p in preds],
            'probability_lscc': probs.round(4)
        }))
        
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.concat(predictions_list).to_csv(RESULTS_DIR / "task2_predictions.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    args = parser.parse_args()
    train() if args.mode == 'train' else predict()