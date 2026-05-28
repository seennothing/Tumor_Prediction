"""
Data Preprocessing Engine.
Executes biological filtering, aligns feature matrices, and caches processed data for Tasks 1 & 2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def biological_filter(df: pd.DataFrame, threshold: float = 5.0, min_ratio: float = 0.2) -> pd.DataFrame:
    """Retains features expressed > threshold in at least min_ratio of samples."""
    recurrent_counts = (df > threshold).sum(axis=1)
    return df[recurrent_counts > (df.shape[1] * min_ratio)]

def load_and_clean_expression(filepath: Path, dataset_name: str) -> pd.DataFrame:
    """Loads TSV, applies biological filtering, and transposes to (Samples x Features)."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing data file: {filepath}")
        
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    original_count = df.shape[0]
    
    df = biological_filter(df)
    filtered_count = df.shape[0]
    
    print(f"  [{dataset_name}] Features filtered: {original_count} -> {filtered_count}")
    
    df = df.T
    df.index.name = "case_id"
    return df

def build_task1_data():
    print("[Preprocessing] Task 1 Data (Protein: Tumor vs NAT)")
    
    luad_t = load_and_clean_expression(TRAIN_DIR / "LUAD_trainingset_protein_expression_tumor.tsv", "LUAD_Prot_Tumor")
    luad_n = load_and_clean_expression(TRAIN_DIR / "LUAD_trainingset_protein_expression_nat.tsv", "LUAD_Prot_NAT")
    lscc_t = load_and_clean_expression(TRAIN_DIR / "LSCC_trainingset_protein_expression_tumor.tsv", "LSCC_Prot_Tumor")
    lscc_n = load_and_clean_expression(TRAIN_DIR / "LSCC_trainingset_protein_expression_nat.tsv", "LSCC_Prot_NAT")
    
    common_genes = list(set(luad_t.columns) & set(luad_n.columns) & set(lscc_t.columns) & set(lscc_n.columns))
    print(f"  -> Merged Matrix: {len(common_genes)} common protein features.")
    
    tumor_df = pd.concat([luad_t[common_genes], lscc_t[common_genes]])
    nat_df = pd.concat([luad_n[common_genes], lscc_n[common_genes]])
    
    X = pd.concat([tumor_df, nat_df])
    y = np.array([1] * len(tumor_df) + [0] * len(nat_df))
    
    # Extract core patient ID for GroupKFold validation
    groups = [pid.rsplit("-", 1)[0] if pid.count("-") > 2 else pid for pid in X.index]
    
    joblib.dump({"X": X, "y": y, "groups": groups}, PROCESSED_DIR / "task1_data.pkl")

def build_task2_data():
    print("[Preprocessing] Task 2 Data (RNA: LUAD vs LSCC)")
    
    luad_t = load_and_clean_expression(TRAIN_DIR / "LUAD_trainingset_rna_expression_tumor.tsv", "LUAD_RNA_Tumor")
    lscc_t = load_and_clean_expression(TRAIN_DIR / "LSCC_trainingset_rna_expression_tumor.tsv", "LSCC_RNA_Tumor")
    
    common_genes = list(set(luad_t.columns) & set(lscc_t.columns))
    print(f"  -> Merged Matrix: {len(common_genes)} common RNA features.")
    
    X = pd.concat([luad_t[common_genes], lscc_t[common_genes]])
    y = np.array([0] * len(luad_t) + [1] * len(lscc_t))
    
    joblib.dump({"X": X, "y": y}, PROCESSED_DIR / "task2_data.pkl")

if __name__ == "__main__":
    print("========================================")
    print("  Executing Preprocessing Engine")
    print("========================================\n")
    build_task1_data()
    print("")
    build_task2_data()
    print("\nPreprocessing complete. Matrices saved to data/processed/")