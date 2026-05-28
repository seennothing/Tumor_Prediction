# Multi-Modal Tumor Classification & Survival Prediction

### 설치
```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### 예측 실행 방법

**1. 테스트 파일 준비**
* `data/test/` 폴더에 평가용 테스트 `.tsv` 파일들을 위치시킵니다.
* 파일 이름에 아래 키워드가 포함되어 있으면 스크립트가 자동으로 파일을 인식하고 매핑합니다.

| 키워드 | 설명 | 사용되는 Task |
|---|---|---|
| `LUAD` / `LSCC` | 암종 구분 | 공통 |
| `protein_expression_tumor` | 종양 단백질 발현 데이터 | Task 1, Task 3 |
| `protein_expression_nat` | 정상 인접 조직 단백질 발현 | Task 1 |
| `rna_expression_tumor` | 종양 RNA 발현 데이터 | Task 2, Task 3 |

**2. 예측 실행**
전체 파이프라인을 한 번에 실행하거나, 각 Task별로 개별 실행할 수 있습니다.

**전체 자동 실행 (권장):**
```bash
python run_pipeline.py
```

**개별 실행:**
```bash
# Task 1: 종양 vs 정상 분류
python src/task1.py --mode predict

# Task 2: LUAD vs LSCC 분류
python src/task2.py --mode predict

# Task 3: 생존 예측
python src/task3.py --mode predict
```

**3. 결과 확인**
모든 예측 결과는 `results/` 폴더에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `task1_predictions.csv` | `sample_id`, `prediction` (Tumor/Normal), `probability_tumor` |
| `task2_predictions.csv` | `sample_id`, `prediction` (LUAD/LSCC), `probability_lscc` |
| `task3_predictions.csv` | `sample_id`, `prediction` (Death/Survival), `probability_death` |

---

### 모델 재학습 (선택)

학습 데이터는 `data/train/` 폴더에 위치해야 합니다. 제공된 `.pkl` 모델을 사용하지 않고 직접 가중치를 갱신하려면 아래 명령어를 순서대로 실행합니다.

```bash
# 1. 통합 전처리 진행 (필수)
python src/preprocess.py

# 2. 모델 학습
python src/task1.py --mode train
python src/task2.py --mode train
python src/task3.py --mode train
```

### 디렉토리 구조

    Tumor_Prediction/
    ├── README.md
    ├── run_pipeline.py            # 파이프라인 자동 실행 스크립트
    ├── data/
    │   ├── train/                 # 학습 데이터 폴더
    │   ├── test/                  # 평가 데이터 폴더
    │   └── processed/             # 전처리 완료된 캐시 데이터
    ├── src/
    │   ├── preprocess.py          # 생물학적 필터링 및 데이터 병합
    │   ├── task1.py               # 단백질 기반 종양/정상 분류
    │   ├── task2.py               # RNA 기반 암종 분류
    │   └── task3.py               # 멀티모달(Protein+RNA) 생존 예측
    ├── models/                    # 학습된 모델 가중치 (.pkl)
    └── results/                   # 예측 결과 출력 폴더 (.csv)
