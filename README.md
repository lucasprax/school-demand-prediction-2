#  Enrollment Prediction Pipeline - School Census (INEP)

> **Status:**  In Production (GPU Optimized Version)

This repository contains a robust Data Engineering and Machine Learning (ETL + ML) pipeline designed to predict school enrollment demand in Brazil. The system automates everything from downloading microdata from INEP (National Institute for Educational Studies and Research) to generating interpretable predictive models using Ensemble Learning on GPU.

---

##  Overview

The goal of this project is to process terabytes of historical data from the School Census (2011-2022+), structure complex time series, and train models to predict the number of enrolled students (`QT_MAT`) by education stage and by school.

The key differentiator of this script is its ability to handle the massive volume of Brazilian educational data using memory optimization, parallel processing, and GPU acceleration.

##  Key Features

### 1.  Automated ETL
- **Intelligent Crawler:** Automatically verifies, downloads, and extracts microdata from the INEP website.
- **Resilience:** Retry system and stream downloading with a progress bar (`tqdm`).

### 2.  High Performance
- **Multiprocessing:** Reading and *feature engineering* distributed across all CPU cores.
- **Memory Optimization:** Automatic *downcasting* of numeric types (int64 -> int16/int8) and string categorization to reduce RAM usage by up to 70%.
- **GPU Support:** Automatic CUDA detection via `torch` to accelerate XGBoost and LightGBM training.

### 3.  Advanced Machine Learning
- **Ensemble Voting:** Combination (VotingRegressor) of **LightGBM** and **XGBoost** for greater stability.
- **Walk-Forward Validation:** Respects the chronological order of data to prevent temporal *data leakage*.
- **Temporal Feature Engineering:** Automatic generation of Lags, Rolling Means, differentials, and student/teacher ratios.

### 4.  Explainability (XAI)
- **SHAP Values:** Generates feature importance plots for each trained model, allowing understanding of the "why" behind the predictions.

---

##  Tech Stack

* **Language:** Python 3.9+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`
* **Deep Learning Utils:** `torch` (for GPU context management)
* **Visualization:** `matplotlib`, `shap`
* **System:** `multiprocessing`, `glob`, `os`

---

##  Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/lucasprax/school-demand-prediction-2.git](https://github.com/lucasprax/school-demand-prediction-2.git)
   cd school-demand-prediction-2
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost shap torch matplotlib requests tqdm
   ```
   > **Note for GPU:** Ensure you install the versions of `torch`, `lightgbm`, and `xgboost` compatible with your CUDA driver if you desire hardware acceleration.

---

##  How to Run

Simply run the main script. The pipeline will manage directory creation and downloads automatically.

```bash
python run_experiment.py
```

### What the script will do:
1.  Create the folder `~/projeto_censo_escolar_v3`.
2.  Download INEP data (if not locally available).
3.  Load and optimize data in parallel.
4.  Train models iteratively for each education stage (Nursery, Pre-school, Elementary, High School, etc.).
5.  Save results in `resultados_finais.csv` and SHAP plots in the project folder.

---

##  Results Structure

The output file `resultados_finais.csv` will contain the following metrics per education stage:

| Column | Description |
| :--- | :--- |
| `etapa` | Educational segment (e.g., IN_MED, IN_FUND) |
| `modelo` | Architecture used (Voting, LGBM, or XGB) |
| `R2_Media_CV` | Coefficient of Determination (cross-validation) |
| `RMSE_Media_CV` | Root Mean Squared Error (real student scale) |
| `MAE_Media_CV` | Mean Absolute Error |

---

##  Important Notes

* **Hardware Requirements:** Due to the size of the School Census, at least **16GB of RAM** is recommended. The script attempts to adjust memory usage, but processing all years (2011-2022) is intensive.
* **Logs:** Execution generates an `execution_log_no_checkpoint.txt` file for error auditing and progress tracking.

---

##  License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  <sub>Developed for research purposes in Educational Data Science.</sub>
</p>
