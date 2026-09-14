# Placement Prediction — End-to-End ML Pipeline

A beginner-friendly, end-to-end machine learning project that predicts whether a student
will be placed based on their **CGPA** and **IQ**. Built to practice the full ML workflow —
from raw data to a deployable, pickled model — using Logistic Regression.

## 📌 Project Overview

This project walks through the standard steps of a supervised ML pipeline:

1. **Data Preprocessing** — load, clean, and validate the dataset (nulls, duplicates)
2. **Exploratory Data Analysis (EDA)** — class balance, distributions, outliers, correlation
3. **Feature/Target Extraction** — separate input (`cgpa`, `iq`) from output (`placement`)
4. **Train/Test Split** — reproducible 80/20 split (`random_state` fixed, stratified)
5. **Feature Scaling** — standardize features with `StandardScaler`
6. **Model Selection** — compare Logistic Regression, SVM, KNN, and Decision Tree via 5-fold cross-validation
7. **Hyperparameter Tuning** — `GridSearchCV` on the best-performing model
8. **Model Evaluation** — accuracy, confusion matrix, precision/recall/F1, ROC-AUC, decision boundary plot
9. **Model Export** — serialize the trained model *and* scaler with `pickle`
10. **Deployment** — `predict.py` script for command-line inference on new inputs

## 🗂️ Project Structure

```
placement-prediction-end-to-end-ml/
├── data/
│   └── placement.csv          # dataset (cgpa, iq, placement)
├── notebooks/
│   └── PlacementPredictionML.ipynb        # main notebook (this project)
├── model/
│   ├── model.pkl              # trained model (generated after running the notebook)
│   └── scaler.pkl             # fitted StandardScaler (generated after running the notebook)
├── predict.py                 # CLI script for inference using the saved model
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** Move `PlacementPredictionML.ipynb` into a `notebooks/` folder and place `placement.csv`
> inside `downloads/` before pushing, or update the file paths in the notebook (`pd.read_csv(...)`)
> to match wherever you keep them.

## 📊 Dataset

The dataset (`placement.csv`) contains 100 records with the following columns:

| Column     | Description                          |
|------------|---------------------------------------|
| `cgpa`     | Student's CGPA                        |
| `iq`       | Student's IQ score                    |
| `placement`| Target: 1 = placed, 0 = not placed    |

*(An unnamed index column from the original CSV is dropped during preprocessing.)*

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/shantanugupta53/placement-prediction-end-to-end-ml.git
cd placement-prediction-end-to-end-ml

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

Run the notebook end-to-end:

```bash
jupyter notebook notebooks/PlacementPredictionML.ipynb
```

This will train the model and save `model.pkl` and `scaler.pkl` to the `model/` folder.

Then run predictions from the command line:

```bash
python predict.py --cgpa 6.8 --iq 123
```

```
Prediction: Placed
Confidence (probability of placement): 87.42%
```

## 📈 Results

- **Models compared:** Logistic Regression, SVM (RBF), KNN, Decision Tree — via 5-fold cross-validation
- **Selection:** best model tuned further with `GridSearchCV`
- **Evaluation:** accuracy, precision/recall/F1, confusion matrix, and ROC-AUC on a held-out, stratified 20% test set

> The dataset is small (~100 rows), so all metrics should be read as a demonstration of the
> pipeline rather than production-grade performance. Cross-validation and a fixed
> `random_state` make the numbers reproducible run-to-run, but a larger, more diverse
> dataset would be needed for a real deployment.

## 🔧 Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- mlxtend (for decision boundary visualization)

## 🛣️ Possible Improvements

- Wrap `predict.py` in a Streamlit or Flask app for a browser-based demo
- Add unit tests and a CI workflow (GitHub Actions)
- Expand the dataset for more robust, less overfitting-prone evaluation
- Try ensemble methods (Random Forest, Gradient Boosting) in the model comparison
- Add SHAP or feature-importance analysis to explain predictions

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙋 Author
