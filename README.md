# 🧠 SVM – Breast Cancer Classification (Task 11)

## 📌 Objective

Build a Support Vector Machine (SVM) model to classify breast cancer tumors as **malignant** or **benign**, compare different kernels, tune hyperparameters, and evaluate performance using ROC–AUC.

---

## 🗂️ Dataset

* **Source:** `sklearn.datasets.load_breast_cancer()`
* **Samples:** 569

## 🛠️ Tools & Libraries

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib
* Joblib

---

## 🚀 Steps Performed

1. Loaded and inspected the dataset.
2. Applied **StandardScaler** to normalize feature values (important for SVM).
3. Split data into **train** and **test** sets (80–20).
4. Trained a **baseline SVM with Linear kernel**.
5. Trained an **SVM with RBF kernel** and compared performance.
6. Used **GridSearchCV** to tune **C** and **gamma** hyperparameters.
7. Evaluated the best model using:

   * Confusion Matrix
   * Classification Report
8. Plotted **ROC Curve** and calculated **AUC score**.
9. Saved the final **Scaler + Tuned SVM model** using Joblib.

---

## 📊 Model Evaluation

* RBF kernel performed better than linear kernel.
* GridSearchCV improved model accuracy further.
* ROC curve shows strong separability between classes.
* High AUC score indicates excellent classification performance.

---

## 💾 Saved Model

The trained pipeline (Scaler + SVM) is saved as:

`svm_breast_cancer_model.pkl`

This model can be reused without retraining.

---

## 🧪 Key Concepts Learned

* Importance of feature scaling for SVM
* Difference between **Linear** and **RBF** kernels
* Role of **C** (regularization) and **gamma** (influence of data points)
* Hyperparameter tuning using GridSearchCV
* ROC–AUC for evaluating classification models

---

## 📁 Repository Contents

* `svm_breast_cancer.ipynb` — Jupyter Notebook
* `svm_breast_cancer_model.pkl` — Saved model
* `README.md` — Project documentation

---

## 🎯 Outcome

This task demonstrates practical understanding of **kernel-based classification**, **model tuning**, and **evaluation techniques** using SVM for a real-world medical dataset.
