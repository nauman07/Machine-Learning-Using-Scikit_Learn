# 🤖 Machine Learning Using Scikit-Learn

A practical guide to building **machine learning models using Scikit-Learn**, covering supervised and unsupervised learning algorithms with real-world examples.

---

## 📖 About

This repository provides a comprehensive introduction to machine learning with Python's most popular ML library — `scikit-learn`. From data preprocessing to model evaluation, it covers the entire ML pipeline in a beginner-to-intermediate friendly format.

---

## ✨ Topics Covered

### Supervised Learning
- **Linear Regression** — Predicting continuous values
- **Logistic Regression** — Binary and multi-class classification
- **Decision Trees** — Rule-based classification and regression
- **Random Forest** — Ensemble learning for improved accuracy
- **Support Vector Machines (SVM)** — Margin-based classification
- **K-Nearest Neighbors (KNN)**

### Unsupervised Learning
- **K-Means Clustering** — Grouping data without labels
- **PCA** — Dimensionality reduction

### Model Evaluation
- Train/test split and cross-validation
- Accuracy, precision, recall, F1-score
- Confusion matrix and ROC curve

### Data Preprocessing
- Feature scaling (StandardScaler, MinMaxScaler)
- Handling missing values
- Encoding categorical variables

---

## 🛠️ Tech Stack

| Library | Purpose |
|--------|---------|
| `scikit-learn` | Machine learning models and utilities |
| `Pandas` | Data manipulation |
| `NumPy` | Numerical computing |
| `Matplotlib` / `Seaborn` | Visualization |
| `Jupyter Notebook` | Interactive environment |

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Run the Notebooks

```bash
git clone https://github.com/nauman07/Machine-Learning-Using-Scikit_Learn.git
cd Machine-Learning-Using-Scikit_Learn
jupyter notebook
```

---

## 📝 Quick Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
