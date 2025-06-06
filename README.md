# 🏦 Customer Transaction Prediction

Machine Learning project for customer transaction prediction in the Kaggle challenge: **Santander Customer Transaction Prediction**.

---

## 📁 Project Structure
```
Customer-Transaction-Prediction/
│
├── .venv/ # Virtual environment (optional)
├── data/ # Challenge data and results
│   ├── sample_submission.csv
│   ├── santander-customer-transaction-prediction.zip
│   ├── submission.csv
│   ├── test.csv
│   └── train.csv
├── kaggle/
│   └── kaggle.json # Kaggle API authentication token
├── .gitignore
├── README.md # [THIS FILE]
├── requirements.txt # Required packages
└── Santander Prediction.ipynb # Main project notebook
```

---

## 📝 Project Description

This project addresses the challenge of predicting whether a customer will make a transaction, using a real and anonymized dataset provided by Santander. The main goal is to train Machine Learning models on tabular data, optimize their performance, and generate predictions for the Kaggle test set.

---

## 🚀 Notebook Workflow

Below are the main steps performed in `Santander Prediction.ipynb`:

### 1. **Library Import**

Libraries such as `numpy`, `pandas`, `matplotlib`, and `seaborn` are used for exploratory analysis, as well as `scikit-learn`, `xgboost`, and other frameworks for modeling.

### 2. **Data Loading and Exploration**

- The files `train.csv` and `test.csv` are loaded from the `/data` folder.
- An initial analysis of the data structure is performed (shape, column types, nulls, etc.).
- Visualization of the target variable and its class imbalance.
- Descriptive statistical analysis of the variables.

### 3. **Exploratory Data Analysis (EDA)**

- Analysis of the distribution of numerical variables.
- Identification and handling of outliers.
- Visualizations with `matplotlib` and `seaborn` to understand the relationship between features and the target.
- Correlation analysis between variables to avoid multicollinearity.

### 4. **Data Processing**

- There are no categorical variables; all feature columns are numerical and anonymous (`var_0` to `var_199`).
- Normalization/standardization of variables.
- Handling class imbalance (using techniques like class_weight or undersampling/oversampling if applicable).
- Splitting the training set into train/validation using `train_test_split`.

### 5. **Model Training**

- Different models are trained to compare performance:
  - **Logistic Regression**
  - **Random Forest**
  - **XGBoost**
  - **SVM**
  - **Neural Networks**
  - ¨Hyperparameter tuning pending (using GridSearchCV or RandomizedSearchCV).
- Use of cross-validation to avoid overfitting.

### 6. **Results Evaluation**

- Main metrics reported:
  - **ROC AUC Score** (main metric due to the nature of the challenge).
  - Confusion matrix.
  - Precision, Recall, F1-score (if applicable).
- Feature importance analysis.

### 7. **Prediction Generation and Submission**

- Applying the best model to the `test.csv` set.
- Generating the `submission.csv` file following the format of `sample_submission.csv`.
- Example of how to upload the result to Kaggle to evaluate the score on the leaderboard.

### 8. **Conclusions**

- Discussion of the results obtained (final score, possible improvements, model limitations, etc.).
- Possible next steps to improve performance.

---

## 📊 Sample Results

> - Best model: SVM  
> - ROC AUC validation score: **0.88**  
> - Most important variables: `var_81`, `var_108`, `var_26`...

---

## ⚙️ Installation & Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/youruser/Customer-Transaction-Prediction.git
    cd Customer-Transaction-Prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the data from [Kaggle](https://www.kaggle.com/c/santander-customer-transaction-prediction/data) and place it in the `/data` folder.

4. Run the notebook:
    - Open `Santander Prediction.ipynb` in Jupyter or VSCode.
    - Follow the cells step by step to reproduce the analysis.

---

## 📚 Resources

- [Kaggle Challenge Description](https://www.kaggle.com/c/santander-customer-transaction-prediction)
- Library docs: [scikit-learn](https://scikit-learn.org/), [xgboost](https://xgboost.readthedocs.io/)

---

## ✒️ Data Scientist Authors

[Kevin Astudillo](https://github.com/KevinAstudillo)  
[Ricardo Jaramillo](https://github.com/Ricardo-Jaramillo)

---

## 📝 License

MIT License
