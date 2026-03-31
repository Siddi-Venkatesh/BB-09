# 💳 From Data Cleaning To Deep Learning : A Full Stack Approach To Loan Default Risk Prediction In Banking Applications

> *A comprehensive machine learning research project — Department of Computer Science & Engineering*

---

## I. Team Information

| Name | Role & Contribution |
|---|---|
| **Siddi Venkatesh** | Machine Learning & Modeling Lead — Model design, LightGBM training, threshold optimization, and performance evaluation |
| **Pokala Appaiah** | Data Processing & Backend Lead — Data cleaning, preprocessing, Flask API development, and batch prediction handling |
| **Pallothu Venkata Sai Krishna** | Frontend & System Integration Lead — UI development, model integration, result visualization, and dashboard design |

---

## II. Abstract

For banks, identifying clients who might default on their loans is still a challenging undertaking. Errors are frequently present in the data they deal with, default situations are typically far fewer than non-default cases, and the relationships between the variables are rarely clear-cut.

Instead of testing a few models separately or using limited setups, this study puts together a **complete pipeline** that covers everything from cleaning and preparing the data to comparing a wide range of **machine learning and deep-learning methods** under the same experimental setup. To deal with the imbalance in the dataset, a two-step method is used: **SMOTE** is applied first to create synthetic minority samples, and then **Tomek-link removal** is used to clean up borderline cases. **SHAP** and **LIME** are also included so the reasoning behind the model predictions can be examined rather than treated as a black box.

On top of the individual models, the work also develops **ensemble versions of DenseNet and ResNet**, which regularly outperform the stand-alone versions. After multiple runs and cross-validation, the best ensemble reached a **precision of 99.2%** and showed clear improvements in recall and MCC, with significance at *p < 0.01*. The entire framework is built with real-world usage in mind, aiming to give banks both dependable predictions and explanations that make sense in practice.

---

## III. About the Project

This project implements an **end-to-end machine learning pipeline** for loan default prediction. The system takes applicant financial and demographic details as input and predicts the loan default risk. The primary goal is to build a **robust, efficient, and deployable** classification system suitable for academic research and financial risk management applications.

### Applications

- Automated credit risk assessment for financial institutions
- Loan approval decision-support systems
- Batch loan portfolio risk analysis
- Research on financial data, class imbalance, and explainable AI

---

## IV. System Workflow

```text
Applicant Input Data (Individual or Batch CSV)
  → Data Cleaning & Preprocessing
  → SMOTE Oversampling + Tomek-link Removal (Class Imbalance Handling)
  → Feature Engineering & Categorical Encoding
  → ML / DL Model Training (LightGBM, DenseNet, ResNet Ensembles)
  → SHAP & LIME Explainability Analysis
  → Threshold-Based Risk Classification
  → Risk Output: Low Risk / High Risk + Probability Score
  → Downloadable Batch Prediction Results
```

---

## V. Dataset

### Loan Default Prediction Dataset

| Property | Details |
|---|---|
| **Total Records** | ~250,000 loan applicant records |
| **Target Classes** | 2 — Low Risk (0) / High Risk (1) |
| **Format** | CSV |
| **File Path** | `Dataset/loan_dataset.csv` |

### Feature Columns

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age |
| `income` | Numeric | Annual income |
| `loan_amount` | Numeric | Requested loan amount |
| `credit_score` | Numeric | Credit score |
| `months_employed` | Numeric | Months of employment |
| `num_credit_lines` | Numeric | Number of active credit lines |
| `interest_rate` | Numeric | Loan interest rate |
| `loan_term` | Numeric | Loan term in months |
| `dti_ratio` | Numeric | Debt-to-income ratio |
| `education` | Categorical | High School / Bachelor's / Master's / PhD |
| `employment_type` | Categorical | Full-time / Part-time / Self-employed / Unemployed |
| `marital_status` | Categorical | Single / Married / Divorced |
| `loan_purpose` | Categorical | Home / Auto / Personal / Education / Business / Other |
| `has_mortgage` | Categorical | Yes / No |
| `has_dependents` | Categorical | Yes / No |
| `has_cosigner` | Categorical | Yes / No |
| `default` | Target | 0 = Low Risk, 1 = High Risk |

---

## VI. Tools & Technologies

| Category | Tools / Libraries |
|---|---|
| **Programming Language** | Python 3.10 |
| **ML Framework** | LightGBM, scikit-learn |
| **Deep Learning** | DenseNet, ResNet (ensemble variants) |
| **Explainability** | SHAP, LIME |
| **Imbalance Handling** | imbalanced-learn (SMOTE + Tomek Links) |
| **Web Framework** | Flask |
| **Data Processing** | Pandas, NumPy |
| **Model Persistence** | joblib |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Notebook** | Jupyter Notebook |
| **Development Environment** | Windows 11, Python 3.10 venv |

---

## VII. Data Preprocessing & EDA

- Dataset cleaned and normalized for consistent column naming and formats
- Categorical features encoded using label and target mapping strategies
- Boolean columns (`has_mortgage`, `has_dependents`, `has_cosigner`) standardized to `Yes/No`
- Missing values and outliers handled appropriately
- **SMOTE** applied to oversample minority (default) class
- **Tomek-link removal** applied to clean overlapping borderline samples
- Dataset split into **training, validation, and testing** sets
- EDA performed in Jupyter Notebook (`Source Code/ModelMaking/`)

---

## VIII. Model Training

- Multiple ML and DL models benchmarked under identical experimental conditions
- **LightGBM** used as primary gradient boosting classifier
- **Ensemble variants of DenseNet and ResNet** developed and compared
- Hyperparameters tuned experimentally for optimal performance
- **Prediction threshold optimized to 80%** for best real-world accuracy
- Model serialized with `joblib` for Flask deployment
- Schema stored as `model_schema.json` for runtime feature validation
- Batch prediction supported via background threading for large CSV files

---

## IX. Model Evaluation

### Metrics Used

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Reliability of High Risk predictions |
| Recall | Coverage of actual defaulters |
| F1-Score | Harmonic mean of Precision & Recall |
| ROC–AUC | Discrimination capability |
| MCC (Matthews Correlation Coefficient) | Balanced quality measure |
| Confusion Matrix | Error breakdown |

Evaluation is performed on **unseen test data** to assess generalization capability. Statistical significance tested at *p < 0.01* with cross-validation.

---

## X. Results

| Metric | Best Ensemble Result |
|---|---|
| **Precision** | **99.2%** |
| **MCC** | Significant improvement over baselines |
| **Statistical Significance** | *p < 0.01* |

- Best performance achieved by **ensemble DenseNet/ResNet** models
- Clear improvements in recall and MCC over stand-alone models
- Robust results confirmed across multiple runs and cross-validation folds

> Detailed numerical results, confusion matrices, and ROC curves are provided in the project documentation.

---

## XI. Web Application

The system includes a full-stack **Flask** web application:

| Route | Page |
|---|---|
| `/` or `/main` | Landing Page |
| `/home` | Home / Overview |
| `/predictloan` | Individual Loan Prediction Form |
| `/predictresult` | Prediction Result Display |
| `/dashboard` | Analytics Dashboard |
| `/dataset` | Dataset Info & Download |
| `/aboutus` | Team Information |
| `/contactus` | Contact Page |

### Running the Application

```bash
# Step 1 — Install dependencies
pip install flask lightgbm pandas numpy scikit-learn joblib imbalanced-learn shap lime

# Step 2 — Run the Flask app
python "Source Code/Frontend/loan_app.py"

# Step 3 — Open in browser
# http://127.0.0.1:5000/
```

> **Prerequisite:** Ensure `better_model.pkl` and `model_schema.json` are present in `Source Code/Frontend/` before running.

---

## XII. Repository Structure

```
BB-09/
├── Dataset/
│   └── loan_dataset.csv                   # Full loan dataset (~250K records)
├── Documents/
│   ├── BB-09_Abstract.pdf
│   ├── BB-09_CameraReady_Paper.pdf
│   ├── BB-09_Conference_PPT.pptx
│   ├── BB-09_Project_Documentation.pdf
│   └── BB-09_Project_PPT.pptx
└── Source Code/
    ├── Frontend/
    │   ├── loan_app.py                     # Flask backend
    │   ├── templates/                      # HTML pages (Jinja2)
    │   └── static/                         # Images & static assets
    └── ModelMaking/
        └── *.ipynb                         # Jupyter Notebooks (EDA + Training)
```

---

## XIII. Documentation

All project documentation is available in the `Documents/` folder:

- **Abstract** — `BB-09_Abstract.pdf`
- **Camera-Ready Conference Paper** — `BB-09_CameraReady_Paper.pdf`
- **Conference Presentation** — `BB-09_Conference_PPT.pptx`
- **Full Project Documentation** — `BB-09_Project_Documentation.pdf`
- **Project Presentation** — `BB-09_Project_PPT.pptx`

---

## XIV. Notes

- This project is intended for **academic and research purposes only**.
- The dataset is subject to its respective license terms.
- The system is **not a replacement for professional financial or credit advisory services**.
- SHAP and LIME analyses are available in the Jupyter Notebooks.
