# Customer Churn Prediction System

A machine-learning project that predicts whether a telecom customer is likely to leave the company (churn), using the IBM Telco Customer Churn dataset.

## Project features

- Dataset inspection and exploratory data analysis (EDA)
- Missing-value handling and numeric conversion
- Categorical encoding with One-Hot Encoding
- Numeric scaling
- Logistic Regression and Decision Tree baseline models
- Evaluation with Accuracy, Precision, Recall, F1-score and ROC-AUC
- Hyperparameter tuning using GridSearchCV
- Flask web interface for individual customer predictions

## Dataset

The included file is `dataset/customer_churn.csv` (7,043 rows and 21 columns).

Target:
- `Churn = Yes`: customer churned
- `Churn = No`: customer did not churn

`customerID` is excluded from model training because it is an identifier, not a useful customer-behaviour feature. `TotalCharges` is converted from text to numeric; blank values become missing values and are imputed inside the training pipeline.

## Folder structure

```text
Customer_Churn_Prediction/
├── dataset/
│   └── customer_churn.csv
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb
├── models/
│   └── churn_pipeline.pkl       # generated after training
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup (Windows)

Open Command Prompt or the VS Code terminal inside the project folder.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you use Conda, you can instead create and activate an environment:

```bash
conda create -n churn python=3.11 -y
conda activate churn
pip install -r requirements.txt
```

## Train and evaluate the models

Run this from the main project folder:

```bash
python train_model.py
```

The script:
1. Loads and cleans the dataset.
2. Splits the data into training and test sets using stratification.
3. Trains Logistic Regression and Decision Tree pipelines.
4. Prints baseline metrics.
5. Tunes both models with 5-fold GridSearchCV using recall as the selection metric.
6. Selects the tuned model with the higher test recall and saves the full preprocessing + model pipeline to `models/churn_pipeline.pkl`.

The model file is created locally and does not need to be committed to GitHub unless you intentionally want to version the trained artifact.

## Run the Flask application

Train the model first. Then run:

```bash
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000/
```

Enter the customer details and click **Predict churn**.

## Evaluation notes

- **Recall:** Of all customers who actually churn, how many did the model identify?
- **Precision:** Of customers predicted to churn, how many actually churned?
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks churners above non-churners.

Recall is used as the GridSearchCV optimization metric because failing to identify a customer who is about to leave can be costly. In a real business, the final threshold should also consider retention-team capacity and the cost of false positives.

## Important limitations

- A prediction is a statistical estimate, not a guarantee.
- The dataset is historical and may not represent current customers or business conditions.
- Model performance should be validated on fresh data before operational use.
- A high-recall model may generate more false-positive alerts; review precision and the confusion matrix too.

## GitHub

Before uploading, ensure that `models/churn_pipeline.pkl` is generated and decide whether you want to include it. Do not upload virtual environments (`venv/`) or cache folders.

Suggested `.gitignore`:

```text
venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
```
