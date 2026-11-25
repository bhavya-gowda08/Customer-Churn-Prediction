\# Customer Churn Prediction Using Machine Learning



\## Project Overview

This project predicts whether a telecom customer is likely to leave a service using Machine Learning.



A Decision Tree Classification model is trained using the IBM Telco Customer Churn dataset. A Gradio web application allows users to enter customer details and receive churn predictions.



\## Objectives

\- Analyze customer churn data.

\- Perform data preprocessing and feature engineering.

\- Train a Machine Learning classification model.

\- Evaluate model performance.

\- Build an interactive prediction dashboard.



\## Technologies Used

\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Gradio

\- Joblib

\- VS Code



\## Machine Learning Workflow

1\. Dataset loading

2\. Exploratory data preparation

3\. Data preprocessing

4\. Encoding categorical variables

5\. Scaling numerical features

6\. Train-test splitting

7\. Decision Tree model training

8\. Model evaluation

9\. Model saving

10\. Gradio dashboard development



\## Model Performance



\*\*Algorithm:\*\* Decision Tree Classifier



\*\*Accuracy:\*\* 79.84%



| Metric | Churn Class |

|---|---:|

| Precision | 63% |

| Recall | 57% |

| F1-Score | 60% |



\## Project Structure



```text

Customer-Churn-Prediction/

│

├── dataset/

│   └── WA\_Fn-UseC\_-Telco-Customer-Churn.csv

│

├── app.py

├── train\_model.py

├── customer\_churn\_model.pkl

├── churn\_preprocessor.pkl

├── requirements.txt

└── README.md

```



\## Installation and Execution



\### 1. Clone the repository



```bash

git clone https://github.com/bhavya-gowda08/Customer-Churn-Prediction.git

```



\### 2. Open the project folder



```bash

cd Customer-Churn-Prediction

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Run the application



```bash

python app.py

```



\### 5. Open the dashboard



Visit:



```text

http://127.0.0.1:7860

```



Enter customer details and click \*\*Predict Churn\*\* to get the prediction.



\## Features

\- Customer churn prediction

\- Interactive Gradio dashboard

\- Saved Machine Learning model

\- Automated data preprocessing

\- Real-time predictions from entered customer details



\## Dataset

IBM Telco Customer Churn dataset containing 7,043 customer records and 21 original columns.



\## Author

Bhavya T S



Information Science and Engineering



