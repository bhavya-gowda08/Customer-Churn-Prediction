
import gradio as gr
import pandas as pd
import joblib

# Load saved model and preprocessor
model = joblib.load("customer_churn_model.pkl")
preprocessor = joblib.load("churn_preprocessor.pkl")


# Prediction function
def predict_churn(
    gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges
):
    data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    processed_data = preprocessor.transform(data)
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        return "⚠️ Customer is likely to churn."
    else:
        return "✅ Customer is likely to stay."


# Create Gradio dashboard
with gr.Blocks(title="Customer Churn Prediction") as demo:

    gr.Markdown("# Customer Churn Prediction")
    gr.Markdown("Predict whether a customer is likely to leave the service.")

    with gr.Row():
        gender = gr.Dropdown(
            ["Female", "Male"], value="Female", label="Gender"
        )
        SeniorCitizen = gr.Dropdown(
            [0, 1], value=0, label="Senior Citizen"
        )
        Partner = gr.Dropdown(
            ["Yes", "No"], value="Yes", label="Partner"
        )
        Dependents = gr.Dropdown(
            ["Yes", "No"], value="No", label="Dependents"
        )

    with gr.Row():
        tenure = gr.Slider(
            0, 72, value=12, step=1, label="Tenure (Months)"
        )
        PhoneService = gr.Dropdown(
            ["Yes", "No"], value="Yes", label="Phone Service"
        )
        MultipleLines = gr.Dropdown(
            ["Yes", "No", "No phone service"],
            value="No", label="Multiple Lines"
        )

    with gr.Row():
        InternetService = gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            value="DSL", label="Internet Service"
        )
        OnlineSecurity = gr.Dropdown(
            ["Yes", "No", "No internet service"],
            value="No", label="Online Security"
        )
        OnlineBackup = gr.Dropdown(
            ["Yes", "No", "No internet service"],
            value="Yes", label="Online Backup"
        )

    with gr.Row():
        DeviceProtection = gr.Dropdown(
            ["Yes", "No", "No internet service"],
            value="No", label="Device Protection"
        )
        TechSupport = gr.Dropdown(
            ["Yes", "No", "No internet service"],
            value="No", label="Tech Support"
        )
        StreamingTV = gr.Dropdown(
            ["Yes", "No", "No internet service"],
            value="No", label="Streaming TV"
        )

    with gr.Row():
        StreamingMovies = gr.Dropdown(
            ["Yes", "No", "No internet service"],
            value="No", label="Streaming Movies"
        )
        Contract = gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            value="Month-to-month", label="Contract"
        )
        PaperlessBilling = gr.Dropdown(
            ["Yes", "No"], value="Yes", label="Paperless Billing"
        )

    with gr.Row():
        PaymentMethod = gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            value="Electronic check",
            label="Payment Method"
        )
        MonthlyCharges = gr.Number(
            value=70, label="Monthly Charges"
        )
        TotalCharges = gr.Number(
            value=840, label="Total Charges"
        )

    predict_button = gr.Button(
        "Predict Churn", variant="primary"
    )

    output = gr.Textbox(label="Prediction Result")

    predict_button.click(
        fn=predict_churn,
        inputs=[
            gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService,
            OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies,
            Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges
        ],
        outputs=output
    )


# Launch application
demo.launch()
