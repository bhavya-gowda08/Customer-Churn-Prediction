import pandas as pd

# Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Convert target column
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("Preprocessing completed!")
print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Churn"].value_counts())

print("\nData types:")
print(df.dtypes)
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData splitting completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Numerical features:", list(numeric_features))
print("Categorical features:", list(categorical_features))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nFeature preprocessing completed!")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

model.fit(X_train_processed, y_train)

# Make predictions
y_pred = model.predict(X_test_processed)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Model Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

# Save trained model
joblib.dump(model, "customer_churn_model.pkl")

# Save preprocessing pipeline
joblib.dump(preprocessor, "churn_preprocessor.pkl")

print("\nModel saved successfully!")
print("Files: customer_churn_model.pkl and churn_preprocessor.pkl")