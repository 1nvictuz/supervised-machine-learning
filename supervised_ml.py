
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score
)

# 1. Load dataset
df = pd.read_csv("exampledataset1.csv")
print("Columns:", df.columns.tolist())

# Check that the final column is the correct label
target_column = df.columns[-1]
print("Target column:", target_column)

df = df.dropna(subset=[target_column]).drop_duplicates()

# 2. Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

if y.nunique() != 2:
    raise ValueError("This example requires exactly two target classes.")

# 3. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# 4. Prepare numeric and text columns
numeric = X.select_dtypes(include="number").columns.tolist()
text = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median")
)
text_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = make_column_transformer(
    (numeric_pipeline, numeric),
    (text_pipeline, text)
)

# 5. Train Logistic Regression
model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Baseline: most common training label
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# 8. Evaluate results
positive_class = model.classes_[-1]

print("\n=== RESULTS ===")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Baseline accuracy:",
      round(accuracy_score(y_test, baseline_predictions), 4))

print("Detected category:", positive_class)
print("Precision:", round(
    precision_score(y_test, predictions,
                    pos_label=positive_class, zero_division=0), 4))
print("Recall:", round(
    recall_score(y_test, predictions,
                 pos_label=positive_class, zero_division=0), 4))

print("\nConfusion Matrix:")
print("Class order:", list(model.classes_))
print(confusion_matrix(
    y_test, predictions, labels=model.classes_
))

# 9. Inspect incorrect predictions
mistakes = X_test.copy()
mistakes["Actual"] = y_test
mistakes["Predicted"] = predictions
mistakes = mistakes[
    mistakes["Actual"] != mistakes["Predicted"]
]

print("\nIncorrect Predictions:")
print(mistakes.head(5).to_string(index=False))

mistakes.to_csv("incorrect_predictions.csv", index=False)
