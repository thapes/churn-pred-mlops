import os
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def load_data(processed_dir):
    X = pd.read_csv(os.path.join(processed_dir, "X.csv"))
    y = pd.read_csv(os.path.join(processed_dir, "y.csv")).squeeze().astype(int)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred)
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_text(report, "classification_report.txt")
        input_example = X_train.iloc[[0]]
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)
        print(f"{model_name} - Accuracy: {accuracy} | AUC: {auc}")
        print(report)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "data", "processed")
    X, y = load_data(processed_dir)
    X_train, X_test, y_train, y_test = split_data(X, y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    n_pos = sum(y == 1)
    n_neg = sum(y == 0)
    weight = n_neg / n_pos

    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        scale_pos_weight=weight
    )

    train_and_log(rf, "Random Forest", X_train, X_test, y_train, y_test)
    train_and_log(xgb_model, "XGBoost", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
