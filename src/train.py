import os
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
import pickle
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
        # salvar o modelo como pkl
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        print(f"{model_name} - Accuracy: {accuracy} | AUC: {auc}")
        print(f"Model saved as pickle file at: {model_path}")
        print(report)
        
        return model

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
    
    # Treina os modelos e salva como pkl
    rf_model = train_and_log(rf, "Random Forest", X_train, X_test, y_train, y_test)
    xgb_model = train_and_log(xgb_model, "XGBoost", X_train, X_test, y_train, y_test)
    
    # Comparação entre accuracy score dos dois modelos
    if accuracy_score(y_test, rf_model.predict(X_test)) > accuracy_score(y_test, xgb_model.predict(X_test)):
        best_model_path = os.path.join(project_root, "model", "random_forest.pkl")
        best_model_name = "Random Forest"
    else:
        best_model_path = os.path.join(project_root, "model", "xgboost.pkl")
        best_model_name = "XGBoost"
    
    # Cria um arquivo generico model.pkl para ser carregado pela api utilizando o melhor modelo
    best_model_generic_path = os.path.join(project_root, "model", "model.pkl")
    with open(best_model_path, "rb") as src, open(best_model_generic_path, "wb") as dst:
        dst.write(src.read())
    
    print(f"\nO modelo ({best_model_name}) foi copiado para {best_model_generic_path} para ser consumido pela API.")

if __name__ == "__main__":
    main()