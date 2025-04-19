import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import os

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

data_path = project_root / "data" / "raw" / "customer_churn_dataset.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")

df = pd.read_csv(data_path)

df = df.drop("Customer_ID", axis=1)

print("Valores ausentes por coluna:\n", df.isnull().sum())

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

label_enc = LabelEncoder()
df['Gender'] = label_enc.fit_transform(df['Gender'])

X = df.drop("Churn", axis=1)
y = df["Churn"]

processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

X.to_csv(processed_dir / "X.csv", index=False)
y.to_csv(processed_dir / "y.csv", index=False)

print(f"Concluído. Arquivos salvos em: {processed_dir}")
