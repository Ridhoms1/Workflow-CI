import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_training():
    data_path = 'namadataset_preprocessing/weather_processed.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Weather Type'])
    y = df['Weather Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="CI_Pipeline_Run") as run:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        
        # Log model ke dalam folder "model"
        mlflow.sklearn.log_model(rf, "model")
        
        # Simpan RUN_ID ke file teks agar bisa dibaca oleh GitHub Actions untuk build Docker
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    run_training()