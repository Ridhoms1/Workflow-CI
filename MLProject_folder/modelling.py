import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

mlflow.set_experiment("Weather_Basic_Experiment")

mlflow.autolog()

def train_basic_model():
    # Memuat Dataset
    if os.path.exists('namadataset_preprocessing/weather_processed.csv'):
        data_path = 'namadataset_preprocessing/weather_processed.csv'
    elif os.path.exists('../namadataset_preprocessing/weather_processed.csv'):
        data_path = '../namadataset_preprocessing/weather_processed.csv'
    else:
        print("Error: File 'weather_processed.csv' tidak ditemukan!")
        return

    print(f"Membaca dataset dari: {data_path}")
    df = pd.read_csv(data_path)
    
    # Memisahkan fitur dan target 
    X = df.drop(columns=['Weather Type'])
    y = df['Weather Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Memulai pelatihan model Basic...")
    
    with mlflow.start_run(run_name="RandomForest_Basic"):
        # MODEL DASAR TANPA TUNING
        model = RandomForestClassifier(random_state=42)
        
        # Proses fit ini akan otomatis mlflow.autolog() untuk mencatat semuanya
        model.fit(X_train, y_train)
        
        print("Pelatihan Kriteria Basic Selesai!")
        print("Parameter, metrik, dan model telah otomatis dicatat di lokal (folder mlruns).")

if __name__ == "__main__":
    train_basic_model()
