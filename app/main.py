from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib

app = FastAPI()
imputer = joblib.load("imputer.pkl")
hgb = joblib.load("hgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
median_cols = joblib.load("median_columns.pkl")

@app.get("/")
def home():
    return {"message": "Exoplanet Detection API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    df = pd.read_csv(
        file.file,
        comment="#",
        on_bad_lines="skip"
    )

    # Keep exactly the 42 features used during training
    X = df[feature_columns].copy()

    # Impute only the columns that were imputed during training
    X[median_cols] = imputer.transform(X[median_cols])

    # Predict
    predictions = hgb.predict(X)

    return {
        "rows_processed": len(X),
        "predictions": predictions.tolist()
    }

@app.get("/model-info")
def model_info():
    return {
        "model": type(hgb).__name__,
        "n_features": hgb.n_features_in_
    }