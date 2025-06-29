import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import numpy as np
import pandas as pd


# ✅ Load your saved model
try:
    model = joblib.load("solar_flare_classifier.pkl")
except Exception as e:
    print("❌ Model loading failed:", e)

# ✅ Start app
app = FastAPI(title="Solar Flare Intensity Predictor", version="1.0")

# ✅ Define the input data format
class FlareInput(BaseModel):
    duration_min: float
    region: int
    instrument: str
    north_south: str
    east_west: str

# ✅ Health check route
@app.get("/")
def root():
    return {"message": "API is running successfully"}

# ✅ Main prediction route
@app.post("/predict")
def predict_flare(data: FlareInput):
    try:
        # Step 1: Prepare input
        input_dict = {
            "duration_min": [data.duration_min],
            "region": [data.region],
            "instrument": [data.instrument],
            "north_south": [data.north_south],
            "east_west": [data.east_west]
        }

        df_input = pd.DataFrame(input_dict)
        print("🧪 Input to model:\n", df_input)

        # Step 2: Predict
        prediction = model.predict(df_input)
        print("🎯 Predicted class:", prediction[0])

        return {"predicted_class": prediction[0]}
    
    except Exception as e:
        print("❌ ERROR during prediction:", str(e))
        return {"error": str(e)}
