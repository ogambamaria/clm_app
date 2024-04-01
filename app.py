from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Define a Pydantic model for the input data. Adjust the fields according to your input features.
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    # Add more fields as necessary

# Load your trained model
model = joblib.load('path/to/your/saved_model.joblib')

# Define a POST endpoint for making predictions
@app.post("/predict/")
async def make_prediction(input: ModelInput):
    try:
        # Convert input to a format your model expects. This might include scaling/normalizing if you did that during training.
        input_data = [[input.feature1, input.feature2]]  # Adjust as necessary
        prediction = model.predict(input_data)
        
        # Return the prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
