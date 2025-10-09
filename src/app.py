from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Create a FastAPI web app
app = FastAPI()


# Define the structure of the input data
class InputData(BaseModel):
    numbers: list[float]


@app.get("/")
def read_root():
    return {"message": "ML Service is running!"}


@app.post("/predict")
def predict(data: InputData):
    # Here you could load a trained model and use it.
    # For now, let's just return the sum of the numbers:
    total = float(np.sum(data.numbers))
    return {"sum": total}
