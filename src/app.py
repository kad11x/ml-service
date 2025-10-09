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
    return {"message": "ML Service is running! We are testinmg the update"}


@app.post("/predict")
def predict():

    return {"message": "ML Service is running! We are testinmg the update"}
