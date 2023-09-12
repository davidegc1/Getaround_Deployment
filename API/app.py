# imports
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")

description = """
API used to predict the optimal price of a car rental, depending on its characteristics.
The goal is to help customers, in this case car owners, choose the best rental price based on historical data. 

## Endpoints 
* _predict_ - predict the optimal price of one car
"""

app = FastAPI(
    title="Getaround: Car Rental Prediction",
    description=description
)

class PredictionFeatures(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: int
    has_gps: int
    has_air_conditioning: int
    automatic_car: int
    has_getaround_connect: int
    has_speed_regulator: int
    winter_tires: int

# "welcome" endpoint
@app.get("/")
async def welcome():
    message = "Hello there! We are glad you are using our API to get the most of Getaround! In order to make a prediction you need to go to our API, which you can access by adding '/docs' to the end of the link. In there you will find instructions, as well as the steps and requirements to make predictions."
    return message

# create "predict" endpoint
@app.post("/predict")
def predict(PredictionFeatures: PredictionFeatures):
    """
    Input Options:\n
    model_key: [Audi, BMW, Citroën, Peugeot, Renault, other]\n
    mileage: *number*\n
    engine_power: *number*\n
    fuel: [diesel, other]\n
    paint_color: [black, blue, grey, white, other]\n
    car_type: [estate, hatchback, sedan, suv, other]\n
    private_parking_available: 0 (no), 1 (yes)\n
    has_gps: 0 (no), 1 (yes)\n
    has_air_conditioning: 0 (no), 1 (yes)\n
    automatic_car: 0 (no), 1 (yes)\n
    has_getaround_connect: 0 (no), 1 (yes)\n
    has_speed_regulator: 0 (no), 1 (yes)\n
    winter_tires: 0 (no), 1 (yes)
    """
    
    # preprocess on pricing dataset, in order to obtain 
    X = pd.read_csv("Data/preprocessed_X.csv")
    X.drop(columns="Unnamed: 0", axis=1, inplace=True)

    # change objects to categories
    X["model_key"] = X["model_key"].astype("category")
    X["fuel"] = X["fuel"].astype("category")
    X["paint_color"] = X["paint_color"].astype("category")
    X["car_type"] = X["car_type"].astype("category")

    # preprocessing
    cat_features = []
    num_features = []

    for col_name, col_type in X.dtypes.items():
        if ((col_type=="category")):
            cat_features.append(col_name)
        elif col_type==np.int64:
            num_features.append(col_name)

    num_transformer = Pipeline(steps=[
        ("standardization", StandardScaler())
    ])

    # bools are created from one hot encoding, so it is important to replace those values with binary code
    cat_transformer = Pipeline(steps=[
        ("one hot encoding", OneHotEncoder(drop="first"))
    ])

    # create preprocessor
    # parameters: name, transformer, columns to be applied on
    preprocessor = ColumnTransformer(transformers=[
        ("numerical", num_transformer, num_features),
        ("categorical", cat_transformer, cat_features)
    ]) 

    # fit preprocessor
    X_preprocessed = preprocessor.fit_transform(X)

   # read features into df
    df = pd.DataFrame(dict(PredictionFeatures), index=[0])
    df = preprocessor.transform(df)

    # load model
    model = joblib.load("final_model_api")

    # make prediction
    prediction = model.predict(df)

    # return prediction in a list
    returned_pred = {"optimal price": prediction.tolist()[0]}
    return returned_pred