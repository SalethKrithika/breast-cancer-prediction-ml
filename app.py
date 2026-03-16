import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
models = pickle.load(open("breast_cancer_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# Best algorithm name
with open("best_algorithm.txt") as f:
    best_algorithm = f.read().strip()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    radius_mean: float = Form(...),
    texture_mean: float = Form(...),
    perimeter_mean: float = Form(...),
    area_mean: float = Form(...),
    smoothness_mean: float = Form(...),
    compactness_mean: float = Form(...),
    concavity_mean: float = Form(...),
    concave_points_mean: float = Form(...),
    symmetry_mean: float = Form(...),
    fractal_dimension_mean: float = Form(...)
):

    # User input
    data = np.array([[

        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean

    ]])

    data = scaler.transform(data)

    predictions = {}

    # Run all models
    for name, model in models.items():

        pred = model.predict(data)[0]

        if pred == 1:
            result = "Malignant"
        else:
            result = "Benign"

        predictions[name] = result


    # Get best prediction
    best_prediction = predictions[best_algorithm]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predictions": predictions,
        "best_algorithm": best_algorithm,
        "best_prediction": best_prediction
    })