from fastapi import FastAPI
from app.schema import Cardio
from app.model import load_model_scaler
import pandas as pd

#for html
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Templates folder
templates = Jinja2Templates(directory="templates")



# object of FastAPI

app = FastAPI()
model , scaler = load_model_scaler()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# get method 
# @app.get("/")
# def home():
#     return "Welcome to FastAPI Application"

# post method for prediction

@app.post("/cardio-predict")
def predict_cardio(data:Cardio):
    input_data = pd.DataFrame([
        # JSON format used for API
        data.model_dump()
    ])
    input_scaler = scaler.transform(input_data)
    prediction = model.predict(input_scaler)[0]
    return{
        "Prediction_Status":int(prediction),
        "Status": "Likely to be Healthy" if prediction==0 else "Likely to be unhealthy"
    }