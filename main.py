from fastapi import FastAPI, File
from controllers.water_body_detection import *
from fastapi import FastAPI
from starlette.responses import Response
import io
from PIL import Image

app = FastAPI()


@app.get('/')
async def root():
    return 'Hello World'

@app.post('/predict/waterbody')
async def predictWaterBody(file: bytes = File(...)):
    result = await predictImage(file)
    return result
