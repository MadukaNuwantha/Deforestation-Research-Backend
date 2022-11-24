from fastapi import FastAPI, File
from controllers.water_body_detection import *
from controllers.forest_patch_detection import *
from controllers.wild_fire_detection import *
from fastapi import FastAPI

app = FastAPI()


@app.get('/')
async def root():
    return 'Hello World'


@app.post('/predict/waterbody')
async def predictWaterBody(file: bytes = File(...)):
    result = await predictWaterBodyImage(file)
    return result


@app.post('/predict/forestpatch')
async def predictForestPatch(file: bytes = File(...)):
    result = await predictForestPatchImage(file)
    return result


@app.post('/predict/wildfire')
async def predictWildFire(file: bytes = File(...)):
    result = await predictWildFireImage(file)
    return result
