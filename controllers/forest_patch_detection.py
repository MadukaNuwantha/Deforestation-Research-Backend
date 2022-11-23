import torch
import io
import base64
from PIL import Image
from io import BytesIO

MODEL = torch.hub.load('./yolov5', 'custom',path='./models/forestpatchmodel.pt', source='local')

async def predictForestPatchImage(imageFile):
    processedImage = processImage(imageFile)
    predictedImage = predict(processedImage)
    convertedImage = encodeImage(predictedImage)
    return convertedImage

def processImage(binary_image, max_size=255):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize((
        int(input_image.width * resize_factor),
        int(input_image.height * resize_factor)
    ))
    return resized_image

def predict(processedImage):
    results = MODEL(processedImage)
    renderedImage = results.render()
    convertedImage = Image.fromarray(renderedImage[0]).convert("RGB")
    convertedImage.show()
    return convertedImage

def encodeImage(convertedImage):
    buffered = BytesIO()
    convertedImage.save(buffered, format="PNG")
    buffered.seek(0)
    imageByte = buffered.getvalue()
    encodedImage = "data:image/png;base64," + base64.b64encode(imageByte).decode()
    return encodedImage

# def predict(processedImage):
#     results = MODEL(processedImage)
#     results.print()
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")
#     detect_res = json.loads(detect_res)
#     return detect_res