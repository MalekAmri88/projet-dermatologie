import base64
import io
import json

from fastapi import FastAPI, UploadFile
from PIL import Image, ImageOps

app = FastAPI()


def fausse_heatmap():
    degrade = Image.radial_gradient("L")
    image = ImageOps.colorize(degrade, black="red", white="blue")
    tampon = io.BytesIO()
    image.save(tampon, format="PNG")
    return base64.b64encode(tampon.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    with open("exemple_reponse.json", encoding="utf-8") as f:
        donnees = json.load(f)
    donnees["heatmap_base64"] = fausse_heatmap()
    return donnees