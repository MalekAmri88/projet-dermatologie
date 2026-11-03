import json
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    with open("exemple_reponse.json", encoding="utf-8") as f:
        return json.load(f)