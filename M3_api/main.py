"""API M3 - Projet dermatologie. Semaine 1 : réponse factice + validation d'image.

Le format JSON de /predict est décrit dans README.md.
L'ordre des classes vient UNIQUEMENT de classes.json.
"""
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

BASE = Path(__file__).parent
UNCERTAINTY_THRESHOLD = 0.6  # proba max < 0,6 => incertain
ALLOWED_FORMATS = {"JPEG", "PNG", "TIFF"}
MAX_BYTES = 15 * 1024 * 1024

_cfg = json.loads((BASE / "classes.json").read_text(encoding="utf-8"))
# Ordre officiel = label_index de classes.json (akiec=0 ... vasc=6)
CLASSES = [
    {"id": _cfg["label_index"][code], "code": code, "label": _cfg["noms"][code], "risk": _cfg["risque"][code]}
    for code in sorted(_cfg["codes"], key=lambda c: _cfg["label_index"][c])
]
assert [c["id"] for c in CLASSES] == list(range(7)), "classes.json invalide"

app = FastAPI(title="API Dermatologie - M3", version="0.1.0")


def load_image(data: bytes) -> Image.Image:
    """Vérifie le format (JPG/PNG/TIFF) et supprime les métadonnées EXIF."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Fichier illisible ou format non supporté (JPG, PNG, TIFF).")
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, f"Format {fmt} refusé. Formats acceptés : JPG, PNG, TIFF.")
    img = img.convert("RGB")
    # Recopie des pixels dans une image neuve : aucune métadonnée (EXIF) n'est conservée.
    return Image.fromarray(np.asarray(img))


def predict_dummy(img: Image.Image) -> list[float]:
    """Réponse factice, déterministe selon l'image. À remplacer par le modèle de M2."""
    seed = int(np.asarray(img.resize((8, 8))).sum()) % (2**32)
    p = np.random.default_rng(seed).dirichlet(np.ones(len(CLASSES)) * 0.5)
    return p.tolist()


def dummy_heatmap(img: Image.Image) -> str:
    """Heatmap factice (PNG base64). Semaine 2 : remplacée par Grad-CAM."""
    buf = io.BytesIO()
    Image.new("RGB", img.size, (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_response(probs: list[float], heatmap_b64: str, elapsed_ms: int) -> dict:
    k = max(range(len(probs)), key=probs.__getitem__)
    c = CLASSES[k]
    return {
        "class_id": c["id"],
        "class_code": c["code"],
        "label": c["label"],
        "risk": c["risk"],
        "confidence": round(probs[k], 4),
        "probabilities": {cl["code"]: round(p, 4) for cl, p in zip(CLASSES, probs)},
        "uncertain": probs[k] < UNCERTAINTY_THRESHOLD,
        "heatmap": heatmap_b64,
        "inference_time_ms": elapsed_ms,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "dummy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Fichier trop volumineux (max 15 Mo).")
    t0 = time.perf_counter()
    img = load_image(data)
    probs = predict_dummy(img)
    heat = dummy_heatmap(img)
    return build_response(probs, heat, int((time.perf_counter() - t0) * 1000))
