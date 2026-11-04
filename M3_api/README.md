# M3 - API Grad-CAM (FastAPI)

## Lancer
```
pip install -r requirements.txt
uvicorn main:app --reload      # docs : http://localhost:8000/docs
pytest
```

## Endpoints
- `GET /health` -> `{"status": "ok", "model": "..."}`
- `POST /predict` (multipart, champ `file`, JPG/PNG/TIFF, max 15 Mo). Erreurs : 415 format refusé, 413 fichier trop gros.

## Format JSON de `/predict` (v1 - à valider par M2 et M4)
```json
{
  "class_id": 4,
  "class_code": "mel",
  "label": "Mélanome",
  "risk": "malin",
  "confidence": 0.81,
  "probabilities": {"akiec": 0.01, "bcc": 0.02, "bkl": 0.05, "df": 0.0, "mel": 0.81, "nv": 0.1, "vasc": 0.01},
  "uncertain": false,
  "heatmap": "<PNG en base64>",
  "inference_time_ms": 1840
}
```
- `risk` : `bénin` | `précancéreux` | `malin` (depuis classes.json)
- `uncertain` : `true` si `confidence` < 0,6
- `probabilities` : clés = codes de classes, dans l'ordre de classes.json
- `heatmap` : PNG encodé en base64 (sans préfixe `data:`). Factice en S1, Grad-CAM en S2.

Règle : ce format ne change plus après la semaine 1 sans accord de M3 et M4.
