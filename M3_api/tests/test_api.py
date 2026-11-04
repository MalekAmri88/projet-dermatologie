import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import app  # noqa: E402

client = TestClient(app)


def img_bytes(fmt="PNG", exif=False):
    im = Image.new("RGB", (64, 64), (120, 80, 60))
    buf = io.BytesIO()
    if exif:
        e = Image.Exif(); e[0x010F] = "Camera"
        im.save(buf, format=fmt, exif=e)
    else:
        im.save(buf, format=fmt)
    return buf.getvalue()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_valid():
    r = client.post("/predict", files={"file": ("a.png", img_bytes("PNG"), "image/png")})
    assert r.status_code == 200
    j = r.json()
    for k in ["class_id", "class_code", "label", "risk", "probabilities", "uncertain", "heatmap", "inference_time_ms"]:
        assert k in j
    assert len(j["probabilities"]) == 7
    assert abs(sum(j["probabilities"].values()) - 1) < 0.01


def test_formats_accepted():
    for fmt in ["JPEG", "TIFF"]:
        r = client.post("/predict", files={"file": ("a", img_bytes(fmt), "image/x")})
        assert r.status_code == 200


def test_format_refused():
    r = client.post("/predict", files={"file": ("a.bmp", img_bytes("BMP"), "image/bmp")})
    assert r.status_code == 415
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_exif_removed():
    from main import load_image
    clean = load_image(img_bytes("JPEG", exif=True))
    assert len(clean.getexif()) == 0
