from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import base64
import io
import os
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Hugging Face
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import torch

# ==========================
# FastAPI app (definido antes dos decorators!)
# ==========================
app = FastAPI(title="Enhanced Food Detection API", version="2.2.1")

# ==========================
# MongoDB
# ==========================
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI) if MONGODB_URI else None
db = client["GuessMyMeal"] if client is not None else None
collection = db["OpenFoodFrance"] if db is not None else None
ingredients_collection = db["Ingredients"] if db is not None else None


# ==========================
# Globals / Model paths
# ==========================
model = None
segmentation_model = None
hf_processor = None
hf_seg_model = None

MODEL_DETECT_PATH = "models/food_detection_model.pt"   # seu modelo de pratos
YOLO_SEG_WEIGHTS = "yolov8s-seg.pt"                     # COCO (80 classes)
HF_SEG_REPO = "mccaly/test2"                             # repo do Hugging Face (UPerNet)

# ==========================
# Utils
# ==========================

def _pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _annotate_ultralytics(result) -> str:
    """Converte o .plot() do Ultralytics em base64."""
    pil = Image.fromarray(result.plot())
    return _pil_to_base64(pil)


def _colorize_segmap(segmap_np: np.ndarray, id2label: Dict[int, str]):
    """Gera um mapa colorido e porcentagem de áreas por classe."""
    h, w = segmap_np.shape

    def _color(i: int):
        return (
            (37 * i) % 255,
            (97 * i) % 255,
            (173 * i) % 255,
        )

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    areas = {}
    total = h * w

    unique_ids, counts = np.unique(segmap_np, return_counts=True)
    for cid, cnt in zip(unique_ids.tolist(), counts.tolist()):
        rgb[segmap_np == cid] = _color(cid)
        label = id2label.get(int(cid), f"class_{cid}")
        areas[label] = round(100.0 * cnt / total, 2)

    return Image.fromarray(rgb), areas


# ==========================
# DB helpers
# ==========================

def find_nutrition_info(food_name: str) -> Optional[Dict]:
    if collection is None:
        return None
    query = {"product_name": {"$regex": food_name.replace("_", " "), "$options": "i"}}
    result = collection.find_one(query)
    if result:
        return {
            "product_name": result.get("product_name"),
            "energy-kcal": result.get("energy-kcal_100g"),
            "proteins": result.get("proteins_100g"),
            "carbohydrates": result.get("carbohydrates_100g"),
            "fat": result.get("fat_100g"),
        }
    return None


# ==========================
# Startup: load all models
# ==========================
@app.on_event("startup")
async def load_models():
    global model, segmentation_model, hf_processor, hf_seg_model
    try:
        # Detector principal (pratos)
        model = YOLO(MODEL_DETECT_PATH)
        print("✅ Main YOLO model loaded:", MODEL_DETECT_PATH)

        # Segmentação por instâncias (COCO)
        segmentation_model = YOLO(YOLO_SEG_WEIGHTS)
        print("✅ YOLOv8 segmentation loaded:", YOLO_SEG_WEIGHTS)

        # Segmentação semântica (Hugging Face UPerNet)
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")  # opcional para privados
        hf_processor = AutoImageProcessor.from_pretrained(HF_SEG_REPO, token=token)
        hf_seg_model = UperNetForSemanticSegmentation.from_pretrained(HF_SEG_REPO, token=token)
        if torch.cuda.is_available():
            hf_seg_model.to("cuda")
        print(f"✅ HF semantic seg loaded: {HF_SEG_REPO} on", "cuda" if torch.cuda.is_available() else "cpu")

    except Exception as e:
        print(f"❌ Error loading models: {e}")


# ==========================
# Routes
# ==========================
@app.get("/")
async def root():
    return {"message": "Enhanced Food Detection API is running."}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "yolo_detect_loaded": model is not None,
        "yolo_seg_loaded": segmentation_model is not None,
        "hf_seg_loaded": hf_seg_model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detecção principal (seu modelo de pratos) + segmentação YOLO COCO (independente).
    """
    if model is None or segmentation_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ---------- 1) Detecção principal (pratos) ----------
        det_results = model.predict(image)
        det_res = det_results[0]
        det_annot_b64 = _annotate_ultralytics(det_res)

        main_detections = []
        if det_res.boxes is not None:
            for box in det_res.boxes:
                cls_name = model.names[int(box.cls.item())]
                conf = float(box.conf.item())
                bbox = [float(c) for c in box.xyxy[0].tolist()]
                nutrition = find_nutrition_info(cls_name)
                main_detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": bbox,
                    "nutrition": nutrition,
                })

        # ---------- 2) Segmentação YOLO (instância, independente) ----------
        seg_detections = []
        yolo_seg_annot_b64 = None
        try:
            seg_results = segmentation_model.predict(image)
            seg_res = seg_results[0]
            if seg_res.boxes is not None and seg_res.masks is not None:
                classes = seg_res.boxes.cls.cpu().numpy()
                confs = seg_res.boxes.conf.cpu().numpy()
                boxes = seg_res.boxes.xyxy.cpu().numpy()
                yolo_seg_annot_b64 = _annotate_ultralytics(seg_res)
                for i in range(len(classes)):
                    cid = int(classes[i])
                    cls_name = segmentation_model.names[cid]
                    conf = float(confs[i])
                    bbox = [float(v) for v in boxes[i].tolist()]
                    seg_detections.append({
                        "class_name": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": bbox,
                    })
        except Exception as e:
            print("Warn: YOLO segmentation failed:", e)

        return JSONResponse(content={
            "success": True,
            "main_dish": {
                "detections": main_detections,
                "annotated_image": det_annot_b64,
                "total_detections": len(main_detections),
            },
            "yolo_segmentation": {
                "segments": seg_detections,
                "annotated_image": yolo_seg_annot_b64,
                "total_segments": len(seg_detections),
            },
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/predict/hf-seg")
async def predict_hf_seg(file: UploadFile = File(...)):
    """Segmentação **semântica** com o modelo do Hugging Face (UPerNet)."""
    if hf_processor is None or hf_seg_model is None:
        raise HTTPException(status_code=500, detail="HF segmentation model not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img = np.array(image)

        # Pré-processamento
        inputs = hf_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Inferência
        with torch.no_grad():
            outputs = hf_seg_model(**inputs)  # logits: [B, C, h', w']
        logits = outputs.logits

        # Upsample para tamanho original
        logits = torch.nn.functional.interpolate(
            logits, size=(np_img.shape[0], np_img.shape[1]), mode="bilinear", align_corners=False
        )
        seg = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

        # Colorização + áreas
        id2label = {int(k): v for k, v in hf_seg_model.config.id2label.items()}
        segmap_pil, areas_pct = _colorize_segmap(seg, id2label)

        # Overlay
        overlay = Image.fromarray(np_img).convert("RGBA")
        seg_rgba = segmap_pil.convert("RGBA")
        seg_rgba.putalpha(100)  # transparência
        composite = Image.alpha_composite(overlay, seg_rgba).convert("RGB")

        return JSONResponse(content={
            "success": True,
            "model": HF_SEG_REPO,
            "classes": list(id2label.values()),
            "areas_percent": areas_pct,
            "top_classes": sorted(areas_pct.items(), key=lambda x: x[1], reverse=True)[:5],
            "segmentation_map": _pil_to_base64(segmap_pil),
            "overlay_on_image": _pil_to_base64(composite),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF segmentation error: {str(e)}")


# ==========================
# Entrypoint local
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
