# api_openai.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
from contextlib import contextmanager
import base64, io, os, json
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# ────────────────────────────────────────────────────────────────────────────────
# Config & Clients
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not MONGODB_URI:
    raise RuntimeError("Missing MONGODB_URI")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ────────────────────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────────────────────
@contextmanager
def get_db_connection():
    """Gestionnaire de connexion MongoDB avec fermeture automatique"""
    mongo_client = None
    try:
        mongo_client = MongoClient(MONGODB_URI)
        yield mongo_client["GuessMyMeal"]["OpenFoodFrance"]
    finally:
        if mongo_client:
            mongo_client.close()

def safe_float(value, default=0.0):
    """Conversion sécurisée vers float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def image_to_data_url(img_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def ensure_rgb(img_bytes: bytes) -> tuple[bytes, tuple[int, int]]:
    """Valide l'image et force RGB en JPEG."""
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buff = io.BytesIO()
        im.save(buff, format="JPEG", quality=90)
        return buff.getvalue(), im.size  # (bytes, (W,H))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def find_nutrition_info(food_name: str):
    """
    Cherche un aliment proche du nom détecté dans la base OFF.
    On essaie product_name, puis generic_name.
    """
    if not food_name:
        return None
    
    try:
        with get_db_connection() as collection:
            pattern = food_name.replace("_", " ")
            query = {
                "$or": [
                    {"product_name": {"$regex": pattern, "$options": "i"}},
                    {"generic_name": {"$regex": pattern, "$options": "i"}},
                ]
            }
            result = collection.find_one(query)
            if result:
                return {
                    "product_name": result.get("product_name") or result.get("generic_name"),
                    "energy-kcal_100g": result.get("energy-kcal_100g"),
                    "proteins_100g": result.get("proteins_100g"),
                    "carbohydrates_100g": result.get("carbohydrates_100g"),
                    "fat_100g": result.get("fat_100g"),
                    "_id": str(result.get("_id")) if result.get("_id") else None,
                }
    except Exception as e:
        logger.error(f"Database query error: {str(e)}")
        
    return None

# ────────────────────────────────────────────────────────────────────────────────
# JSON Schema attendu (Structured Outputs)
# ────────────────────────────────────────────────────────────────────────────────
PLATE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "plate_assessment": {
            "type": "object",
            "properties": {
                "total_weight_g_source": {
                    "type": "string",
                    "enum": ["user_input", "model_estimate", "none"]
                },
                "total_weight_g": {"type": ["number", "null"]}
            },
            "required": ["total_weight_g_source", "total_weight_g"],
            "additionalProperties": False
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "synonyms": {"type": "array", "items": {"type": "string"}},
                    "portion": {
                        "type": "object",
                        "properties": {
                            "ratio": {"type": "number"},
                            "grams": {"type": ["number", "null"]}
                        },
                        "required": ["ratio", "grams"],
                        "additionalProperties": False
                    },
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"}
                },
                "required": ["label", "portion", "confidence"],
                "additionalProperties": False
            }
        },
        "uncertainty": {
            "type": "object",
            "properties": {
                "overall": {"type": "number"},
                "comments": {"type": "string"}
            },
            "required": ["overall"],
            "additionalProperties": False
        }
    },
    "required": ["plate_assessment", "items", "uncertainty"],
    "additionalProperties": False
}

# ────────────────────────────────────────────────────────────────────────────────
# Health
# ────────────────────────────────────────────────────────────────────────────────
@router.get("/health-openai")
async def health_openai():
    mongo_ok = False
    try:
        with get_db_connection() as collection:
            # Test simple de connexion
            collection.find_one({"_id": "test"}, {"_id": 1})
            mongo_ok = True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
    
    return {
        "status": "healthy" if mongo_ok else "degraded",
        "mongo_connected": mongo_ok,
        "openai_key_loaded": bool(OPENAI_API_KEY),
        "yolo_disabled": True
    }

# ────────────────────────────────────────────────────────────────────────────────
# Endpoint principal (LLM-only)
# ────────────────────────────────────────────────────────────────────────────────
@router.post("/predict-llm")
async def predict_llm(
    file: UploadFile = File(...),
    total_weight_g: float | None = Form(None)
):
    """
    Détection via OpenAI Vision (gpt-4o-mini) -> JSON structuré (items + ratios/grams).
    Puis lookup OFF (100g) + nutrition estimée si grams connus.
    """
    # Validation du fichier
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Limite de taille (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    try:
        # 1) Lecture & normalisation image
        raw_bytes = await file.read()
        img_bytes, (W, H) = ensure_rgb(raw_bytes)
        data_url = image_to_data_url(img_bytes, mime="image/jpeg")

        # 2) Prompts
        system_prompt = (
            "You are a food vision assistant. Detect distinct foods on the plate. "
            "Estimate portion ratios per item that sum roughly to 1.0. "
            "If a total weight is provided by the user, use it to compute grams per item; "
            "otherwise you may estimate grams but must set total_weight_g_source='model_estimate'. "
            "Return ONLY valid JSON following the provided schema. No extra text."
        )
        user_prompt = (
            "Return items with label, ratio (0..1), and grams if known. "
            "If total_weight_g is provided, grams must sum to that value."
        )

        # 3) Appel OpenAI avec debugging
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]}
                ],
                temperature=0.1
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

        llm_json = resp.choices[0].message.content
        logger.info(f"Raw LLM response: {llm_json}")  # Debug log
        
        if not llm_json or llm_json.strip() == "":
            raise HTTPException(status_code=500, detail="Empty response from LLM")
        
        # Nettoyer la réponse : supprimer les balises markdown
        cleaned_json = llm_json.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json[7:]  # Supprimer ```json
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json[3:]   # Supprimer ```
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-3]  # Supprimer ``` de fin
        cleaned_json = cleaned_json.strip()
        
        logger.info(f"Cleaned JSON: {cleaned_json}")
        
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error. Cleaned response: {repr(cleaned_json)}")
            raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}. Raw: {cleaned_json[:200]}")

        # 4) Harmoniser grams si l'utilisateur a fourni le poids total
        if total_weight_g is not None and parsed.get("items"):
            total_ratio = sum([max(0.0, safe_float(it["portion"]["ratio"])) for it in parsed["items"]])
            if total_ratio == 0:
                total_ratio = 1.0  # Éviter division par zéro
            
            for it in parsed["items"]:
                r = max(0.0, safe_float(it["portion"]["ratio"])) / total_ratio
                it["portion"]["ratio"] = r
                it["portion"]["grams"] = round(r * float(total_weight_g), 1)
            
            parsed["plate_assessment"]["total_weight_g_source"] = "user_input"
            parsed["plate_assessment"]["total_weight_g"] = float(total_weight_g)
        elif total_weight_g is None and parsed.get("plate_assessment"):
            if parsed["plate_assessment"]["total_weight_g"] is None:
                parsed["plate_assessment"]["total_weight_g_source"] = "none"

        # 5) Jointure OFF + calcul nutrition
        nutrition_per_item = []
        totals = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

        for it in parsed.get("items", []):
            label = it.get("label", "")
            grams = it.get("grams", None)
            ratio = it.get("ratio", None)
            confidence = it.get("confidence", None)

            nut = find_nutrition_info(label)
            est = None
            
            if nut and grams is not None:
                kcal100 = safe_float(nut.get("energy-kcal_100g"))
                p100 = safe_float(nut.get("proteins_100g"))
                c100 = safe_float(nut.get("carbohydrates_100g"))
                f100 = safe_float(nut.get("fat_100g"))

                # Calcul des nutriments estimés
                if any([kcal100, p100, c100, f100]):  # Au moins une valeur non nulle
                    factor = safe_float(grams) / 100.0
                    est = {
                        "kcal": round(kcal100 * factor, 1),
                        "protein_g": round(p100 * factor, 2),
                        "carbs_g": round(c100 * factor, 2),
                        "fat_g": round(f100 * factor, 2),
                    }
                    totals["kcal"] += est["kcal"]
                    totals["protein_g"] += est["protein_g"]
                    totals["carbs_g"] += est["carbs_g"]
                    totals["fat_g"] += est["fat_g"]

            nutrition_per_item.append({
                "label": label,
                "confidence": confidence,
                "portion": {"ratio": ratio, "grams": grams},
                "nutrition_100g": nut,
                "nutrition_estimated": est
            })

        # Arrondis finaux
        totals = {k: (round(v, 1) if k == "kcal" else round(v, 2)) for k, v in totals.items()}

        response = {
            "success": True,
            "image_size": {"width": W, "height": H},
            "assumptions": {
                "weight_source": parsed.get("plate_assessment", {}).get("total_weight_g_source"),
                "total_weight_g": parsed.get("plate_assessment", {}).get("total_weight_g"),
            },
            "items": nutrition_per_item,
            "nutrition_total": totals,
            "llm_uncertainty": parsed.get("uncertainty", {}),
            # optionnel pour debug (à désactiver en prod si besoin)
            "llm_raw": parsed
        }
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_llm: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

# ────────────────────────────────────────────────────────────────────────────────
# Création de l'app FastAPI
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Food Detection API", 
    version="1.0.0",
    description="API de détection d'aliments avec OpenAI Vision"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure le router
app.include_router(router, prefix="/api", tags=["food-detection"])

# Route racine optionnelle
@app.get("/")
async def root():
    return {"message": "Food Detection API is running", "docs": "/docs"}