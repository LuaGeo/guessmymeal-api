# api_openai.py - Version complète avec fibres, sucre, sodium
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ────────────────────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────────────────────
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

def calculate_nutrition_totals(items):
    """Calcule les totaux nutritionnels à partir des items - AVEC fibres, sucre, sodium"""
    totals = {
        "calories": 0.0,
        "proteins": 0.0,
        "carbohydrates": 0.0,
        "fat": 0.0,
        "fiber": 0.0,        # ← AJOUTÉ
        "sugar": 0.0,        # ← AJOUTÉ
        "sodium": 0.0        # ← AJOUTÉ
    }
    
    for item in items:
        nutrition = item.get("nutrition_estimated", {})
        for key in totals.keys():
            totals[key] += nutrition.get(key, 0)
    
    # Arrondir les résultats
    return {k: round(v, 1 if k == "calories" else 2) for k, v in totals.items()}

# ────────────────────────────────────────────────────────────────────────────────
# JSON Schema avec nutrition COMPLÈTE (macros + fibres + sucre + sodium)
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
                    "nutrition_per_100g": {
                        "type": "object",
                        "properties": {
                            "calories": {"type": "number"},
                            "proteins": {"type": "number"},
                            "carbohydrates": {"type": "number"},
                            "fat": {"type": "number"},
                            "fiber": {"type": "number"},      # ← AJOUTÉ
                            "sugar": {"type": "number"},      # ← AJOUTÉ
                            "sodium": {"type": "number"}      # ← AJOUTÉ
                        },
                        "required": ["calories", "proteins", "carbohydrates", "fat", "fiber", "sugar", "sodium"],
                        "additionalProperties": False
                    },
                    "nutrition_estimated": {
                        "type": "object",
                        "properties": {
                            "calories": {"type": "number"},
                            "proteins": {"type": "number"},
                            "carbohydrates": {"type": "number"},
                            "fat": {"type": "number"},
                            "fiber": {"type": "number"},      # ← AJOUTÉ
                            "sugar": {"type": "number"},      # ← AJOUTÉ
                            "sodium": {"type": "number"}      # ← AJOUTÉ
                        },
                        "required": ["calories", "proteins", "carbohydrates", "fat", "fiber", "sugar", "sodium"],
                        "additionalProperties": False
                    },
                    "health_score": {"type": "number"}
                },
                "required": ["label", "portion", "confidence", "nutrition_per_100g", "nutrition_estimated", "health_score"],
                "additionalProperties": False
            }
        },
        "nutrition_total": {
            "type": "object",
            "properties": {
                "calories": {"type": "number"},
                "proteins": {"type": "number"},
                "carbohydrates": {"type": "number"},
                "fat": {"type": "number"},
                "fiber": {"type": "number"},      # ← AJOUTÉ
                "sugar": {"type": "number"},      # ← AJOUTÉ
                "sodium": {"type": "number"}      # ← AJOUTÉ
            },
            "required": ["calories", "proteins", "carbohydrates", "fat", "fiber", "sugar", "sodium"],
            "additionalProperties": False
        },
        "uncertainty": {
            "type": "object",
            "properties": {
                "overall": {"type": "number"},
                "comments": {"type": "string"}
            },
            "required": ["overall", "comments"],
            "additionalProperties": False
        }
    },
    "required": ["plate_assessment", "items", "nutrition_total", "uncertainty"],
    "additionalProperties": False
}

# ────────────────────────────────────────────────────────────────────────────────
# Health
# ────────────────────────────────────────────────────────────────────────────────
@router.get("/health-openai")
async def health_openai():
    return {
        "status": "healthy",
        "openai_key_loaded": bool(OPENAI_API_KEY),
        "database": "openai_only"
    }

# ────────────────────────────────────────────────────────────────────────────────
# Endpoint principal avec NUTRITION COMPLÈTE
# ────────────────────────────────────────────────────────────────────────────────
@router.post("/predict-llm")
async def predict_llm(
    file: UploadFile = File(...),
    total_weight_g: float | None = Form(None)
):
    """
    Analyse nutritionnelle COMPLÈTE via OpenAI Vision uniquement.
    Inclut : calories, protéines, glucides, lipides, FIBRES, SUCRE, SODIUM + score santé
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

        # 2) Prompts pour nutrition COMPLÈTE
        system_prompt = (
            "You are a professional nutritionist AI. Analyze the food image and provide:\n"
            "1. Detection of all foods with portion ratios\n"
            "2. COMPLETE nutritional data per 100g for each food:\n"
            "   - calories (kcal), proteins (g), carbohydrates (g), fat (g)\n"
            "   - fiber (g), sugar (g), sodium (mg)\n"
            "3. Calculated nutrition for actual portions\n"
            "4. Health score (0-100) per food based on nutritional quality\n"
            "5. Total nutrition summary\n\n"
            "IMPORTANT: Always provide ALL 7 nutritional values. Use 0 if the food naturally contains none.\n"
            "Return ONLY valid JSON following the exact schema. No extra text."
        )
        
        weight_instruction = f" Total weight: {total_weight_g}g" if total_weight_g else " Estimate weight"
        user_prompt = (
            f"Analyze foods and provide COMPLETE nutrition data including:\n"
            f"- Macronutrients: calories, proteins, carbohydrates, fat\n"
            f"- Additional: fiber, sugar, sodium\n"
            f"- Health score for each food{weight_instruction}"
        )

        # 3) Appel OpenAI avec Structured Outputs
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
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "nutrition_analysis_schema",
                        "schema": PLATE_JSON_SCHEMA,
                        "strict": True
                    }
                },
                temperature=0.1
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

        llm_json = resp.choices[0].message.content
        logger.info(f"Raw LLM response: {llm_json}")
        
        if not llm_json or llm_json.strip() == "":
            raise HTTPException(status_code=500, detail="Empty response from LLM")
        
        # Nettoyage de la réponse
        cleaned_json = llm_json.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json[7:]
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json[3:]
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-3]
        cleaned_json = cleaned_json.strip()
        
        logger.info(f"Cleaned JSON: {cleaned_json}")
        
        try:
            parsed = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error. Cleaned response: {repr(cleaned_json)}")
            raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}. Raw: {cleaned_json[:200]}")

        # 4) Ajustement si poids utilisateur fourni
        if total_weight_g is not None and parsed.get("items"):
            total_ratio = sum([max(0.0, safe_float(it["portion"]["ratio"])) for it in parsed["items"]])
            if total_ratio == 0:
                total_ratio = 1.0
            
            for item in parsed["items"]:
                # Ajuster ratio et grammes
                normalized_ratio = max(0.0, safe_float(item["portion"]["ratio"])) / total_ratio
                item["portion"]["ratio"] = normalized_ratio
                item["portion"]["grams"] = round(normalized_ratio * float(total_weight_g), 1)
                
                # Recalculer nutrition estimée pour nouvelle portion - AVEC TOUS LES NUTRIMENTS
                grams = item["portion"]["grams"]
                nutrition_100g = item["nutrition_per_100g"]
                factor = grams / 100.0
                
                item["nutrition_estimated"] = {
                    "calories": round(nutrition_100g["calories"] * factor, 1),
                    "proteins": round(nutrition_100g["proteins"] * factor, 2),
                    "carbohydrates": round(nutrition_100g["carbohydrates"] * factor, 2),
                    "fat": round(nutrition_100g["fat"] * factor, 2),
                    "fiber": round(nutrition_100g["fiber"] * factor, 2),      # ← AJOUTÉ
                    "sugar": round(nutrition_100g["sugar"] * factor, 2),      # ← AJOUTÉ
                    "sodium": round(nutrition_100g["sodium"] * factor, 2)     # ← AJOUTÉ
                }
            
            # Recalculer totaux avec TOUS les nutriments
            parsed["nutrition_total"] = calculate_nutrition_totals(parsed["items"])
            
            # Mettre à jour source du poids
            parsed["plate_assessment"]["total_weight_g_source"] = "user_input"
            parsed["plate_assessment"]["total_weight_g"] = float(total_weight_g)

        # 5) Préparer la réponse finale
        response = {
            "success": True,
            "image_size": {"width": W, "height": H},
            "assumptions": {
                "weight_source": parsed.get("plate_assessment", {}).get("total_weight_g_source"),
                "total_weight_g": parsed.get("plate_assessment", {}).get("total_weight_g"),
            },
            "items": parsed.get("items", []),
            "nutrition_total": parsed.get("nutrition_total", {}),
            "llm_uncertainty": parsed.get("uncertainty", {}),
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
    version="2.1.0",
    description="API de détection d'aliments avec analyse nutritionnelle COMPLÈTE via OpenAI"
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

# Route racine
@app.get("/")
async def root():
    return {"message": "Food Detection API v2.1 with COMPLETE Nutrition (fiber, sugar, sodium)", "docs": "/docs"}