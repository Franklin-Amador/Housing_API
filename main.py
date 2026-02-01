from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np
from typing import List, Optional
import os

# Importar el módulo de Vercel Blob
from blob_storage import get_blob_storage

# Inicializar FastAPI
app = FastAPI(
    title="California Housing Price Predictor",
    description="API para prediccion de precios de viviendas (ONNX optimizado)",
    version="2.0.0"
)

# Configuración de rutas
MODEL_BLOB_URL = "https://vjrbqsew9s3w1szr.public.blob.vercel-storage.com/model.onnx"
MODEL_LOCAL_PATH = "/tmp/model.onnx"

# Variable global para almacenar la sesión ONNX
ort_session = None

def load_model():
    """Carga el modelo ONNX desde Vercel Blob"""
    global ort_session
    
    try:
        print("Cargando modelo ONNX...")
        blob_storage = get_blob_storage()
        blob_content = blob_storage.download_file(MODEL_BLOB_URL)
        
        # Guardar temporalmente
        os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
        with open(MODEL_LOCAL_PATH, 'wb') as f:
            f.write(blob_content)
        
        # Crear sesión ONNX
        ort_session = ort.InferenceSession(MODEL_LOCAL_PATH)
        
        print("Modelo ONNX cargado correctamente")
        return ort_session
            
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None

# Cargar modelo al iniciar
ort_session = load_model()

# Mapeo de ocean_proximity a valores
OCEAN_PROXIMITY_MAP = {
    "INLAND": 0,
    "NEAR BAY": 1,
    "NEAR OCEAN": 2,
    "<1H OCEAN": 3,
    "ISLAND": 4
}

# Modelos de datos con Pydantic
class HousingFeatures(BaseModel):
    """Features para prediccion de precios de vivienda"""
    longitude: float = Field(..., description="Longitud geografica", ge=-124.35, le=-114.31)
    latitude: float = Field(..., description="Latitud geografica", ge=32.54, le=41.95)
    housing_median_age: float = Field(..., description="Edad mediana de las viviendas", ge=1, le=52)
    total_rooms: float = Field(..., description="Total de habitaciones", ge=1)
    total_bedrooms: float = Field(..., description="Total de dormitorios", ge=1)
    population: float = Field(..., description="Poblacion del area", ge=1)
    households: float = Field(..., description="Numero de hogares", ge=1)
    median_income: float = Field(..., description="Ingreso mediano (en $10,000s)", ge=0)
    ocean_proximity: Optional[str] = Field("INLAND", description="Proximidad al oceano")

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        }


class PredictionResponse(BaseModel):
    """Respuesta de prediccion"""
    predicted_price: float
    prediction_range: Optional[dict] = None
    message: str = "Prediccion exitosa"


class BatchPredictionRequest(BaseModel):
    """Request para predicciones batch"""
    instances: List[HousingFeatures]


class BatchPredictionResponse(BaseModel):
    """Respuesta para predicciones batch"""
    predictions: List[float]
    count: int


# Endpoints
@app.get("/")
async def root():
    """Endpoint raiz con informacion de la API"""
    return {
        "message": "API de Prediccion de Precios de Viviendas (ONNX)",
        "status": "active" if ort_session is not None else "model not loaded",
        "model_format": "ONNX",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica el estado de la API y el modelo"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "ONNX Runtime"
    }


@app.get("/model-info")
async def model_info():
    """Informacion sobre el modelo cargado"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    inputs = ort_session.get_inputs()
    outputs = ort_session.get_outputs()
    
    return {
        "model_type": "ONNX",
        "inputs": [{"name": inp.name, "type": str(inp.type), "shape": inp.shape} for inp in inputs],
        "outputs": [{"name": out.name, "type": str(out.type), "shape": out.shape} for out in outputs]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HousingFeatures):
    """Realiza una prediccion individual del precio de vivienda"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preparar inputs para ONNX (como arrays numpy individuales)
        input_data = {
            'longitude': np.array([[features.longitude]], dtype=np.float32),
            'latitude': np.array([[features.latitude]], dtype=np.float32),
            'housing_median_age': np.array([[features.housing_median_age]], dtype=np.float32),
            'total_rooms': np.array([[features.total_rooms]], dtype=np.float32),
            'total_bedrooms': np.array([[features.total_bedrooms]], dtype=np.float32),
            'population': np.array([[features.population]], dtype=np.float32),
            'households': np.array([[features.households]], dtype=np.float32),
            'median_income': np.array([[features.median_income]], dtype=np.float32),
            'ocean_proximity': np.array([[features.ocean_proximity or "INLAND"]], dtype=object)
        }
        
        # Hacer prediccion con ONNX
        outputs = ort_session.run(None, input_data)
        predicted_price = float(outputs[0][0][0])
        
        # Calcular rango estimado (+-10%)
        prediction_range = {
            "min": predicted_price * 0.9,
            "max": predicted_price * 1.1
        }
        
        return PredictionResponse(
            predicted_price=predicted_price,
            prediction_range=prediction_range,
            message="Prediccion exitosa (ONNX)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Realiza predicciones en batch (multiples instancias)"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        predictions = []
        
        for features in request.instances:
            # Preparar inputs para ONNX
            input_data = {
                'longitude': np.array([[features.longitude]], dtype=np.float32),
                'latitude': np.array([[features.latitude]], dtype=np.float32),
                'housing_median_age': np.array([[features.housing_median_age]], dtype=np.float32),
                'total_rooms': np.array([[features.total_rooms]], dtype=np.float32),
                'total_bedrooms': np.array([[features.total_bedrooms]], dtype=np.float32),
                'population': np.array([[features.population]], dtype=np.float32),
                'households': np.array([[features.households]], dtype=np.float32),
                'median_income': np.array([[features.median_income]], dtype=np.float32),
                'ocean_proximity': np.array([[features.ocean_proximity or "INLAND"]], dtype=object)
            }
            
            # Hacer prediccion
            outputs = ort_session.run(None, input_data)
            predictions.append(float(outputs[0][0][0]))
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion batch: {str(e)}")
