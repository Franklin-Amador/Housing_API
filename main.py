from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import dill
import pandas as pd
from typing import List, Optional
import os

# Importar el módulo de Vercel Blob
from blob_storage import get_blob_storage

# Inicializar FastAPI
app = FastAPI(
    title="California Housing Price Predictor",
    description="API para prediccion de precios de viviendas usando el modelo del capitulo 2",
    version="1.0.0"
)

# Configuración de rutas
MODEL_BLOB_URL = "https://vjrbqsew9s3w1szr.public.blob.vercel-storage.com/model_sklearn_1_7_2.pkl"
MODEL_LOCAL_PATH = "/tmp/model_sklearn_1_7_2.pkl"

# Variable global para almacenar el modelo cargado
model = None

def load_model():
    """Carga el modelo desde Vercel Blob"""
    global model
    
    try:
        print("Cargando modelo...")
        blob_storage = get_blob_storage()
        blob_content = blob_storage.download_file(MODEL_BLOB_URL)
        
        # Guardar temporalmente para desserializar
        os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
        with open(MODEL_LOCAL_PATH, 'wb') as f:
            f.write(blob_content)
        
        with open(MODEL_LOCAL_PATH, 'rb') as f:
            model = dill.load(f)
        
        print("Modelo cargado correctamente")
        return model
            
    except Exception as e:
        print(f"Error al cargar modelo: {e}")
        return None

# Cargar modelo al iniciar
model = load_model()


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
        "message": "API de Prediccion de Precios de Viviendas",
        "status": "active" if model is not None else "model not loaded",
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
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": str(type(model).__name__)
    }


@app.get("/model-info")
async def model_info():
    """Informacion sobre el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    info = {
        "model_type": str(type(model).__name__),
        "model_class": str(type(model))
    }
    
    if hasattr(model, 'get_params'):
        info["parameters"] = model.get_params()
    
    if hasattr(model, 'feature_names_in_'):
        info["features"] = list(model.feature_names_in_)
    
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HousingFeatures):
    """Realiza una prediccion individual del precio de vivienda"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preparar datos como DataFrame con nombres de columnas
        data = {
            'longitude': [features.longitude],
            'latitude': [features.latitude],
            'housing_median_age': [features.housing_median_age],
            'total_rooms': [features.total_rooms],
            'total_bedrooms': [features.total_bedrooms],
            'population': [features.population],
            'households': [features.households],
            'median_income': [features.median_income],
            'ocean_proximity': [features.ocean_proximity if features.ocean_proximity else "INLAND"]
        }
        
        X = pd.DataFrame(data)
        
        # Hacer prediccion
        prediction = model.predict(X)
        predicted_price = float(prediction[0])
        
        # Calcular rango estimado (+-10%)
        prediction_range = {
            "min": predicted_price * 0.9,
            "max": predicted_price * 1.1
        }
        
        return PredictionResponse(
            predicted_price=predicted_price,
            prediction_range=prediction_range,
            message="Prediccion exitosa"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Realiza predicciones en batch (multiples instancias)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preparar datos como DataFrame con nombres de columnas
        data = {
            'longitude': [],
            'latitude': [],
            'housing_median_age': [],
            'total_rooms': [],
            'total_bedrooms': [],
            'population': [],
            'households': [],
            'median_income': [],
            'ocean_proximity': []
        }
        
        for features in request.instances:
            data['longitude'].append(features.longitude)
            data['latitude'].append(features.latitude)
            data['housing_median_age'].append(features.housing_median_age)
            data['total_rooms'].append(features.total_rooms)
            data['total_bedrooms'].append(features.total_bedrooms)
            data['population'].append(features.population)
            data['households'].append(features.households)
            data['median_income'].append(features.median_income)
            data['ocean_proximity'].append(features.ocean_proximity if features.ocean_proximity else "INLAND")
        
        X = pd.DataFrame(data)
        
        # Hacer predicciones
        predictions = model.predict(X)
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion batch: {str(e)}")
