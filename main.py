from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import dill
import pandas as pd
from typing import List, Optional
import uvicorn
import os

# Imports necesarios para que dill deserialice correctamente
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Importar el módulo de Vercel Blob
from blob_storage import get_blob_storage

# Inicializar FastAPI
app = FastAPI(
    title="California Housing Price Predictor",
    description="API para predicción de precios de viviendas usando el modelo del capítulo 2",
    version="1.0.0"
)

# Configuración de rutas
MODEL_BLOB_URL = "https://vjrbqsew9s3w1szr.public.blob.vercel-storage.com/model_sklearn_1_7_2.pkl"
MODEL_LOCAL_PATH = "/tmp/model_sklearn_1_7_2.pkl"  # Ruta temporal local

# Variable global para almacenar información del modelo
model_sklearn_version = None

# Asegurar que las funciones necesarias estén en el namespace global antes de cargar
import sklearn.utils.validation
globals()['check_is_fitted'] = sklearn.utils.validation.check_is_fitted
globals()['check_array'] = sklearn.utils.validation.check_array

def load_model():
    """Carga el modelo desde Vercel Blob o local"""
    global model, model_sklearn_version
    
    try:
        # Intentar cargar desde Vercel Blob primero
        print("📦 Intentando cargar modelo desde Vercel Blob...")
        try:
            blob_storage = get_blob_storage()
            blob_content = blob_storage.download_file(MODEL_BLOB_URL)
            
            # Guardar temporalmente para desserializar
            os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
            with open(MODEL_LOCAL_PATH, 'wb') as f:
                f.write(blob_content)
            
            with open(MODEL_LOCAL_PATH, 'rb') as f:
                model = dill.load(f)
            print(f"✅ Modelo cargado exitosamente desde Vercel Blob")
        except Exception as blob_error:
            # Fallback: cargar desde archivo local
            print(f"⚠️  Error al cargar desde Blob: {blob_error}")
            print("📦 Intentando cargar modelo desde archivo local...")
            with open("model_sklearn_1_7_2.pkl", "rb") as f:
                model = dill.load(f)
            print(f"✅ Modelo cargado exitosamente desde archivo local")
        
        # Intentar obtener la versión de sklearn con la que se entrenó
        try:
            import sklearn
            current_version = sklearn.__version__
            print(f"📦 Versión actual de scikit-learn: {current_version}")
            
            # Intentar obtener metadata del modelo
            if hasattr(model, '_sklearn_version'):
                model_sklearn_version = model._sklearn_version
                print(f"📦 Modelo entrenado con scikit-learn: {model_sklearn_version}")
                if model_sklearn_version != current_version:
                    print(f"⚠️  ADVERTENCIA: Incompatibilidad de versiones detectada")
                    print(f"   Modelo: {model_sklearn_version} | Actual: {current_version}")
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        print(f"💡 Tip: Verifica que el archivo exista en Vercel Blob o localmente")
        import traceback
        traceback.print_exc()
        return None
    
    return model

# Cargar modelo al iniciar
model = load_model()


# Modelos de datos con Pydantic
class HousingFeatures(BaseModel):
    """Features para predicción de precios de vivienda"""
    longitude: float = Field(..., description="Longitud geográfica", ge=-124.35, le=-114.31)
    latitude: float = Field(..., description="Latitud geográfica", ge=32.54, le=41.95)
    housing_median_age: float = Field(..., description="Edad mediana de las viviendas", ge=1, le=52)
    total_rooms: float = Field(..., description="Total de habitaciones", ge=1)
    total_bedrooms: float = Field(..., description="Total de dormitorios", ge=1)
    population: float = Field(..., description="Población del área", ge=1)
    households: float = Field(..., description="Número de hogares", ge=1)
    median_income: float = Field(..., description="Ingreso mediano (en $10,000s)", ge=0)
    ocean_proximity: Optional[str] = Field("INLAND", description="Proximidad al océano: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND")

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
    """Respuesta de predicción"""
    predicted_price: float
    prediction_range: Optional[dict] = None
    message: str = "Predicción exitosa"


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
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Predicción de Precios de Viviendas",
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
    """Información sobre el modelo cargado"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    info = {
        "model_type": str(type(model).__name__),
        "model_class": str(type(model))
    }
    
    # Intentar obtener más información si está disponible
    if hasattr(model, 'get_params'):
        info["parameters"] = model.get_params()
    
    if hasattr(model, 'feature_names_in_'):
        info["features"] = list(model.feature_names_in_)
    
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HousingFeatures):
    """
    Realiza una predicción individual del precio de vivienda
    
    Args:
        features: Características de la vivienda
        
    Returns:
        Precio predicho en dólares
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preparar los datos como DataFrame con nombres de columnas
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
        
        # Hacer predicción
        prediction = model.predict(X)
        predicted_price = float(prediction[0])
        
        # Calcular rango estimado (±10%)
        prediction_range = {
            "min": predicted_price * 0.9,
            "max": predicted_price * 1.1
        }
        
        return PredictionResponse(
            predicted_price=predicted_price,
            prediction_range=prediction_range,
            message="Predicción exitosa"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Realiza predicciones en batch (múltiples instancias)
    
    Args:
        request: Lista de instancias para predicción
        
    Returns:
        Lista de precios predichos
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preparar datos como DataFrame
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
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")


# Para ejecutar la aplicación
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Cambia "main" por el nombre de tu archivo
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload en desarrollo
    )