# California Housing Price Predictor API

API de predicción de precios de viviendas en California usando FastAPI y scikit-learn.

## 🚀 Despliegue en Vercel

Este proyecto está listo para desplegarse en Vercel:

1. Sube el código a GitHub
2. Importa el repositorio en Vercel
3. Configura la variable de entorno (opcional, ya que usamos URL pública del blob)
4. Despliega

## 📦 Modelo

El modelo está alojado en Vercel Blob Storage:
- URL: `https://vjrbqsew9s3w1szr.public.blob.vercel-storage.com/model_sklearn_1_7_2.pkl`

## 🛠️ Desarrollo Local

```bash
# Activar entorno virtual
.\mls\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
python main.py
```

La API estará disponible en `http://localhost:8000`

## 📚 Documentación

- API Docs: `/docs`
- ReDoc: `/redoc`

## 🔑 Endpoints

- `POST /predict` - Predice precio de una vivienda
- `POST /predict/batch` - Predice precios de múltiples viviendas
- `GET /model/info` - Información del modelo
- `GET /health` - Estado del servicio
