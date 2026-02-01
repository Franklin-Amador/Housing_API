# California Housing Price Predictor API

API de predicción de precios de viviendas en California usando FastAPI y scikit-learn.

## ⚠️ Importante: Despliegue

**Vercel no es ideal para ML** debido a limitaciones de tamaño (250 MB max). Recomendamos:

### Opción 1: Railway.app (RECOMENDADO) 🚂
```bash
# 1. Instalar Railway CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Desplegar
railway up
```
- ✅ Soporte completo para Python/ML
- ✅ 500MB RAM gratis
- ✅ Muy fácil de usar

### Opción 2: Render.com 🎨
1. Ve a https://render.com
2. Conecta tu repo de GitHub
3. Selecciona "Web Service"
4. Render detectará automáticamente el `requirements.txt`
5. Deploy!

### Opción 3: Fly.io 🪂
```bash
# 1. Instalar Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Login
fly auth login

# 3. Lanzar app
fly launch

# 4. Desplegar
fly deploy
```

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
