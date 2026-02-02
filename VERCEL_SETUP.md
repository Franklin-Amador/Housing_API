# 🚀 Configuración de Variables de Entorno en Vercel

## Paso 1: Ve a tu proyecto en Vercel
1. Entra a [vercel.com](https://vercel.com)
2. Selecciona tu proyecto "Housing_API"
3. Ve a **Settings** → **Environment Variables**

## Paso 2: Agregar la Variable del Modelo

Necesitas agregar **UNA** de estas opciones:

### Opción A: URL Completa (RECOMENDADA) ✅
```
Variable Name:  MODEL_BLOB_URL
Value:          https://tu-blob-id.public.blob.vercel-storage.com/model.onnx
Environments:   Production, Preview, Development (selecciona todos)
```

### Opción B: URL Base
```
Variable Name:  BLOB_PUBLIC_BASE_URL
Value:          https://tu-blob-id.public.blob.vercel-storage.com
Environments:   Production, Preview, Development
```

O también puedes usar:
```
Variable Name:  BLOB_PUBLIC
Value:          https://tu-blob-id.public.blob.vercel-storage.com
```

## Paso 3: ¿Cómo conseguir la URL del Blob?

1. Ve a **Storage** en tu proyecto Vercel
2. Encuentra tu archivo `model.onnx`
3. Copia la **URL pública** completa
4. Debería verse algo así:
   ```
   https://abc123xyz.public.blob.vercel-storage.com/model.onnx
   ```

## Paso 4: Guardar y Re-deploy

1. Haz clic en **Save**
2. Ve a **Deployments**
3. En el último deployment, haz clic en los 3 puntos (**...**)
4. Selecciona **Redeploy** → **Redeploy with existing Build Cache**

## ✅ Verificar que Funciona

Una vez desplegado, revisa los logs:
1. Ve a **Deployments** → selecciona el deployment
2. Haz clic en **View Function Logs**
3. Deberías ver:
   ```
   DEBUG - MODEL_BLOB_URL: configurada
   Cargando modelo ONNX...
   Modelo ONNX cargado correctamente
   ```

## ❌ Si Sigue Fallando

Revisa que:
- La URL sea **pública** (debe contener `.public.blob.vercel-storage.com`)
- No tenga espacios al inicio o final
- Esté disponible en todos los entornos (Production, Preview, Development)
- Hayas hecho **redeploy** después de agregar las variables

## 📝 Notas
- Las variables de entorno **NO se aplican automáticamente** a deployments existentes
- **SIEMPRE** debes hacer redeploy después de cambiar variables
- Los cambios en variables solo afectan nuevos deployments
