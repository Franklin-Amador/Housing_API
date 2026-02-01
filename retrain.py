"""
Script para re-entrenar el modelo del Capítulo 2 de Hands-On ML
Compatible con scikit-learn 1.7.2
"""

import dill
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils import check_random_state
import sklearn

# Asegurar que estas funciones estén disponibles globalmente para dill
globals()['check_is_fitted'] = check_is_fitted
globals()['check_array'] = check_array
globals()['check_random_state'] = check_random_state

print(f"📦 Usando scikit-learn versión: {sklearn.__version__}")


def load_california_housing_data():
    """
    Carga el dataset de California Housing
    Puedes usar sklearn.datasets o cargar tu propio CSV
    """
    from sklearn.datasets import fetch_california_housing
    
    print("📥 Cargando datos de California Housing...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    # Renombrar a los nombres esperados por el pipeline
    rename_map = {
        'MedInc': 'median_income',
        'HouseAge': 'housing_median_age',
        'AveRooms': 'total_rooms',
        'AveBedrms': 'total_bedrooms',
        'Population': 'population',
        'AveOccup': 'households',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'MedHouseVal': 'median_house_value',
    }
    df = df.rename(columns=rename_map)

    # Añadir ocean_proximity simulado (ya que el dataset original no lo tiene)
    # En el libro original, esto viene del CSV
    np.random.seed(42)
    ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
    df['ocean_proximity'] = np.random.choice(
        ocean_proximity_options, 
        size=len(df), 
        p=[0.3, 0.4, 0.15, 0.10, 0.05]
    )
    
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value'] * 100000  # Convertir a dólares
    
    print(f"✅ Datos cargados: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"   Features: {list(X.columns)}")
    
    return X, y


def load_from_csv(csv_path):
    """
    Alternativa: Cargar desde tu propio CSV del capítulo 2
    """
    print(f"📥 Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Separar features y target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    print(f"✅ Datos cargados: {X.shape[0]} muestras, {X.shape[1]} features")
    return X, y


def create_preprocessing_pipeline():
    """
    Crea el pipeline de preprocesamiento del Capítulo 2
    """
    # Columnas numéricas
    num_features = [
        'longitude', 'latitude', 'housing_median_age', 
        'total_rooms', 'total_bedrooms', 'population', 
        'households', 'median_income'
    ]
    
    # Columna categórica
    cat_features = ['ocean_proximity']
    
    # Pipeline para features numéricas
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Transformador completo
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    return preprocessor


def train_model(X, y, model_type='random_forest'):
    """
    Entrena el modelo completo con pipeline
    """
    print(f"\n🎯 Entrenando modelo: {model_type}")
    print("=" * 60)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train: {X_train.shape[0]} muestras")
    print(f"📊 Test:  {X_test.shape[0]} muestras")
    
    # Crear pipeline completo
    preprocessor = create_preprocessing_pipeline()
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    
    # Pipeline completo
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Entrenar
    print("\n⏳ Entrenando modelo...")
    full_pipeline.fit(X_train, y_train)
    
    # Evaluar
    train_score = full_pipeline.score(X_train, y_train)
    test_score = full_pipeline.score(X_test, y_test)
    
    print(f"\n✅ Modelo entrenado!")
    print(f"   R² Train: {train_score:.4f}")
    print(f"   R² Test:  {test_score:.4f}")
    
    # Hacer algunas predicciones de prueba
    print(f"\n🧪 Predicciones de prueba:")
    sample_predictions = full_pipeline.predict(X_test[:5])
    for i, (pred, actual) in enumerate(zip(sample_predictions, y_test[:5]), 1):
        print(f"   {i}. Predicho: ${pred:,.0f} | Real: ${actual:,.0f}")
    
    return full_pipeline


def save_model(model, filepath="model_sklearn_1_7_2.pkl"):
    """
    Guarda el modelo con dill incluyendo todas las referencias necesarias
    """
    print(f"\n💾 Guardando modelo en {filepath}...")
    
    try:
        # Configurar dill para serializar todo correctamente
        dill.settings['recurse'] = True
        
        with open(filepath, "wb") as f:
            dill.dump(model, f)
        
        # Verificar tamaño
        import os
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ Modelo guardado exitosamente")
        print(f"📦 Tamaño: {size_mb:.2f} MB")
        
        # Verificar que se puede cargar
        print(f"\n🔍 Verificando modelo guardado...")
        
        # Asegurar imports para la carga
        from sklearn.utils.validation import check_is_fitted, check_array
        globals()['check_is_fitted'] = check_is_fitted
        globals()['check_array'] = check_array
        
        with open(filepath, "rb") as f:
            loaded_model = dill.load(f)
        
        # Probar predicción
        test_data = pd.DataFrame({
            'longitude': [-122.23],
            'latitude': [37.88],
            'housing_median_age': [41.0],
            'total_rooms': [880.0],
            'total_bedrooms': [129.0],
            'population': [322.0],
            'households': [126.0],
            'median_income': [8.3252],
            'ocean_proximity': ['NEAR BAY']
        })
        
        prediction = loaded_model.predict(test_data)
        print(f"✅ Verificación exitosa")
        print(f"   Predicción de prueba: ${prediction[0]:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al guardar: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Proceso completo de entrenamiento y guardado
    """
    print("=" * 60)
    print("🏠 Re-entrenamiento de modelo California Housing")
    print("   Compatible con scikit-learn 1.7.2")
    print("=" * 60)
    print()
    
    # Opción 1: Usar datos de sklearn
    print("📋 OPCIÓN 1: Usar dataset de sklearn.datasets")
    print("📋 OPCIÓN 2: Cargar desde CSV propio")
    print()
    
    # Cargar datos (usa la opción que prefieras)
    try:
        X, y = load_california_housing_data()
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        print("\n💡 Si tienes un CSV del capítulo 2, usa:")
        print("   X, y = load_from_csv('housing.csv')")
        return
    
    # Entrenar modelo
    model = train_model(X, y, model_type='random_forest')
    
    # Guardar modelo
    success = save_model(model, "model_sklearn_1_7_2.pkl")
    
    if success:
        print("\n" + "=" * 60)
        print("✅ PROCESO COMPLETADO")
        print("=" * 60)
        print(f"\n📝 Próximos pasos:")
        print(f"   1. Usa 'model_sklearn_1_7_2.pkl' en tu API")
        print(f"   2. Actualiza MODEL_PATH en main.py")
        print(f"   3. Reinicia el servidor FastAPI")
        print(f"\n🚀 Tu API ahora funcionará con sklearn 1.7.2!")
    else:
        print("\n❌ Hubo un problema guardando el modelo")


if __name__ == "__main__":
    main()