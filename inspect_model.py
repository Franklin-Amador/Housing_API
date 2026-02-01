import dill
import sys

# Cargar el modelo
try:
    with open('model_sklearn_1_7_2.pkl', 'rb') as f:
        model = dill.load(f)
    
    print("=== TIPO DE MODELO ===")
    print(f"Tipo: {type(model)}")
    print(f"Nombre: {type(model).__name__}")
    print()
    
    print("=== ESTRUCTURA ===")
    if hasattr(model, 'named_steps'):
        print("Es un Pipeline con estos pasos:")
        for name, step in model.named_steps.items():
            print(f"  - {name}: {type(step).__module__}.{type(step).__name__}")
            if hasattr(step, 'named_transformers_'):
                print(f"    Tiene transformadores:")
                for t_name, transformer in step.named_transformers_.items():
                    print(f"      - {t_name}: {type(transformer).__module__}.{type(transformer).__name__}")
    else:
        print(f"No es pipeline. Es: {type(model).__module__}.{type(model).__name__}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
