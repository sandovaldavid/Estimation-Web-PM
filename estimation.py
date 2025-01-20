import os
import numpy as np
import tensorflow as tf
from ml_model import EstimacionModel, DataPreprocessor
import traceback
from sklearn.model_selection import train_test_split
import pandas as pd

def setup_environment():
    """Configura el ambiente para TensorFlow"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_and_process_data():
    """
    Carga y procesa los datos del CSV para el modelo de estimación

    Returns:
        tuple: (X_num_train, X_num_val, X_task_train, X_task_val, y_train, y_val, vocab_size)
    """
    try:
        # Cargar datos
        data = pd.read_csv("estimacion_tiempos.csv")

        # Validar columnas requeridas
        required_columns = [
            "idrequerimiento",  # Añadido para identificar requerimientos
            "complejidad",
            "prioridad",
            "tipo_tarea",
            "duracion",
        ]
        if not all(col in data.columns for col in required_columns):
            raise ValueError("El CSV no contiene todas las columnas requeridas")

        # Validar rangos de datos
        if not (
            data["complejidad"].between(1, 5).all()
            and data["prioridad"].between(1, 3).all()
        ):
            raise ValueError(
                "Valores fuera de rango en complejidad (1-5) o prioridad (1-3)"
            )

        # Inicializar el preprocessor
        preprocessor = DataPreprocessor()

        # Obtener y validar tipos de tareas únicos
        tipos_tarea_unicos = sorted(data["tipo_tarea"].unique())
        expected_tasks = ["backend", "frontend", "database", "testing", "deployment"]
        if not all(task in tipos_tarea_unicos for task in expected_tasks):
            raise ValueError(f"Faltan tipos de tarea. Esperados: {expected_tasks}")

        print("Tipos de tareas encontrados:", tipos_tarea_unicos)

        # Procesar tipos de tarea
        preprocessor.fit_tokenizer(tipos_tarea_unicos)
        tipos_tarea_encoded = preprocessor.encode_task_types(data["tipo_tarea"].values)
        tipos_tarea_encoded = np.array(tipos_tarea_encoded)

        # Preparar features numéricas (solo complejidad y prioridad)
        X_numeric = data[["complejidad", "prioridad"]].values
        y = data["duracion"].values

        # Validar que no hay valores nulos
        if np.isnan(X_numeric).any() or np.isnan(y).any():
            raise ValueError("Hay valores nulos en los datos")

        # División estratificada por tipo de tarea
        X_num_train, X_num_val, X_task_train, X_task_val, y_train, y_val = (
            train_test_split(
                X_numeric,
                tipos_tarea_encoded,
                y,
                test_size=0.2,
                random_state=42,
                stratify=data["tipo_tarea"],
            )
        )

        return (
            X_num_train,
            X_num_val,
            X_task_train,
            X_task_val,
            y_train,
            y_val,
            len(tipos_tarea_unicos) + 1,
        )

    except FileNotFoundError:
        raise FileNotFoundError("No se encontró el archivo estimacion_tiempos.csv")
    except Exception as e:
        raise Exception(f"Error al procesar los datos: {str(e)}")

def main():
    try:
        # Setup
        setup_environment()

        # Cargar y procesar datos
        X_num_train, X_num_val, X_task_train, X_task_val, y_train, y_val, vocab_size = (
            load_and_process_data()
        )

        # Configuración del modelo
        config = {
            "vocab_size": vocab_size,
            "lstm_units": 32,
            "dense_units": [64, 32],
            "dropout_rate": 0.2,
        }

        # Crear modelo
        model = EstimacionModel(config)

        # Normalizar datos
        X_num_train_norm, scaler = model.normalize_data(X_num_train)
        X_num_val_norm = scaler.transform(X_num_val)

        # Validación cruzada
        mean_score, std_score = model.cross_validate_model(
            X_num_train_norm, X_task_train, y_train
        )
        print(f"CV Score: {mean_score:.4f} (+/- {std_score:.4f})")

        # Entrenar modelo final
        history = model.train(
            [X_num_train_norm, X_task_train],
            y_train,
            validation_data=([X_num_val_norm, X_task_val], y_val),
            epochs=100,
        )

        # Analizar importancia de features
        feature_names = ["Complejidad", "Prioridad"]  # Solo estas dos características numéricas
        importance_scores = model.analyze_feature_importance(
            X_num_train_norm, 
            X_task_train, 
            y_train, 
            feature_names
        )

        print("\nImportancia de características:")
        print("===============================")
        for feature, score in importance_scores:
            print(f"{feature}: {score:.4f}")

        # Guardar modelo
        model.model.save("models/modelo_estimacion.keras")
        print("Modelo guardado exitosamente")

        return history

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        print(traceback.format_exc())
    return None

if __name__ == "__main__":
    main()
