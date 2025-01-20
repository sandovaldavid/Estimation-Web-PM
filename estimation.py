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
    """Carga y procesa los datos del CSV"""
    # Cargar datos
    data = pd.read_csv("estimacion_tiempos.csv")

    # Inicializar el preprocessor
    preprocessor = DataPreprocessor()

    # Obtener tipos de tareas únicos
    tipos_tarea_unicos = sorted(data["tipo_tarea"].unique())
    print("Tipos de tareas encontrados:", tipos_tarea_unicos)

    # Importante: El vocab_size debe ser el número de tipos únicos + 1
    vocab_size = len(tipos_tarea_unicos) + 1
    print(f"Tamaño del vocabulario: {vocab_size}")

    # Entrenar el tokenizer y codificar los tipos de tarea
    preprocessor.fit_tokenizer(tipos_tarea_unicos)
    tipos_tarea_encoded = preprocessor.encode_task_types(data["tipo_tarea"].values)
    tipos_tarea_encoded = np.array(tipos_tarea_encoded)

    # Separar features y target
    X_numeric = data[["complexidad", "prioridad", "tareas_requerimiento"]].values
    y = data["duracion"].values

    # Dividir datos en train y test
    X_num_train, X_num_val, X_task_train, X_task_val, y_train, y_val = train_test_split(
        X_numeric, tipos_tarea_encoded, y, test_size=0.2, random_state=42
    )

    return (
        X_num_train,
        X_num_val,
        X_task_train,
        X_task_val,
        y_train,
        y_val,
        vocab_size,
    )


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
            "vocab_size": vocab_size,  # Usar el tamaño calculado dinámicamente
            "lstm_units": 32,
            "dense_units": [64, 32],
            "dropout_rate": 0.2,
        }

        # Crear y entrenar modelo
        model = EstimacionModel(config)
        history = model.train(
            [X_num_train, X_task_train],
            y_train,
            validation_data=([X_num_val, X_task_val], y_val),
            epochs=100,
        )

        # Guardar modelo
        model.model.summary()
        model.model.save("models/modelo_estimacion.keras")
        print("Modelo guardado exitosamente")

        return history

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        print(traceback.format_exc())
    return None


if __name__ == "__main__":
    main()
