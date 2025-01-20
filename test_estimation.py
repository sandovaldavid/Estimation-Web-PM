import numpy as np
from ml_model import EstimacionModel, DataPreprocessor
import tensorflow as tf
import joblib


def test_estimaciones():
    """Test the trained model with different test cases"""
    try:
        config = {
            "vocab_size": 6,
            "lstm_units": 32,
            "dense_units": [64, 32],
            "dropout_rate": 0.2,
        }
        model = EstimacionModel(config)
        model.model = tf.keras.models.load_model("models/modelo_estimacion.keras")

        # Cargar preprocessors y scalers
        preprocessor = joblib.load("models/preprocessor.pkl")
        scaler_num = joblib.load("models/scaler.pkl")
        scaler_req = joblib.load("models/scaler_req.pkl")

    except Exception as e:
        print(f"Error: No se pudo cargar el modelo o preprocessors: {str(e)}")
        return

    # Casos de prueba
    test_cases = [
        # (idreq, complejidad, prioridad, tipo_tarea)
        (1, 1, 1, "backend"),  # Tarea simple backend
        (1, 3, 2, "frontend"),  # Tarea media frontend
        (2, 5, 3, "database"),  # Tarea compleja database
        (2, 4, 2, "backend"),  # Tarea alta backend
        (3, 2, 3, "testing"),  # Tarea baja testing
    ]

    print("\nPruebas de Estimación de Duración:")
    print("====================================")

    for idreq, comp, prior, tipo in test_cases:
        try:
            # Preparar datos numéricos (2 características)
            X_num = np.array([[comp, prior]], dtype=np.float32)
            X_task = np.array(preprocessor.encode_task_types([tipo]))

            # Generar características del requerimiento (4 características)
            X_req = np.array([[comp, comp, 1, prior]], dtype=np.float32)

            # Normalizar datos usando los scalers correctos
            X_num_norm = scaler_num.transform(X_num)
            X_req_norm = scaler_req.transform(X_req)

            # Realizar predicción
            resultado = model.predict_individual_task(X_num_norm, X_task, X_req_norm)

            print(f"\nRequerimiento ID: {idreq}")
            print(f"Tipo de tarea: {tipo}")
            print(f"Complejidad: {comp}")
            print(f"Prioridad: {prior}")
            print(f"Estimación de duración: {resultado['tiempo_estimado']:.2f} horas")
            print("------------------------------------")

        except Exception as e:
            print(f"Error en predicción para {tipo}: {str(e)}")


if __name__ == "__main__":
    test_estimaciones()
