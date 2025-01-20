import numpy as np
from ml_model import EstimacionModel, DataPreprocessor
import tensorflow as tf


def test_estimaciones():
    """Test the trained model with different test cases"""

    # Cargar el modelo entrenado
    try:
        config = {
            "vocab_size": 6,  # 5 tipos de tareas + 1 para padding
            "lstm_units": 32,
            "dense_units": [64, 32],
            "dropout_rate": 0.2,
        }
        model = EstimacionModel(config)
        model.model = tf.keras.models.load_model("models/modelo_estimacion.keras")
    except Exception as e:
        print(f"Error: No se pudo cargar el modelo: {str(e)}")
        return

    # Inicializar el preprocessor
    preprocessor = DataPreprocessor()
    tipos_tarea = ["backend", "frontend", "database", "testing", "deployment"]
    preprocessor.fit_tokenizer(tipos_tarea)

    # Casos de prueba representativos basados en estimacion_tiempos.csv
    test_cases = [
        # Casos diversos basados en el dataset
        # (complejidad, prioridad, tipo_tarea)
        (1, 1, "backend"),  # Tarea simple backend baja prioridad
        (3, 2, "frontend"),  # Tarea media frontend prioridad media
        (5, 3, "database"),  # Tarea compleja database alta prioridad
        (4, 2, "backend"),  # Tarea alta backend prioridad media
        (2, 3, "testing"),  # Tarea baja testing alta prioridad
        (3, 1, "deployment"),  # Tarea media deployment baja prioridad
    ]

    print("\nPruebas de Estimación de Duración:")
    print("====================================")

    for comp, prior, tipo in test_cases:
        try:
            # Preparar datos de entrada
            X_num = np.array([[comp, prior]], dtype=np.float32)
            X_task = np.array(preprocessor.encode_task_types([tipo]))

            # Realizar predicción
            resultado = model.predict_individual_task(X_num, X_task)

            print(f"\nTipo de tarea: {tipo}")
            print(f"Complejidad: {comp}")
            print(f"Prioridad: {prior}")
            print(f"Tiempo estimado: {resultado['tiempo_estimado']:.2f} horas")
            print("------------------------------------")

        except Exception as e:
            print(f"Error en predicción para {tipo}: {str(e)}")


if __name__ == "__main__":
    test_estimaciones()
