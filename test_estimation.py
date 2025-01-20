import numpy as np
from ml_model import EstimacionModel, DataPreprocessor
import tensorflow as tf


def test_estimaciones():
    """Test the trained model with different test cases"""

    # Cargar el modelo entrenado
    try:
        model = tf.keras.models.load_model("models/modelo_estimacion.keras")
    except:
        print(
            "Error: No se pudo cargar el modelo. Asegúrate de que existe el archivo 'models/modelo_estimacion.keras'"
        )
        return

    # Inicializar el preprocessor
    preprocessor = DataPreprocessor()
    tipos_tarea = ["backend", "frontend", "database"]
    preprocessor.fit_tokenizer(tipos_tarea)

    # Casos de prueba
    test_cases = [
        # (complejidad, prioridad, subtareas, tipo_tarea)
        (1, 1, 2, "backend"),  # Tarea simple backend
        (3, 2, 5, "frontend"),  # Tarea media frontend
        (5, 3, 10, "database"),  # Tarea compleja database
        (4, 2, 7, "backend"),  # Tarea media-alta backend
        (2, 3, 3, "frontend"),  # Tarea simple frontend
    ]

    print("\nPruebas de Estimación de Duración:")
    print("====================================")

    for comp, prior, sub, tipo in test_cases:
        # Preparar datos de entrada
        X_num = np.array([[comp, prior, sub]], dtype=np.float32)
        X_task = np.array(preprocessor.encode_task_types([tipo]))

        # Realizar predicción
        estimacion = model.predict([X_num, X_task], verbose=0)

        # Mostrar resultados
        print(f"\nTipo de tarea: {tipo}")
        print(f"Complejidad: {comp}")
        print(f"Prioridad: {prior}")
        print(f"Subtareas: {sub}")
        print(f"Estimación de duración: {estimacion[0][0]:.2f} horas")
        print("------------------------------------")


if __name__ == "__main__":
    test_estimaciones()
