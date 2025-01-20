import os
import numpy as np
import tensorflow as tf
from ml_model import EstimacionModel, DataPreprocessor
import traceback


def setup_environment():
    """Configure environment settings"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging
    tf.random.set_seed(42)
    np.random.seed(42)


def load_sample_data():
    """Load and prepare sample data"""
    X_numeric = np.array([[3, 2, 5], [5, 3, 10], [1, 1, 2]], dtype=np.float32)
    task_types = ["backend", "frontend", "database"]
    y = np.array([10, 20, 3], dtype=np.float32)
    return X_numeric, task_types, y


def main():
    try:
        # Setup
        setup_environment()
        X_numeric, task_types, y = load_sample_data()

        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.fit_tokenizer(task_types)
        X_task = np.array(preprocessor.encode_task_types(task_types))

        # Model configuration
        config = {
            "vocab_size": 50,
            "lstm_units": 32,
            "dense_units": [128, 64],
            "dropout_rate": 0.2,
        }

        # Train model
        model = EstimacionModel(config)
        history = model.train(X_numeric, X_task, y, epochs=100)

        # Save model and preprocessor
        model.model.save("modelo_estimacion.h5")
        print("Modelo guardado exitosamente")

        return history
    except ImportError as e:
        print(f"Error de importación: {str(e)}")
        print(
            "Por favor, verifica que todas las dependencias estén instaladas correctamente"
        )
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        print("Stacktrace completo:", traceback.format_exc())
    return None


if __name__ == "__main__":
    main()
