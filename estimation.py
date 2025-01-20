import os
import numpy as np
import tensorflow as tf
from ml_model import EstimacionModel, DataPreprocessor
import traceback
from sklearn.model_selection import train_test_split


def setup_environment():
    """Configure environment settings"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    tf.random.set_seed(42)
    np.random.seed(42)


def load_sample_data():
    """Load and prepare sample data"""
    X_numeric = np.array([[3, 2, 5], [5, 3, 10], [1, 1, 2]], dtype=np.float32)
    task_types = ["backend", "frontend", "database"]
    y = np.array([10, 20, 3], dtype=np.float32)
    return X_numeric, task_types, y


def prepare_data():
    """Prepare and split training data"""
    # Extend sample data
    X_numeric = np.array(
        [[3, 2, 5], [5, 3, 10], [1, 1, 2], [4, 2, 6], [2, 1, 4], [3, 2, 5]],
        dtype=np.float32,
    )

    task_types = ["backend", "frontend", "database", "backend", "frontend", "database"]

    y = np.array([10, 20, 3, 12, 8, 11], dtype=np.float32)

    # Split data
    X_num_train, X_num_val, types_train, types_val, y_train, y_val = train_test_split(
        X_numeric, task_types, y, test_size=0.2, random_state=42
    )
    return X_num_train, X_num_val, types_train, types_val, y_train, y_val


def main():
    try:
        setup_environment()

        # Prepare data
        X_num_train, X_num_val, types_train, types_val, y_train, y_val = prepare_data()

        # Preprocess
        preprocessor = DataPreprocessor()
        preprocessor.fit_tokenizer(types_train)
        X_task_train = np.array(preprocessor.encode_task_types(types_train))
        X_task_val = np.array(preprocessor.encode_task_types(types_val))

        # Model config
        config = {
            "vocab_size": 10,
            "lstm_units": 32,
            "dense_units": [64, 32],
            "dropout_rate": 0.2,
        }

        # Train
        model = EstimacionModel(config)
        history = model.train(
            [X_num_train, X_task_train],
            y_train,
            validation_data=([X_num_val, X_task_val], y_val),
            epochs=100,
        )

        # Save
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
