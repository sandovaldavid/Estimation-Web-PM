import numpy as np
from ml_model import EstimacionModel, DataPreprocessor

# Configuración de reproducibilidad
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

# Datos de ejemplo
X_numeric = np.array([[3, 2, 5], [5, 3, 10], [1, 1, 2]], dtype=np.float32)
task_types = ["backend", "frontend", "database"]
y = np.array([10, 20, 3], dtype=np.float32)

# Preprocesar datos
preprocessor = DataPreprocessor()
preprocessor.fit_tokenizer(task_types)
X_task = np.array(preprocessor.encode_task_types(task_types))

# Crear y entrenar modelo
config = {
    "vocab_size": 50,
    "lstm_units": 32,
    "dense_units": [128, 64],
    "dropout_rate": 0.2,
}

model = EstimacionModel(config)
history = model.train(X_numeric, X_task, y, epochs=100)

# Guardar el modelo entrenado
model.model.save("modelo_estimacion.h5")
