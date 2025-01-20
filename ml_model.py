# ml_model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LSTM,
    Input,
    Concatenate,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.layers import Reshape

class EstimacionModel:
    """Modelo para estimar tiempos de proyectos usando RNN"""

    def __init__(self, config):
        """
        Inicializa el modelo con la configuración dada

        Args:
            config (dict): Diccionario con parámetros de configuración
        """
        self.config = config
        self.model = self._build_model()

    def _build_numeric_branch(self):
        """Construye la rama para procesar características numéricas"""
        numeric_input = Input(shape=(3,), name="numeric_input")
        x = Dense(64, activation="relu")(numeric_input)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return numeric_input, x

    def _build_task_type_branch(self):
        """Construye la rama para procesar tipos de tareas"""
        task_input = Input(shape=(1,), name="task_input")
        x = Embedding(
            self.config["vocab_size"],
            32,
            name="embedding"
        )(task_input)
        x = Reshape((1, 32), name="reshape")(x)
        x = LSTM(
            32,
            return_sequences=False,
            name="lstm"
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return task_input, x

    def _build_model(self):
        """Construye el modelo completo"""
        numeric_input, numeric_features = self._build_numeric_branch()
        task_input, task_features = self._build_task_type_branch()

        combined = Concatenate(name="concat")([numeric_features, task_features])
        x = Dense(64, activation="relu")(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu")(x)
        output = Dense(1, name="output")(x)

        model = Model(inputs=[numeric_input, task_input], outputs=output)

        model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

        return model

    def get_callbacks(self):
        """Retorna callbacks para entrenamiento"""
        return [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
        ]

    def train(self, inputs, targets, validation_data=None, epochs=100):
        """Train model with validation"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss" if validation_data is None else "val_loss",
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "models/best_model.keras",
                monitor="loss" if validation_data is None else "val_loss",
                save_best_only=True,
            ),
        ]

        return self.model.fit(
            inputs,
            targets,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

    def predict(self, X_num, X_task):
        """Realiza predicciones"""
        return self.model.predict([X_num, X_task])


class DataPreprocessor:
    """Clase para preprocesamiento de datos"""

    def __init__(self):
        self.tokenizer = Tokenizer()

    def fit_tokenizer(self, task_types):
        """Entrena el tokenizer con tipos de tarea"""
        self.tokenizer.fit_on_texts(task_types)

    def encode_task_types(self, task_types):
        """Codifica tipos de tarea"""
        return self.tokenizer.texts_to_sequences(task_types)

    def decode_task_types(self, encoded_tasks):
        """Decodifica tipos de tarea"""
        return self.tokenizer.sequences_to_texts(encoded_tasks)
