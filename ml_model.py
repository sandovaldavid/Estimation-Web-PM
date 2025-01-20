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
        x = Embedding(self.config["vocab_size"], 32, input_length=1)(task_input)
        x = LSTM(32, return_sequences=True)(x)
        x = LSTM(16)(x)
        x = Dropout(0.2)(x)
        return task_input, x

    def _build_model(self):
        """Construye el modelo completo combinando las ramas"""
        # Rama numérica
        numeric_input, numeric_features = self._build_numeric_branch()

        # Rama de tipos de tarea
        task_input, task_features = self._build_task_type_branch()

        # Combinar características
        combined = Concatenate()([numeric_features, task_features])

        # Capas densas finales
        x = Dense(128, activation="relu")(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        output = Dense(1, activation="linear", name="time_estimate")(x)

        # Crear modelo
        model = Model(inputs=[numeric_input, task_input], outputs=output)

        # Compilar
        model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae", "mse"])

        return model

    def get_callbacks(self):
        """Retorna callbacks para entrenamiento"""
        return [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
        ]

    def train(self, X_num, X_task, y, validation_data=None, epochs=100):
        """
        Entrena el modelo

        Args:
            X_num: Features numéricos
            X_task: Tipos de tarea codificados
            y: Tiempos objetivo
            validation_data: Datos de validación (opcional)
            epochs: Número de épocas
        """
        return self.model.fit(
            [X_num, X_task],
            y,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=self.get_callbacks(),
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
