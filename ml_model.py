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
    Add
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.layers import Reshape
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np

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

    def normalize_data(self, X_numeric):
        """
        Normaliza las características numéricas usando StandardScaler
        
        Args:
            X_numeric: Array con features numéricas [complejidad, prioridad, tareas_requerimiento]
        Returns:
            X_normalized: Array normalizado
            scaler: Objeto StandardScaler entrenado
        """
        self.scaler = StandardScaler()
        X_normalized = self.scaler.fit_transform(X_numeric)
        return X_normalized, self.scaler

    def _build_numeric_branch(self):
        """Construye la rama para procesar características numéricas"""
        numeric_input = Input(shape=(2,), name="numeric_input")  # Changed from 3 to 2
        x1 = Dense(64, activation="relu", name="numeric_dense")(numeric_input)
        x1 = BatchNormalization(name="numeric_batch_norm")(x1)
        x1 = Dropout(self.config["dropout_rate"], name="numeric_dropout")(x1)
        return numeric_input, x1

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
        """Construye el modelo completo para estimación de tiempos"""
        # Rama numérica (2 features: complejidad, prioridad)
        numeric_input, x1 = self._build_numeric_branch()

        # Rama de tipo de tarea
        task_input = Input(shape=(1,), name="task_input")
        x2 = Embedding(self.config["vocab_size"], 32, name="task_embedding")(task_input)
        x2 = Reshape((32,), name="task_reshape")(x2)
        x2 = Dense(32, activation="relu", name="task_dense")(x2)
        x2 = BatchNormalization(name="task_batch_norm")(x2)
        x2 = Dropout(self.config["dropout_rate"], name="task_dropout")(x2)

        # Combinar características
        combined = Concatenate(name="concatenate")([x1, x2])

        # Capas densas finales
        x = Dense(128, activation="relu", name="combined_dense_1")(combined)
        x = BatchNormalization(name="combined_batch_norm_1")(x)
        x = Dropout(self.config["dropout_rate"], name="combined_dropout_1")(x)

        x = Dense(64, activation="relu", name="combined_dense_2")(x)
        x = BatchNormalization(name="combined_batch_norm_2")(x)
        x = Dropout(self.config["dropout_rate"], name="combined_dropout_2")(x)

        output = Dense(1, activation="linear", name="output")(x)

        model = Model(inputs=[numeric_input, task_input], outputs=output)
        model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

        return model

    def cross_validate_model(self, X_num, X_task, y, n_splits=5):
        """
        Implementa validación cruzada

        Args:
            X_num: Features numéricas
            X_task: Tipos de tarea codificados
            y: Valores objetivo (duracion)
            n_splits: Número de divisiones para CV
        Returns:
            mean_score: Score promedio
            std_score: Desviación estándar del score
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(X_num):
            # Dividir datos
            X_num_train, X_num_val = X_num[train_idx], X_num[val_idx]
            X_task_train, X_task_val = X_task[train_idx], X_task[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Normalizar datos numéricos
            X_num_train_norm, scaler = self.normalize_data(X_num_train)
            X_num_val_norm = scaler.transform(X_num_val)

            # Entrenar modelo
            history = self.train(
                [X_num_train_norm, X_task_train],
                y_train,
                validation_data=([X_num_val_norm, X_task_val], y_val),
                epochs=100,
            )

            # Guardar score de validación
            val_score = min(history.history["val_loss"])
            scores.append(val_score)

        return np.mean(scores), np.std(scores)

    def analyze_feature_importance(self, X_num, X_task, y, feature_names=None):
        """
        Analiza la importancia de todas las características incluyendo tipo de tarea
        
        Args:
            X_num: Features numéricas
            X_task: Tipos de tarea codificados
            y: Valores objetivo
            feature_names: Nombres de las características numéricas
        Returns:
            importance_scores: Lista de tuplas (feature, score)
        """
        if feature_names is None:
            feature_names = ['Complejidad', 'Prioridad']

        # Predicción base
        base_prediction = self.model.predict([X_num, X_task])
        importance_scores = []

        # Analizar features numéricas
        for i in range(X_num.shape[1]):
            X_modified = X_num.copy()
            X_modified[:, i] = np.mean(X_num[:, i])
            new_prediction = self.model.predict([X_modified, X_task])
            importance = np.mean(np.abs(base_prediction - new_prediction))
            importance_scores.append((feature_names[i], importance))

        # Analizar importancia del tipo de tarea
        X_task_modified = np.full_like(X_task, np.mean(X_task))
        task_prediction = self.model.predict([X_num, X_task_modified])
        task_importance = np.mean(np.abs(base_prediction - task_prediction))
        importance_scores.append(("Tipo de Tarea", task_importance))

        # Ordenar por importancia
        return sorted(importance_scores, key=lambda x: x[1], reverse=True)

    def predict_individual_task(self, X_num, X_task):
        """
        Predice el tiempo para una tarea individual
        
        Args:
            X_num: Array con [complejidad, prioridad]
            X_task: Tipo de tarea codificado
        Returns:
            dict: Diccionario con el tiempo estimado para la tarea
        """
        # Realizar predicción
        prediction = self.model.predict([X_num, X_task])

        return {
            'tiempo_estimado': prediction[0][0],
            'complejidad': X_num[0][0],
            'prioridad': X_num[0][1],
            'tipo_tarea': X_task[0]
        }

    def get_callbacks(self):
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
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
