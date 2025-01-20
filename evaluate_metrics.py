import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import joblib


def load_and_preprocess_data():
    """Carga y preprocesa los datos para evaluación"""
    try:
        # Cargar datos
        data = pd.read_csv("estimacion_tiempos.csv")

        # Calcular estadísticas por requerimiento
        req_stats = (
            data.groupby("idrequerimiento")
            .agg(
                {
                    "complejidad": ["mean", "max", "count"],
                    "duracion": "sum",
                    "prioridad": "mean",
                }
            )
            .reset_index()
        )

        # Renombrar columnas agregadas
        req_stats.columns = [
            "idrequerimiento",
            "complejidad_media_req",
            "complejidad_max_req",
            "num_tareas_req",
            "duracion_total_req",
            "prioridad_media_req",
        ]

        # Unir con datos originales
        data = data.merge(req_stats, on="idrequerimiento")

        # Separar features
        X_numeric = data[["complejidad", "prioridad"]].values
        X_task = data["tipo_tarea"].values
        X_req = data[
            [
                "complejidad_media_req",
                "complejidad_max_req",
                "num_tareas_req",
                "prioridad_media_req",
            ]
        ].values
        y = data["duracion"].values

        # Dividir datos
        (
            X_num_train,
            X_num_test,
            X_task_train,
            X_task_test,
            X_req_train,
            X_req_test,
            y_train,
            y_test,
        ) = train_test_split(
            X_numeric, X_task, X_req, y, test_size=0.2, random_state=42
        )

        print("\nEstadísticas del dataset:")
        print(f"Número total de requerimientos: {data['idrequerimiento'].nunique()}")
        print(f"Número total de tareas: {len(data)}")
        print(f"Promedio de duración: {y.mean():.2f} horas")
        print(f"Mediana de duración: {np.median(y):.2f} horas")
        print(f"Desviación estándar: {y.std():.2f} horas")

        return {
            "train": [X_num_train, X_req_train, X_task_train, y_train],
            "test": [X_num_test, X_req_test, X_task_test, y_test],
        }

    except Exception as e:
        print(f"Error en carga de datos: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred, threshold=0.1):
    """Calcula métricas de rendimiento"""
    # Métricas de regresión
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Convertir a clasificación binaria para métricas adicionales
    y_true_class = (y_true >= np.mean(y_true)).astype(int)
    y_pred_class = (y_pred >= np.mean(y_pred)).astype(int)

    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(
        y_true_class, y_pred_class, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_true_class, y_pred_class, average="weighted", zero_division=0
    )
    f1 = f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

def evaluate_model():
    """Evalúa el modelo entrenado usando múltiples métricas"""
    try:
        # Cargar datos
        data = load_and_preprocess_data()
        if data is None:
            return None

        # Cargar modelo y preprocessors
        model = tf.keras.models.load_model("models/modelo_estimacion.keras")
        preprocessor = joblib.load("models/preprocessor.pkl")
        
        # Cargar scalers separados
        scaler_num = joblib.load("models/scaler.pkl")
        scaler_req = joblib.load("models/scaler_req.pkl")  # Necesitamos crear este scaler

        # Procesar los datos
        X_num_test = data["test"][0]  # Features numéricas
        X_req_test = data["test"][1]  # Info del requerimiento
        X_task_test = data["test"][2]  # Tipos de tarea
        y_test = data["test"][3]      # Valores reales

        # Codificar tipos de tarea
        X_task_encoded = preprocessor.encode_task_types(X_task_test)

        # Normalizar características por separado
        X_num_norm = scaler_num.transform(X_num_test)
        X_req_norm = scaler_req.transform(X_req_test)

        # Realizar predicciones
        y_pred = model.predict([
            X_num_norm,
            X_req_norm, 
            np.array(X_task_encoded).reshape(-1, 1)
        ])

        # Calcular métricas
        metrics = calculate_metrics(y_test, y_pred.flatten())

        # Imprimir resultados
        print("\nMétricas de Rendimiento del Modelo:")
        print("=====================================")
        print(f"Error Cuadrático Medio (MSE): {metrics['MSE']:.4f}")
        print(f"Raíz del Error Cuadrático Medio (RMSE): {metrics['RMSE']:.4f}")
        print(f"Error Absoluto Medio (MAE): {metrics['MAE']:.4f}")
        print(f"Coeficiente de Determinación (R²): {metrics['R2']:.4f}")
        print(f"Precisión (Accuracy): {metrics['Accuracy']:.4f}")
        print(f"Exactitud (Precision): {metrics['Precision']:.4f}")
        print(f"Recuperación (Recall): {metrics['Recall']:.4f}")
        print(f"Puntuación F1 (F1-Score): {metrics['F1']:.4f}")

        return metrics

    except Exception as e:
        print(f"Error durante la evaluación: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    evaluate_model()
