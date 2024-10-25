import pandas as pd
from predictor import AdaptiveHybridForecast, preprocess_and_explain,quick_forecast

archivo = pd.read_csv('datos_sinteticos.csv')

df = preprocess_and_explain(archivo,'fecha','ventas')

# Opción 1: Uso rápido con detección automática
predicciones, dashboard = quick_forecast(df)

# Opción 2: Uso con especificación manual de columnas
modelo = AdaptiveHybridForecast(
    date_column='fecha',  # opcional
    target_column='ventas'  # opcional
)
modelo.fit(df)
predicciones = modelo.predict(future_periods=30)
dashboard = modelo.create_interactive_dashboard(predicciones, df)