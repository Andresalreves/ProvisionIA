import pandas as pd
from predictor import AdaptiveHybridForecast, preprocess_and_explain,quick_forecast

archivo = pd.read_csv('datos_sinteticos.csv')

df = preprocess_and_explain(archivo,'fecha','ventas')

modelo = AdaptiveHybridForecast(
    date_column='fecha',
    target_column='ventas'
)
modelo.fit(df)
predicciones = modelo.predict(future_periods=90)
dashboard = modelo.create_interactive_dashboard(predicciones, df)
dashboard.show()