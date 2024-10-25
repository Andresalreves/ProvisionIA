import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta

class HybridForecastModel:
    def __init__(self):
        self.prophet_model = Prophet(yearly_seasonality=True, 
                                   weekly_seasonality=True, 
                                   daily_seasonality=True)
        self.rf_model = RandomForestRegressor(n_estimators=100, 
                                            random_state=42)
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        
    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, 
                 return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_lstm_data(self, data, lookback=30):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i])
        return np.array(X), np.array(y)
    
    def fit(self, df, target_col, external_features=None):
        # 1. Entrenar Prophet
        prophet_df = df[['ds', target_col]].rename(columns={target_col: 'y'})
        self.prophet_model.fit(prophet_df)
        
        # Generar predicciones de Prophet
        prophet_forecast = self.prophet_model.predict(prophet_df)
        prophet_predictions = prophet_forecast['yhat'].values
        
        # 2. Entrenar Random Forest con predicciones de Prophet y features externos
        if external_features is not None:
            rf_features = np.column_stack([prophet_predictions] + 
                                        [df[col] for col in external_features])
        else:
            rf_features = prophet_predictions.reshape(-1, 1)
            
        self.rf_model.fit(rf_features, df[target_col])
        rf_predictions = self.rf_model.predict(rf_features)
        
        # 3. Entrenar LSTM
        X_lstm, y_lstm = self.prepare_lstm_data(rf_predictions)
        self.lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        
    def predict(self, future_dates, external_features=None):
        # 1. Predicciones de Prophet
        prophet_future = self.prophet_model.predict(future_dates)
        prophet_predictions = prophet_future['yhat'].values
        
        # 2. Predicciones de Random Forest
        if external_features is not None:
            rf_features = np.column_stack([prophet_predictions] + external_features)
        else:
            rf_features = prophet_predictions.reshape(-1, 1)
            
        rf_predictions = self.rf_model.predict(rf_features)
        
        # 3. Predicciones de LSTM
        X_lstm, _ = self.prepare_lstm_data(rf_predictions)
        lstm_predictions = self.lstm_model.predict(X_lstm)
        
        # Combinar predicciones (promedio ponderado)
        final_predictions = (0.3 * prophet_predictions + 
                           0.3 * rf_predictions + 
                           0.4 * lstm_predictions.flatten())
        
        return final_predictions
    
    def create_interactive_simulation(self, df, target_col, feature_ranges):
        def update_plot(feature_values):
            future_dates = pd.DataFrame({
                'ds': pd.date_range(start=df['ds'].max(), 
                                  periods=30, 
                                  freq='D')
            })
            predictions = self.predict(future_dates, feature_values)
            
            fig = go.Figure()
            
            # Datos históricos
            fig.add_trace(go.Scatter(
                x=df['ds'],
                y=df[target_col],
                name='Histórico',
                mode='lines'
            ))
            
            # Predicciones
            fig.add_trace(go.Scatter(
                x=future_dates['ds'],
                y=predictions,
                name='Predicción',
                mode='lines',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title='Predicciones con Simulación Interactiva',
                xaxis_title='Fecha',
                yaxis_title='Valor',
                hovermode='x unified'
            )
            
            return fig

        return update_plot
    

# Ejemplo de uso
df = pd.DataFrame({
    'ds': [...],  # fechas
    'target': [...],  # variable objetivo
    'feature1': [...],  # variables económicas adicionales
    'feature2': [...]
})

# Inicializar y entrenar el modelo
model = HybridForecastModel()
model.fit(df, 'target', external_features=['feature1', 'feature2'])

# Realizar predicciones
future_dates = pd.DataFrame({'ds': pd.date_range(start='2024-01-01', periods=30)})
predictions = model.predict(future_dates, external_features=[...])