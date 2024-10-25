import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_and_explain(df, date_column, target_column):
    # Paso 1: Mostrar información básica del DataFrame
    print(df.shape)
    print("Resumen de los datos iniciales:")
    print(df.info())
    print("\nPrimeras filas de los datos:")
    print(df.head())

    # Paso 2: Manejo de valores nulos
    print("\nPaso 2: Identificando valores nulos en cada columna...")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Visualización de valores nulos

    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Visualización de valores faltantes en los datos")
    plt.show()

    missing_data = df.isnull().sum()
    plt.figure(figsize=(10,6))
    missing_data.plot(kind='bar')
    plt.title("Cantidad de valores faltantes por variable")
    plt.ylabel("Cantidad de valores faltantes")
    plt.xlabel("Variables")
    plt.show()

    # Eliminamos filas con valores nulos en la columna objetivo para evitar problemas
    print(f"\nEliminando filas con valores nulos en la columna objetivo '{target_column}'...")
    #df = df.dropna(subset=[target_column]).copy()

    porcentajes_nulos = df.isnull().mean() * 100
    porcentajes_nulos = porcentajes_nulos.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    porcentajes_nulos.plot(kind='bar')
    plt.title('Porcentaje de valores nulos por columna')
    plt.ylabel('Porcentaje de valores nulos')
    plt.xlabel('Columnas')
    plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejor legibilidad
    plt.show()
    missing_data = df.isnull().sum()
    plt.figure(figsize=(10,6))
    missing_data.plot(kind='bar')
    plt.title("Cantidad de valores faltantes por variable")
    plt.ylabel("Cantidad de valores faltantes")
    plt.xlabel("Variables")
    plt.show()

    df = df.dropna().copy()

    # Paso 3: Convertir columna de fecha
    print(f"\nPaso 3: Asegurándonos de que la columna de fecha '{date_column}' esté en formato de fecha...")
    df[date_column] = pd.to_datetime(df[date_column])
    print(f"Tipo de datos de la columna '{date_column}':", df[date_column].dtype)

    # Visualización del objetivo en función del tiempo
    plt.figure(figsize=(10, 5))
    plt.plot(df[date_column], df[target_column])
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.title(f'Visualización de las ventas a lo largo del tiempo')
    plt.show()

    # Paso 4: Escalado y codificación de variables
    scalers = {}
    encoders = {}

    print("\nPaso 4: Escalando y codificando las columnas de características...")
    for col in df.columns:
        if col in [date_column, target_column]:
            continue  # Ignoramos columna de fecha y objetivo

        # Si la columna es numérica, aplicamos escalado
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Escalando columna numérica '{col}' para que esté en el rango [0, 1]...")
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            scalers[col] = scaler
            
            # Visualización antes y después del escalado
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(df[col] * scaler.scale_[0] + scaler.min_[0], kde=True, ax=ax[0], color='skyblue')
            ax[0].set_title(f"Distribución original de '{col}'")
            sns.histplot(df[col], kde=True, ax=ax[1], color='orange')
            ax[1].set_title(f"Distribución escalada de '{col}'")
            plt.show()

        # Si la columna es categórica, aplicamos codificación
        elif pd.api.types.is_object_dtype(df[col]):
            print(f"Codificando columna categórica '{col}' en valores numéricos...")
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
            
            # Visualización de la codificación
            plt.figure(figsize=(8, 5))
            sns.countplot(x=df[col], palette="viridis")
            plt.title(f"Distribución de la columna '{col}' después de la codificación")
            plt.xlabel(col)
            plt.ylabel('Conteo')
            plt.show()

    print("\nDatos después del preprocesamiento:")
    print(df.info())
    print(df.head())
    return df

class AdaptiveHybridForecast:
    def __init__(self, date_column=None, target_column=None):
        """
        Inicializa el modelo híbrido adaptativo.
        
        Args:
            date_column: Nombre de la columna de fecha (opcional)
            target_column: Nombre de la columna objetivo (opcional)
        """
        self.prophet_model = Prophet(yearly_seasonality='auto', 
                                   weekly_seasonality='auto', 
                                   daily_seasonality='auto')
        self.rf_model = RandomForestRegressor(n_estimators=100, 
                                            random_state=42)
        self.lstm_model = None
        self.scalers = {}
        self.label_encoders = {}
        self.date_column = date_column
        self.target_column = target_column
        self.feature_columns = None
        
    def _detect_date_column(self, df):
        """Detecta automáticamente la columna de fecha"""
        for col in df.columns:
            if df[col].dtype in ['datetime64[ns]', 'datetime64[ms]', 'datetime64[us]']:
                return col
            try:
                pd.to_datetime(df[col])
                return col
            except:
                continue
        raise ValueError("No se encontró una columna de fecha válida")
    
    def _detect_target_column(self, df):
        """Detecta la columna objetivo basada en heurísticas"""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        elif 'target' in numeric_cols:
            return 'target'
        elif 'y' in numeric_cols:
            return 'y'
        else:
            # Selecciona la columna numérica con mayor varianza
            variances = df[numeric_cols].var()
            return variances.idxmax()
    
    def _preprocess_data(self, df):
        """Preprocesa automáticamente el conjunto de datos"""
        df_processed = df.copy()
        
        # Detectar o verificar columna de fecha
        if self.date_column is None:
            self.date_column = self._detect_date_column(df)
        
        # Detectar o verificar columna objetivo
        if self.target_column is None:
            self.target_column = self._detect_target_column(df)
        
        # Convertir fecha a datetime si no lo es
        df_processed[self.date_column] = pd.to_datetime(df_processed[self.date_column])
        
        # Identificar columnas de características
        self.feature_columns = [col for col in df.columns 
                              if col not in [self.date_column, self.target_column]]
        
        # Preprocesar cada columna
        for col in df_processed.columns:
            if col == self.date_column:
                continue
                
            if df_processed[col].dtype in ['float64', 'int64']:
                # Escalar variables numéricas
                self.scalers[col] = MinMaxScaler()
                df_processed[col] = self.scalers[col].fit_transform(
                    df_processed[col].values.reshape(-1, 1)
                )
            elif df_processed[col].dtype == 'object':
                # Codificar variables categóricas
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(
                    df_processed[col]
                )
        
        return df_processed
    
    def _create_lstm_model(self, input_shape):
        """Crea un modelo LSTM adaptativo"""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape, 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _prepare_prophet_data(self, df):
        """Prepara los datos para Prophet"""
        return pd.DataFrame({
            'ds': df[self.date_column],
            'y': df[self.target_column]
        })
    
    def fit(self, df):
        """
        Entrena el modelo con cualquier conjunto de datos
        
        Args:
            df: DataFrame con al menos una columna de fecha y una numérica
        """
        # Preprocesar datos
        df_processed = self._preprocess_data(df)
        
        # 1. Entrenar Prophet
        prophet_df = self._prepare_prophet_data(df_processed)
        self.prophet_model.fit(prophet_df)
        
        # Generar predicciones de Prophet
        prophet_forecast = self.prophet_model.predict(prophet_df)
        prophet_predictions = prophet_forecast['yhat'].values
        
        # 2. Entrenar Random Forest
        rf_features = []
        rf_features.append(prophet_predictions)
        
        # Agregar features adicionales si existen
        for col in self.feature_columns:
            rf_features.append(df_processed[col].values)
        
        rf_features = np.column_stack(rf_features)
        self.rf_model.fit(rf_features, df_processed[self.target_column])
        rf_predictions = self.rf_model.predict(rf_features)
        
        # 3. Entrenar LSTM
        lookback = min(30, len(df_processed) // 3)  # Adaptativo según tamaño de datos
        X_lstm = []
        y_lstm = []
        
        for i in range(lookback, len(rf_predictions)):
            X_lstm.append(rf_predictions[i-lookback:i])
            y_lstm.append(rf_predictions[i])
        
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        self.lstm_model = self._create_lstm_model((lookback, 1))
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        
    def predict(self, future_periods=30):
        """
        Realiza predicciones para períodos futuros
        
        Args:
            future_periods: Número de períodos futuros a predecir
        """
        # Generar fechas futuras
        last_date = pd.to_datetime(self.prophet_model.history['ds'].iloc[-1])
        future_dates = pd.DataFrame({
            'ds': pd.date_range(start=last_date + timedelta(days=1), 
                              periods=future_periods)
        })
        
        # 1. Predicciones de Prophet
        prophet_future = self.prophet_model.predict(future_dates)
        prophet_predictions = prophet_future['yhat'].values
        
        # 2. Predicciones de Random Forest
        if len(self.feature_columns) > 0:
            # Usar últimos valores conocidos para features externos
            last_features = []
            for col in self.feature_columns:
                last_features.append(np.repeat(
                    self.rf_model.feature_importances_[-1], 
                    future_periods
                ))
            rf_features = np.column_stack([prophet_predictions] + last_features)
        else:
            rf_features = prophet_predictions.reshape(-1, 1)
            
        rf_predictions = self.rf_model.predict(rf_features)
        
        # 3. Predicciones de LSTM
        lookback = self.lstm_model.input_shape[1]
        lstm_input = rf_predictions[-lookback:].reshape(1, lookback, 1)
        lstm_predictions = []
        
        for _ in range(future_periods):
            next_pred = self.lstm_model.predict(lstm_input, verbose=0)
            lstm_predictions.append(next_pred[0, 0])
            lstm_input = np.roll(lstm_input, -1)
            lstm_input[0, -1, 0] = next_pred
            
        lstm_predictions = np.array(lstm_predictions)
        
        # Combinar predicciones con pesos adaptativos
        final_predictions = (0.3 * prophet_predictions + 
                           0.3 * rf_predictions + 
                           0.4 * lstm_predictions)
        
        # Desescalar predicciones
        if self.target_column in self.scalers:
            final_predictions = self.scalers[self.target_column].inverse_transform(
                final_predictions.reshape(-1, 1)
            ).flatten()
        
        return pd.DataFrame({
            'fecha': future_dates['ds'],
            'prediccion': final_predictions
        })
    
    def create_interactive_dashboard(self, predictions, historical_data=None):
        """
        Crea un dashboard interactivo con Plotly
        """
        fig = go.Figure()
        
        # Agregar datos históricos si están disponibles
        if historical_data is not None:
            fig.add_trace(go.Scatter(
                x=historical_data[self.date_column],
                y=historical_data[self.target_column],
                name='Histórico',
                mode='lines'
            ))
        
        # Agregar predicciones
        fig.add_trace(go.Scatter(
            x=predictions['fecha'],
            y=predictions['prediccion'],
            name='Predicción',
            mode='lines',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='Predicciones del Modelo Híbrido',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

def quick_forecast(df, date_column=None, target_column=None, future_periods=30):
    """
    Función de utilidad para realizar predicciones rápidas
    
    Args:
        df: DataFrame con los datos
        date_column: Nombre de la columna de fecha (opcional)
        target_column: Nombre de la columna objetivo (opcional)
        future_periods: Número de períodos futuros a predecir
    """
    model = AdaptiveHybridForecast(date_column, target_column)
    model.fit(df)
    predictions = model.predict(future_periods)
    dashboard = model.create_interactive_dashboard(predictions, df)
    
    return predictions, dashboard