import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SyntheticDataGenerator:
    def __init__(self, start_date='2020-01-01', n_years=4):
        """
        Inicializa el generador de datos sintéticos
        
        Args:
            start_date: Fecha de inicio para los datos
            n_years: Número de años de datos a generar
        """
        self.start_date = pd.to_datetime(start_date)
        self.n_years = n_years
        self.n_days = n_years * 365
        
    def _generate_base_trend(self):
        """Genera una tendencia base con crecimiento"""
        x = np.linspace(0, self.n_years, self.n_days)
        trend = 1000 + 500 * x + np.random.normal(0, 50, self.n_days)
        return trend
    
    def _add_seasonality(self, data):
        """Añade patrones estacionales"""
        # Estacionalidad anual
        annual = 200 * np.sin(2 * np.pi * np.linspace(0, self.n_years, self.n_days))
        
        # Estacionalidad mensual
        monthly = 100 * np.sin(2 * np.pi * np.linspace(0, self.n_years * 12, self.n_days))
        
        # Estacionalidad semanal
        weekly = 50 * np.sin(2 * np.pi * np.linspace(0, self.n_years * 52, self.n_days))
        
        return data + annual + monthly + weekly
    
    def _add_special_events(self, data):
        """Añade efectos de eventos especiales"""
        dates = pd.date_range(self.start_date, periods=self.n_days, freq='D')
        
        # Efecto de Black Friday (noviembre)
        for year in range(self.n_years):
            black_friday_idx = np.where(
                (dates.month == 11) & 
                (dates.day >= 23) & 
                (dates.day <= 30) & 
                (dates.year == self.start_date.year + year)
            )[0]
            data[black_friday_idx] *= 1.5
        
        # Efecto de Navidad (diciembre)
        christmas_idx = np.where((dates.month == 12) & (dates.day >= 15))[0]
        data[christmas_idx] *= 1.3
        
        # Efectos aleatorios de promociones
        n_promotions = self.n_years * 6  # 6 promociones por año
        for _ in range(n_promotions):
            promo_start = np.random.randint(0, self.n_days - 7)
            promo_duration = np.random.randint(3, 8)
            data[promo_start:promo_start + promo_duration] *= np.random.uniform(1.2, 1.4)
        
        return data
    
    def _generate_economic_indicators(self):
        """Genera indicadores económicos sintéticos"""
        dates = pd.date_range(self.start_date, periods=self.n_days, freq='D')
        
        # PIB trimestral
        gdp_base = np.linspace(100, 130, self.n_days) + np.random.normal(0, 2, self.n_days)
        gdp = pd.Series(gdp_base).rolling(90).mean()
        
        # Tasa de desempleo
        unemployment = (8 + np.sin(np.linspace(0, 4*np.pi, self.n_days)) + 
                      np.random.normal(0, 0.5, self.n_days))
        unemployment = pd.Series(unemployment).rolling(30).mean()
        
        # Índice de confianza del consumidor
        consumer_confidence = (100 + 20*np.sin(np.linspace(0, 8*np.pi, self.n_days)) + 
                             np.random.normal(0, 5, self.n_days))
        
        # Tasa de inflación
        inflation = (3 + 2*np.sin(np.linspace(0, 2*np.pi, self.n_days)) + 
                    np.random.normal(0, 0.3, self.n_days))
        inflation = pd.Series(inflation).rolling(90).mean()
        
        return pd.DataFrame({
            'fecha': dates,
            'pib': gdp,
            'desempleo': unemployment,
            'confianza_consumidor': consumer_confidence,
            'inflacion': inflation
        })
    
    def _generate_categorical_features(self):
        """Genera variables categóricas"""
        # Regiones
        regions = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
        
        # Categorías de productos
        product_categories = ['Electrónica', 'Ropa', 'Alimentos', 
                            'Hogar', 'Deportes', 'Libros']
        
        # Canales de venta
        sales_channels = ['Online', 'Tienda Física', 'Marketplace', 'Distribuidor']
        
        n_records = self.n_days * len(regions)  # Un registro por región por día
        
        return pd.DataFrame({
            'region': np.repeat(regions, self.n_days),
            'categoria_producto': np.random.choice(
                product_categories, 
                size=n_records
            ),
            'canal_venta': np.random.choice(
                sales_channels, 
                size=n_records
            )
        })
    
    def generate_dataset(self, add_noise=True):
        """
        Genera el conjunto de datos completo
        
        Args:
            add_noise: Si se debe añadir ruido aleatorio a los datos
            
        Returns:
            DataFrame con todos los datos generados
        """
        # Generar fechas base
        dates = pd.date_range(self.start_date, periods=self.n_days, freq='D')
        regions = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
        
        # Generar datos base para cada región
        all_data = []
        for region in regions:
            # Generar ventas base
            sales = self._generate_base_trend()
            sales = self._add_seasonality(sales)
            sales = self._add_special_events(sales)
            
            if add_noise:
                sales += np.random.normal(0, sales * 0.05)  # 5% de ruido
            
            # Datos específicos de la región
            region_data = pd.DataFrame({
                'fecha': dates,
                'ventas': sales,
                'region': region
            })
            
            all_data.append(region_data)
        
        # Combinar todos los datos
        df = pd.concat(all_data, ignore_index=True)
        
        # Añadir indicadores económicos
        economic_indicators = self._generate_economic_indicators()
        df = df.merge(economic_indicators, on='fecha')
        
        # Añadir variables categóricas
        categorical_data = self._generate_categorical_features()
        df = df.reset_index(drop=True)
        df = pd.concat([df, categorical_data[['categoria_producto', 'canal_venta']]], 
                      axis=1)
        
        # Añadir algunos valores faltantes de forma aleatoria
        for col in df.select_dtypes(include=[np.number]).columns:
            mask = np.random.random(len(df)) < 0.01  # 1% de valores faltantes
            df.loc[mask, col] = np.nan
        
        return df

def generate_test_data(start_date='2020-01-01', n_years=4, add_noise=True):
    """
    Función de utilidad para generar datos de prueba rápidamente
    
    Args:
        start_date: Fecha de inicio
        n_years: Número de años de datos
        add_noise: Si se debe añadir ruido aleatorio
        
    Returns:
        DataFrame con datos sintéticos
    """
    generator = SyntheticDataGenerator(start_date, n_years)
    return generator.generate_dataset(add_noise)

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos
    df = generate_test_data()
    
    # Mostrar información sobre el dataset
    print("\nInformación del dataset generado:")
    print("-" * 50)
    print(f"Dimensiones: {df.shape}")
    print("\nColumnas y tipos de datos:")
    print(df.dtypes)
    print("\nPrimeras 5 filas:")
    print(df.head())
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Guardar datos en CSV
    df.to_csv('datos_sinteticos.csv', index=False)
    print("\nDatos guardados en 'datos_sinteticos.csv'")