# -*- coding: utf-8 -*-
"""PrevisionAI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S7SD7PBj_ErN7IqTOOgh143aIeBtVt_L
"""

from google.colab import files

load = files.upload()

# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('datos_sinteticos.csv')

data.head()

data.info()

categoricas = data.select_dtypes(include=['object']).columns.tolist()
numericas = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Columnas categóricas:", categoricas)
print("Columnas numéricas:", numericas)

print(data.sample(n=10))

valores_faltantes = data.isnull().sum()
print(valores_faltantes)

# Visualizar en un heatmap
sns.heatmap(data.isnull(), cbar=False)
plt.show()

filas_con_faltantes = data[data['pib'].isnull()]

print(filas_con_faltantes)

def convertir_a_int_ignorar_nan(valor):
    """Convierte un valor a entero, ignorando los NaN.

    Args:
        valor: El valor a convertir.

    Returns:
        El valor convertido a entero si es posible, o el valor original si no.
    """

    if isinstance(valor, float) and not np.isnan(valor):
        return int(valor)
    else:
        return valor

# Aplicar la función solo a las columnas de tipo float64
float_cols = data.select_dtypes(include=['float64']).columns
data[float_cols] = data[float_cols].applymap(convertir_a_int_ignorar_nan)

print(data)

data.info()

# Eliminar las filas con al menos un valor nulo
data = data.dropna()

print(data)

# Iterar sobre las columnas y convertir a int si es float
for col in data.columns:
    if data[col].dtype == 'float64':
        data[col] = data[col].astype(int)

print(data)

data.count()

# Calcular la desviación estándar para las columnas numéricas
desviaciones_estandar = data[['ventas', 'pib', 'desempleo', 'confianza_consumidor', 'inflacion']].std()

# Imprimir las desviaciones estándar
print(desviaciones_estandar)

# Crear una gráfica de barras para representar las desviaciones estándar
desviaciones_estandar.plot(kind='bar')
plt.title('Desviación Estándar de las Variables Numéricas')
plt.xlabel('Variables')
plt.ylabel('Desviación Estándar')
plt.xticks(rotation=45)
plt.show()

# Calcula la media del atributo "pdays: dias de contacto al usuario antes de la campaña actual"
media = data['ventas'].mean()

print("Promedio de ventas:", media)

# Graficar un histograma del atributo "pdays"
plt.hist(data['ventas'], bins=20, color='skyblue', edgecolor='black')

# Agregar una línea vertical para mostrar la media
plt.axvline(media, color='red', linestyle='dashed', linewidth=1)

# Etiquetas y título
plt.xlabel('Valor de ventas')
plt.ylabel('Frecuencia')
plt.title('Histograma de ventas con Media')

# Mostrar la gráfica
plt.show()