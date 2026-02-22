import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Ruta de los datos
ruta_datos = r"C:\Users\yhael\Documents\Yhael\UAQ\1 semestre\Aprendizaje Automatico\Analisis de base de datos\Datos faltantes\bd_endireh_2021_csv\TSDem.csv"
ruta_guardado = r"C:\Users\yhael\Documents\Yhael\UAQ\1 semestre\Aprendizaje Automatico\Analisis de base de datos\Datos faltantes\ActividadDatosFaltantes\Datos_faltantes"

# Paso 1: Carga del conjunto de datos con codificación latin-1
print("CARGANDO DATOS ENDIREH 2021...")
df = pd.read_csv(ruta_datos, encoding='latin-1')
print(f"Registros: {len(df)}, Variables: {len(df.columns)}")

# Paso 2: Cálculo del porcentaje de valores nulos por variable
print("\n=== CALCULO DE VALORES FALTANTES POR VARIABLE ===")
nulos_pct = (df.isnull().sum() / len(df)) * 100
df_nulos = pd.DataFrame({'Variable': nulos_pct.index, 'Porcentaje_Nulos': nulos_pct.values})
df_nulos = df_nulos.sort_values('Porcentaje_Nulos', ascending=False)
print(df_nulos[df_nulos['Porcentaje_Nulos'] > 0].to_string(index=False))

# Paso 3: Descarte de variables con >50% nulos
print("\n=== DESCARTE DE VARIABLES CON >50% NULOS ===")
umbral = 50
descartadas = df_nulos[df_nulos['Porcentaje_Nulos'] > umbral]['Variable'].tolist()
print(f"Variables descartadas ({len(descartadas)}): {descartadas}")

df_limpio = df.drop(columns=descartadas)
print(f"Variables restantes: {len(df_limpio.columns)}")

# Seleccionar variables numéricas
df_numericas = df_limpio.select_dtypes(include=[np.number])
cols_seleccionadas = ['EDAD', 'NIV']

# Paso 4: Detección de datos atípicos mediante el método del Rango Intercuartil (IQR)
# Se calculan los percentiles 25 y 75 para cada variable
# Se consideran outliers los valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
print("\n=== DETECCION DE DATOS ATIPICOS (METODO IQR) ===")
for col in cols_seleccionadas:
    Q1 = df_numericas[col].quantile(0.25)
    Q3 = df_numericas[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df_numericas[(df_numericas[col] < limite_inferior) | (df_numericas[col] > limite_superior)]
    print(f"{col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Limite inferior: {limite_inferior:.2f}, Limite superior: {limite_superior:.2f}")
    print(f"  Outliers detectados: {len(outliers)}")

# Paso 5: Generación de boxplots iniciales para visualizar distribución
print("\n=== GENERANDO BOXPLOTS INICIALES ===")
fig, ax = plt.subplots(figsize=(12, 6))
df_numericas[cols_seleccionadas].boxplot(ax=ax)
plt.title('Distribucion antes de imputacion')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{ruta_guardado}/figures/boxplot_inicial.png', dpi=150)
plt.close()
print("Guardado: figures/boxplot_inicial.png")

# Paso 6 y 7: Selección específica de método de imputación y aplicación
# EDAD: Imputación por media (distribución simétrica)
# NIV: Imputación por moda
print("\n=== IMPUTACION ESPECIFICA ===")
print("EDAD: imputacion por MEDIA")
print("NIV: imputacion por MODO")

df_imputado = df_numericas.copy()

# Imputación por media para EDAD
if 'EDAD' in df_numericas.columns:
    imputer = SimpleImputer(strategy='mean')
    df_imputado['EDAD'] = imputer.fit_transform(df_numericas[['EDAD']])
    print("  EDAD: Media aplicada")

# Imputación por moda para NIV
if 'NIV' in df_numericas.columns:
    imputer_niv = SimpleImputer(strategy='most_frequent')
    df_imputado['NIV'] = imputer_niv.fit_transform(df_numericas[['NIV']])
    print("  NIV: Moda aplicada")

# Paso 8: Normalización con MinMaxScaler para escalar al rango [0,1]
print("\n=== NORMALIZACION CON MINMAXSCALER ===")
scaler = MinMaxScaler()
df_normalizado = pd.DataFrame(
    scaler.fit_transform(df_imputado),
    columns=df_imputado.columns,
    index=df_imputado.index
)
print("Normalizacion completada")

# Generación de boxplots finales para comparación
print("\n=== COMPARACION ESPECIFICA EDAD Y NIV ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].boxplot(df_numericas['EDAD'].dropna())
axes[0, 0].set_title('EDAD - Antes')
axes[0, 0].set_ylabel('Valor')

axes[0, 1].boxplot(df_imputado['EDAD'])
axes[0, 1].set_title('EDAD - Despues (Media)')
axes[0, 1].set_ylabel('Valor')

axes[1, 0].boxplot(df_numericas['NIV'].dropna())
axes[1, 0].set_title('NIV - Antes')
axes[1, 0].set_ylabel('Valor')

axes[1, 1].boxplot(df_imputado['NIV'])
axes[1, 1].set_title('NIV - Despues (Moda)')
axes[1, 1].set_ylabel('Valor')

plt.tight_layout()
plt.savefig(f'{ruta_guardado}/figures/boxplot_edad_niv.png', dpi=150)
plt.close()
print("Guardado: figures/boxplot_edad_niv.png")

# Resumen estadístico
print("\n=== RESUMEN ESTADISTICO ===")
estadisticos = df_imputado.describe().T
print(estadisticos.to_string())

# Generación de código LaTeX para tabla resumen

# print("\n=== CODIGO LATEX TABLA RESUMEN ===")
# print("\\begin{table}[H]")
# print("\\centering")
# print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
# print("\\hline")
# print("Variable & Count & Mean & Std & Min & 25\\% & 50\\% & Max \\\\")
# print("\\hline")
# for idx, row in estadisticos.iterrows():
#     print(f"{idx} & {row['count']:.0f} & {row['mean']:.2f} & {row['std']:.2f} & {row['min']:.2f} & {row['25%']:.2f} & {row['50%']:.2f} & {row['max']:.2f} \\\\")
# print("\\hline")
# print("\\end{tabular}")
# print("\\caption{Resumen estadistico de variables numericas}")
# print("\\label{tab:resumen}")
# print("\\end{table}")
# print("Referencia: \\ref{tab:resumen}")

print("\n=== RESUMEN FINAL ===")
print(f"Total variables analizadas: {len(df.columns)}")
print(f"Variables descartadas (>50% nulos): {len(descartadas)}")
print(f"EDAD: Imputacion por MEDIA")
print(f"NIV: Imputacion por MODO")
