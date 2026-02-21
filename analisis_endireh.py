import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

ruta_datos = r"C:\Users\yhael\Documents\Yhael\UAQ\1 semestre\Aprendizaje Automatico\Analisis de base de datos\Datos faltantes\bd_endireh_2021_csv\TSDem.csv"
ruta_guardado = r"C:\Users\yhael\Documents\Yhael\UAQ\1 semestre\Aprendizaje Automatico\Analisis de base de datos\Datos faltantes\ActividadDatosFaltantes\Datos_faltantes"

print("CARGANDO DATOS ENDIREH 2021...")
df = pd.read_csv(ruta_datos, encoding='latin-1')
print(f"Registros: {len(df)}, Variables: {len(df.columns)}")

print("\n=== CALCULO DE VALORES FALTANTES POR VARIABLE ===")
nulos_pct = (df.isnull().sum() / len(df)) * 100
df_nulos = pd.DataFrame({'Variable': nulos_pct.index, 'Porcentaje_Nulos': nulos_pct.values})
df_nulos = df_nulos.sort_values('Porcentaje_Nulos', ascending=False)
print(df_nulos[df_nulos['Porcentaje_Nulos'] > 0].to_string(index=False))

print("\n=== DESCARTE DE VARIABLES CON >50% NULOS ===")
umbral = 50
descartadas = df_nulos[df_nulos['Porcentaje_Nulos'] > umbral]['Variable'].tolist()
print(f"Variables descartadas ({len(descartadas)}): {descartadas}")

df_limpio = df.drop(columns=descartadas)
print(f"Variables restantes: {len(df_limpio.columns)}")

df_numericas = df_limpio.select_dtypes(include=[np.number])
cols_seleccionadas = ['EDAD', 'NIV']

print("\n=== IMPUTACION ESPECIFICA ===")
print("EDAD: imputacion por MEDIA")
print("NIV: imputacion por REGRESION")

df_imputado = df_numericas.copy()

if 'EDAD' in df_numericas.columns:
    imputer = SimpleImputer(strategy='mean')
    df_imputado['EDAD'] = imputer.fit_transform(df_numericas[['EDAD']])
    print("  EDAD: Media aplicada")

if 'NIV' in df_numericas.columns and 'EDAD' in df_numericas.columns:
    datos_validos = df_numericas[['NIV', 'EDAD']].dropna()
    X = datos_validos[['EDAD']].values
    y = datos_validos['NIV'].values
    modelo = LinearRegression()
    modelo.fit(X, y)
    nulos = df_numericas['NIV'].isnull()
    if nulos.sum() > 0:
        X_pred = df_numericas.loc[nulos, ['EDAD']].values
        df_imputado.loc[nulos, 'NIV'] = modelo.predict(X_pred)
    print(f"  NIV: Regresion aplicada (R2={modelo.score(X, y):.4f})")

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
axes[1, 1].set_title('NIV - Despues (Regresion)')
axes[1, 1].set_ylabel('Valor')

plt.tight_layout()
plt.savefig(f'{ruta_guardado}/figures/boxplot_edad_niv.png', dpi=150)
plt.close()
print("Guardado: figures/boxplot_edad_niv.png")

print("\n=== RESUMEN ESTADISTICO ===")
estadisticos = df_imputado.describe().T
print(estadisticos.to_string())

print("\n=== CODIGO LATEX TABLA RESUMEN ===")
print("\\begin{table}[H]")
print("\\centering")
print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
print("\\hline")
print("Variable & Count & Mean & Std & Min & 25\\% & 50\\% & Max \\\\")
print("\\hline")
for idx, row in estadisticos.iterrows():
    print(f"{idx} & {row['count']:.0f} & {row['mean']:.2f} & {row['std']:.2f} & {row['min']:.2f} & {row['25%']:.2f} & {row['50%']:.2f} & {row['max']:.2f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{Resumen estadistico de variables numericas}")
print("\\label{tab:resumen}")
print("\\end{table}")
print("Referencia: \\ref{tab:resumen}")

print("\n=== RESUMEN FINAL ===")
print(f"Total variables analizadas: {len(df.columns)}")
print(f"Variables descartadas (>50% nulos): {len(descartadas)}")
print(f"EDAD: Imputacion por MEDIA")
print(f"NIV: Imputacion por REGRESION")
