import pandas as pd
import numpy as np

ruta = r"C:\Users\yhael\Documents\Yhael\UAQ\1 semestre\Aprendizaje Automatico\Analisis de base de datos\Datos faltantes\bd_endireh_2021_csv\TSDem.csv"

print("*****CARGANDO DATOS*****")
df = pd.read_csv(ruta, encoding='latin-1')
print(f"Total de registros: {len(df)}")
print(f"Total de variables: {len(df.columns)}")

print("\n" + "="*60)
print("1. ANALISIS DE DATOS FALTANTES")
print("="*60)

datos_faltantes = df.isnull().sum()
datos_faltantes_pct = (df.isnull().sum() / len(df)) * 100

df_faltantes = pd.DataFrame({
    'Variable': datos_faltantes.index,
    'Valores_Faltantes': datos_faltantes.values,
    'Porcentaje': datos_faltantes_pct.values
})

df_faltantes = df_faltantes[df_faltantes['Valores_Faltantes'] > 0].sort_values('Porcentaje', ascending=False)
print(df_faltantes.to_string(index=False))

print(f"\nTotal de variables con datos faltantes: {len(df_faltantes)}")
print(f"Total de valores faltantes: {df.isnull().sum().sum()}")

print("\n" + "="*60)
print("2. DETECCION DE DATOS ATIPICOS (METODO IQR)")
print("="*60)

def detectar_outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return len(outliers), limite_inferior, limite_superior

df_numericas = df.select_dtypes(include=[np.number])

print("\nVariables numericas analizadas:", len(df_numericas.columns))
print("\nOutliers detectados por variable:")

resultados_outliers = []
for col in df_numericas.columns:
    n_outliers, li, ls = detectar_outliers_iqr(df_numericas, col)
    if n_outliers > 0:
        resultados_outliers.append({
            'Variable': col,
            'Num_Outliers': n_outliers,
            'Pct_Outliers': (n_outliers/len(df))*100,
            'Lim_Inf': li,
            'Lim_Sup': ls
        })

df_outliers = pd.DataFrame(resultados_outliers)
if len(df_outliers) > 0:
    print(df_outliers.to_string(index=False))
else:
    print("No se detectaron outliers con el metodo IQR")

print("\n" + "="*60)
print("3. ANALISIS DE CARDINALIDAD")
print("="*60)

cardinalidad = df.nunique()
df_cardinalidad = pd.DataFrame({
    'Variable': cardinalidad.index,
    'Valores_Unicos': cardinalidad.values
}).sort_values('Valores_Unicos', ascending=False)

print("\nCardinalidad por variable:")
print(df_cardinalidad.to_string(index=False))

print(f"\nVariables con alta cardinalidad (>50 valores unicos): {len(df_cardinalidad[df_cardinalidad['Valores_Unicos'] > 50])}")
print(f"Variables con baja cardinalidad (<=5 valores unicos): {len(df_cardinalidad[df_cardinalidad['Valores_Unicos'] <= 5])}")

print("\n" + "="*60)
print("4. RESUMEN ESTADISTICO DE VARIABLES NUMERICAS")
print("="*60)

estadisticos = df_numericas.describe().T

print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "Variable", "Count", "Mean", "Std", "Min", "50%", "Max"))
print("-" * 70)
for idx, row in estadisticos.iterrows():
    print("{:<12} {:>10.0f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
        idx, row['count'], row['mean'], row['std'], row['min'], row['50%'], row['max']))

print("\n\n=== CODIGO LATEX ===")
print("\\begin{table}[h]")
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
