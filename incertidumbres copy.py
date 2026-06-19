import numpy as np
from scipy.stats import norm
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Creación del modelo de red bayesiana
modelo = DiscreteBayesianNetwork([
    ('HistorialCompras', 'Compra'), # El historial influye en la compra
    ('TiempoEnSitio', 'Compra'), # El tiempo en el sitio influye en la compra
    ('ClicoPromocion', 'Compra') # El clic en la promoción influye en la compra
])

# Definición de las distribuciones de probabilidad condicional (CPDs)
cpd_historial = TabularCPD(
    variable='HistorialCompras', 
    variable_card=2, # 2 valores posibles: 0 o 1
    values=[[0.7], # P(HistorialCompras=0) = 70%
            [0.3]] # P(HistorialCompras=1) = 30%
)

cpd_tiempo = TabularCPD(
    variable='TiempoEnSitio', 
    variable_card=2,  # 2 valores posibles: 0 o 1
    values=[[0.6], # P(TiempoEnSitio=0) = 60%
            [0.4]] # P(TiempoEnSitio=1) = 40%
)

cpd_promocion = TabularCPD(
    variable='ClicoPromocion',
    variable_card=2, # 2 valores posibles: 0 o 1
    values=[[0.8], # P(ClicoPromocion=0) = 80%
            [0.2]] # P(ClicoPromocion=1) = 20%
)

cpd_compra = TabularCPD(
   variable='Compra',
   variable_card=2, # 2 valores: 0 (no compra) o 1 (compra)
   values=[
       # Probabilidades de NO comprar (Compra=0)
       [0.9, 0.7, 0.8, 0.4, 0.6, 0.2, 0.3, 0.1],
       # Probabilidades de SÍ comprar (Compra=1)
       [0.1, 0.3, 0.2, 0.6, 0.4, 0.8, 0.7, 0.9]
   ],
   evidence=['HistorialCompras','TiempoEnSitio', 'ClicoPromocion'], # Variables que influyen en la compra
   evidence_card=[2, 2, 2] # Cada una tiene 2 valores posibles
)

# Añadir las CPDs al modelo
modelo.add_cpds(cpd_historial, cpd_tiempo, cpd_promocion, cpd_compra)

# Verificación del modelo
assert modelo.check_model(), "¡Error! El modelo tiene inconsistencias"
print(" Modelo validado correctamente")

# Realizar inferencias
inferencia = VariableElimination(modelo)

# Consulta 1: Probabilidad de compra con historial, poco tiempo y clic en promoción
resultado1 = inferencia.query(
    variables=['Compra'], # Variable que queremos predecir
    evidence={
        'HistorialCompras': 1, # El cliente SÍ tiene historial de compras
        'TiempoEnSitio': 0, # El cliente pasa poco tiempo en el sitio
        'ClicoPromocion': 1 # El cliente SÍ hizo clic en la promoción
    }
)
print("\nProbabilidad de compra (Historial=1, Tiempo=0, Promoción=1):")
print(resultado1)

# Consulta 2: Probabilidad de compra con historial, mucho tiempo y clic en promoción
resultado2 = inferencia.query(
    variables=['Compra'],
    evidence={
        'HistorialCompras': 1, # El cliente SÍ tiene historial de compras
        'TiempoEnSitio': 1, # El cliente pasó MUCHO tiempo en el sitio
        'ClicoPromocion': 1 # El cliente SÍ clicó en una promoción
    }
)
print("\nProbabilidad de compra (Historial=1, Tiempo=1, Promoción=1):")
print(resultado2)

# Análisis de tiempo en el sitio (distribución normal)
media_tiempo = 5
desviacion_estandar = 2
tiempo_observado = 6

probabilidad_tiempo = norm.cdf(
    tiempo_observado,
    loc=media_tiempo,
    scale=desviacion_estandar
)

print(f"\n\nANÁLISIS DE TIEMPO EN EL SITIO")
print(f"Tiempo observado: {tiempo_observado} minutos")
print(f"Media esperada: {media_tiempo} minutos")
print(f"Desviación estándar: {desviacion_estandar} minutos")
print(f"Probabilidad de pasar menos de {tiempo_observado} minutos: {probabilidad_tiempo:.2%}")
print(f"Esto significa que el {(probabilidad_tiempo):.1%} de clientes pasan menos tiempo.")