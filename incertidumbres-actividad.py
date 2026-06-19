# 1. Creando la Red Bayesiana
probabilidades = {
    "HistorialCompras": {0: 0.7, 1: 0.3},  # 0: No tiene historial, 1: Tiene historial
    "TiempoEnElSitio": {0: 0.6, 1: 0.4},   # 0: Poco tiempo, 1: Mucho tiempo
    "ClicóEnPromocion": {0: 0.8, 1: 0.2},  # 0: No clicó, 1: Clicó
    "Compra": {
        (0, 0, 0): 0.1,  # No tiene historial, poco tiempo, no clicó
        (0, 0, 1): 0.3,  # No tiene historial, poco tiempo, clicó
        (0, 1, 0): 0.2,  # No tiene historial, mucho tiempo, no clicó
        (0, 1, 1): 0.6,  # No tiene historial, mucho tiempo, clicó
        (1, 0, 0): 0.4,  # Tiene historial, poco tiempo, no clicó
        (1, 0, 1): 0.7,  # Tiene historial, poco tiempo, clicó
        (1, 1, 0): 0.8,  # Tiene historial, mucho tiempo, no clicó
        (1, 1, 1): 0.9   # Tiene historial, mucho tiempo, clicó
    }
}

# 2. Función para calcular la probabilidad conjunta de compra
def calcular_probabilidad_compra(evidencias):
    historial = evidencias["HistorialCompras"] 
    tiempo = evidencias["TiempoEnElSitio"] 
    promocion = evidencias["ClicóEnPromocion"] 

    prob_compra = probabilidades["Compra"][(historial, tiempo, promocion)] # Obtiene la probabilidad de compra basada en las evidencias
    prob_no_compra = 1 - prob_compra # Calcula la probabilidad de no compra como complemento de la probabilidad de compra

    return {"Comprar": prob_compra, "No Comprar": prob_no_compra} # Devuelve un diccionario con las probabilidades de comprar y no comprar basado en las evidencias proporcionadas

# 3. Probando con el escenario descrito
evidencias = {
    "HistorialCompras": 1,  # Cliente tiene historial de compras
    "TiempoEnElSitio": 0,   # Cliente pasó poco tiempo en el sitio
    "ClicóEnPromocion": 1    # Cliente clicó en promociones
}

resultados = calcular_probabilidad_compra(evidencias) # Calcula la probabilidad de compra basada en las evidencias
print("Probabilidades de Compra:") # Imprime las probabilidades de compra y no compra
for resultado, probabilidad in resultados.items(): # Itera sobre los resultados y sus probabilidades para imprimirlos
    print(f"{resultado}: {probabilidad:.2f}") # Formatea la salida para mostrar las probabilidades con dos decimales