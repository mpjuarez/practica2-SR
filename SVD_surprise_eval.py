#!/usr/bin/env python
# coding: utf-8

# # **Evaluación de sistema de recomendación basado en SVD usando el dataset movielens**
# Este enfoque proporciona una solución completa para entrenar, predecir, recomendar, y evaluar un modelo basado en SVD utilizando la librería Surprise.

# ## Importar librerias y Cargar Datos¶
# Se cargan las calificaciones y detalles de películas.

# In[40]:


from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Cargar el dataset de calificaciones desde un archivo CSV
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

### 1. Cargar los datos en el formato requerido por Surprise
#Reader preparara y cargar datos en un formato que Surprise pueda procesar. Define el rango de calificaciones

reader = Reader(rating_scale=(0.5, 5))  # Rango de calificaciones en el dataset
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# ### 2. Dividir los datos en entrenamiento y prueba

# In[15]:


trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# ### 3. Entrenar el modelo SVD
# Entrena el modelo SVD utilizando el conjunto de datos de entrenamiento. El modelo generará una matriz latente para usuarios y películas basada en los datos de calificación.

# In[26]:


model = SVD(n_factors=100, random_state=42)  # SVD con 50 componentes latentes
model.fit(trainset)


# ### 4. Evaluar el modelo en el conjunto de prueba

#  Evaluación de predicción de calificaciones usando MAE

# In[27]:


# Realizar predicciones en el conjunto de prueba
predictions = model.test(testset)

# Calcular las métricas de evaluación
# Mean Absolute Error(MAE)
mae = accuracy.mae(predictions, verbose=True) 


# ### Evaluar la calidad de las recomendaciones usando Precisión y recall

# In[28]:


# Calcular Precisión y Recall
def calcular_precision_recall(predictions, threshold=4.0):
    """
    Calcula precisión y recall basados en un umbral.
    """
    relevant_items = 0
    recommended_relevant = 0
    recommended_items = 0

    for pred in predictions:
        true_rating = pred.r_ui
        predicted_rating = pred.est

        # Considerar ítems relevantes si la calificación real >= threshold
        if true_rating >= threshold:
            relevant_items += 1

        # Considerar ítems recomendados relevantes si la predicción >= threshold
        if predicted_rating >= threshold:
            recommended_items += 1
            if true_rating >= threshold:
                recommended_relevant += 1

    # Calcular precisión y recall
    precision = recommended_relevant / recommended_items if recommended_items > 0 else 0
    recall = recommended_relevant / relevant_items if relevant_items > 0 else 0

    return precision, recall



# ### 5. Función para Predicción
# Predice la calificación que un usuario daría a una película específica.
# Retorna un objeto Predicción, del cual se extrae el valor predicho (est).

# In[29]:


def predecir_calificacion(user_id, movie_id):
    """
    Predice la calificación de un usuario para una película específica.
    
    Parámetros:
    - user_id: ID del usuario
    - movie_id: ID de la película
    
    Retorna:
    - Calificación predicha.
    """
    try:
        prediction = model.predict(user_id, movie_id)
        return prediction.est
    except Exception as e:
        print(f"Error al predecir: {e}")
        return None


# ### 6. Función para Generar Recomendaciones
# - Filtrar películas vistas: Se excluyen las películas que el usuario ya ha calificado.
# - Predecir calificaciones para películas no vistas: Usa la función predecir_calificacion para calcular las calificaciones predichas.
# - Ordenar y seleccionar Top-N: Ordena las películas no vistas por calificación predicha y selecciona las mejores

# In[30]:


def recomendar_top_n(user_id, num_recommendations=10):
    """
    Genera recomendaciones para un usuario específico.
    
    Parámetros:
    - user_id: ID del usuario
    - num_recommendations: Número de recomendaciones a devolver.
    
    Retorna:
    - DataFrame con las películas recomendadas.
    """
    # Obtener todas las películas del dataset
    all_movies = movies['movieId'].unique()
    
    # Filtrar películas ya vistas por el usuario
    peliculas_vistas = ratings[ratings['userId'] == user_id]['movieId'].values
    peliculas_no_vistas = [movie for movie in all_movies if movie not in peliculas_vistas]
    
    # Predecir calificaciones para todas las películas no vistas
    recomendaciones = []
    for movie_id in peliculas_no_vistas:
        prediccion = predecir_calificacion(user_id, movie_id)
        if prediccion is not None:
            recomendaciones.append((movie_id, prediccion))
    
    # Ordenar las películas por calificación predicha en orden descendente
    recomendaciones = sorted(recomendaciones, key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    # Crear un DataFrame con los títulos de las películas recomendadas
    recomendaciones_df = movies[movies['movieId'].isin([rec[0] for rec in recomendaciones])].copy()
    recomendaciones_df['Predicted Rating'] = [rec[1] for rec in recomendaciones]
    
    return recomendaciones_df




# # Ejemplo de Uso
# Predicción de calificación y Recomendación de películas

# In[31]:


# Predicción de calificación
user_id = 1  # Cambia el ID del usuario para probar con otros usuarios
movie_id = 2  # Cambia el ID de la película para probar con otras películas
prediccion = predecir_calificacion(user_id, movie_id)
print(f"\nPredicción de calificación para el usuario {user_id} en la película {movie_id}: {prediccion:.2f}" if prediccion else "No se pudo predecir la calificación.")

# Recomendación de películas
num_recommendations = 10
print(f"\nTop-{num_recommendations} recomendaciones para el usuario {user_id}:")
print(recomendar_top_n(user_id, num_recommendations))


# # Mostrar resultados de evaluación

# In[32]:


print(f"\nEvaluación del modelo SVD:")
print(f" - MAE: {mae:.4f}")


# In[33]:


# Calcular Precisión y Recall
precision, recall = calcular_precision_recall(predictions, threshold=4.0)
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")



# In[ ]:




