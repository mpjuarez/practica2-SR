#!/usr/bin/env python
# coding: utf-8

# # **Evaluación Sistema de Recomendación usando KNN basado en ítem**

# In[26]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import numpy as np

# Cargar el dataset de calificaciones y películas
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')


# # Dividir los datos en entrenamiento y prueba

# In[27]:


# Dividir el dataset en entrenamiento y prueba
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42) #42 por defecto

# Crear la matriz usuario-película para entrenamiento
train_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)


# # Algoritmo KNN basado en ítems usando la métrica coseno

# In[28]:


knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=10)
knn.fit(train_matrix.T.values)  # Transponer la matriz para basar el modelo en ítems


# # Función para predecir calificaciones basadas en ítems

# In[29]:


def predecir_calificaciones(user_id, movie_id):
    # Verificar si la película está en el conjunto de entrenamiento
    if movie_id not in train_matrix.columns:
        return 0  # Calificación predicha será 0 si la película no está
    
    movie_index = train_matrix.columns.get_loc(movie_id)
    distances, indices = knn.kneighbors([train_matrix.T.iloc[movie_index].values], n_neighbors=10)
    
    # Obtener las calificaciones de este usuario para las películas vecinas
    user_ratings = train_matrix.loc[user_id]
    score = 0
    suma_similitud = 0
    for i in range(1, len(distances.flatten())):  # Ignorar el primer vecino (la película misma)
        vecino_index = indices.flatten()[i]
        vecino_movie_id = train_matrix.columns[vecino_index]
        vecino_similitud = 1 - distances.flatten()[i]  # Coseno se convierte en similitud (1 - distancia)
        vecino_calificacion = user_ratings.get(vecino_movie_id, 0)
        if vecino_calificacion > 0:
            score += vecino_similitud * vecino_calificacion
            suma_similitud += vecino_similitud
    
    # Calcular el promedio ponderado de calificaciones
    return score / suma_similitud if suma_similitud > 0 else 0


# # Evaluación del modelo usando RMSE

# In[30]:


def evaluar_modelo(test_data):
    """
    Calcular el RMSE del modelo en el conjunto de prueba.
    """
    actual = []
    predicted = []
    
    for _, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        predicted_rating = predecir_calificaciones(user_id, movie_id)
        actual.append(actual_rating)
        predicted.append(predicted_rating)
    
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    return rmse


# # Función para recomendar películas

# In[31]:


def recomendar_peliculas(user_id, num_recommendations=5):
    """
    Obtener recomendaciones para un usuario específico.
    """
    user_ratings = train_matrix.loc[user_id]
    peliculas_no_vistas = user_ratings[user_ratings == 0].index
    recomendacion_scores = {}
    
    for movie_id in peliculas_no_vistas:
        predicted_rating = predecir_calificaciones(user_id, movie_id)
        if predicted_rating > 0:
            recomendacion_scores[movie_id] = predicted_rating
    
    # Ordenar por calificación predicha
    peliculas_recomendadas = sorted(recomendacion_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_movie_ids = [id for id, _ in peliculas_recomendadas[:num_recommendations]]
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)][['movieId', 'title']]
    
    return recommended_movies


# # Ejemplo de predicción de calificación 

# In[32]:


user_id = 3  # Cambia el ID del usuario para probar con otros usuarios
movie_id = 2  # Cambia el ID de la película para probar con otras películas

# Predicción de calificación para una película específica

prediccion = predecir_calificaciones(user_id, movie_id)
if prediccion:
    print(f"Calificación predicha para el usuario {user_id} en la película {movie_id}: {prediccion:.2f}")
else:
    print("No fue posible predecir la calificación.")


# # Ejemplo de recomendaciones para un usuario específico

# In[33]:


# Ejemplo de recomendaciones para un usuario específico
user_id = 2  # Cambia el ID del usuario para probar con otros usuarios
print(f"\nRecomendaciones para el usuario {user_id}:")
print(recomendar_peliculas(user_id))


# # Evaluar el modelo usando RMSE

# In[34]:


rmse_value = evaluar_modelo(test_data)
print(f"RMSE del modelo KNN basado en ítems en el conjunto de prueba: {rmse_value:.4f}")


# In[ ]:




