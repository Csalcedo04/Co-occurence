import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carga el corpus de texto
corpus = pd.read_csv('data/Valence_train_oc_es.csv', sep=',')

# Crea la matriz de co-ocurrencia
co_occurrence_matrix = np.zeros((len(corpus), len(corpus)))

for i in range(len(corpus)):
    for j in range(len(corpus)):
        # Cuenta el número de veces que la palabra i aparece con la palabra j
        co_occurrence_matrix[i, j] = corpus.iloc[i]['word'].count(corpus.iloc[j]['word'])

# Normaliza la matriz de co-ocurrencia
co_occurrence_matrix /= co_occurrence_matrix.sum()

# Predice la probabilidad de que la palabra "perro" aparezca después de la palabra "el"
probability_of_dog_after_the = co_occurrence_matrix[
    corpus[corpus['word'] == 'perro'].index[0], corpus[corpus['word'] == 'el'].index[0]]

# Imprime la probabilidad
print(probability_of_dog_after_the)

# Visualiza la matriz de co-ocurrencia
sns.heatmap(co_occurrence_matrix, annot=True)
# Importa matplotlib explícitamente
plt.show()
