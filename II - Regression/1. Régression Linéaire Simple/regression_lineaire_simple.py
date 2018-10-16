# II - Régression Liénraire Simple

# 1. Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importer le dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #équiv. à 'Cells(rows, columns)' avec ':' = all
Y = dataset.iloc[:, -1].values #-1 désigne la dernière colonne du dataset

# 3. Diviser le dataset entre le Training set (2/3 du dataset) et le Test set (1/3)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1.0/3, random_state = 0)
# On  entre '1.0' pour définir un float permettant l'affichage de résultats floats et non entier réél (0)

# 4. Construction du modèle
# NOTES:
# Nous allons : 
# - importer une classe qui va construire le modèle
from sklearn.linear_model import LinearRegression
# - créer un objet qui sera le modèle de régression linéaire simple
regressor = LinearRegression()
# - Lier le régresseur au training set (X_train & Y_train)
regressor.fit(X_train, Y_train)

# 5. Faire de nouvelles prédictions
# On va créer un vecteur qui va contenir les 10 prédictions des 10 observations du Test set
y_pred = regressor.predict(X_test)
# Pour appliquer une prédiction sur une valeur hors du champs du dataset
regressor.predict(15) # on insère la valeur du nbre d'années d'xp dont on veut prédire le salaire : 15ans

# 6. Visualiser les résultats
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salaire vs Expérience')
plt.xlabel('Expérience')
plt.ylabel('Salaire')
plt.show()

