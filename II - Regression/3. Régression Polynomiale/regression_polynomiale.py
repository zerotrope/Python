# IV - Regression Polynomiale

# 1. Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importer le dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #Equiv. a 'Cells(rows, columns)' avec ':' tout seul = all
Y = dataset.iloc[:, -1].values #-1 designe la derniere colonne du dataset

# 3. Construction du modele
# NOTES:
# Nous allons creer des 'polynomial features', des puissances des variables X ajoutees au X existant :
# - importer une classe qui va construire le modele tout en conservant 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# - creer un objet qui sera le modele de regression polynomiale de degre 2, 3...etc. le plus, le mieux
# mais attention a "l'overfitting' ou le modele aura trop bien appris les correlations
# on verifie l'existance d'overfitting ou non par un Training_Set pousse a creer de nouvelles variables
# dont on observera si elles sont trop eloignees de la realite. 
poly_regressor = PolynomialFeatures(degree = 4)
# - On va utiliser la methode Fit_transform pour lier poly_reg a X ET transformer X afin qu'elle comprenne
# que les nouvelles variables creees vont etre ajoutees en une nouvelle colonne, on cree une nouvelle
# matrice de variables independantes
X_poly = poly_regressor.fit_transform(X)
# - creer un objet qui sera le modele de regression lineaire simple
regressor = LinearRegression()
# - Lier le regresseur au training set (X_train & Y_train)
regressor.fit(X_poly, Y)

# 4. Visualiser les resultats
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X_poly), color = 'blue')
plt.title('Salaire vs Expérience')
plt.xlabel('Expérience')
plt.ylabel('Salaire')
plt.show()
