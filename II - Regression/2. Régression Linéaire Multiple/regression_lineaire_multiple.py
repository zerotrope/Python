# III - Régression Linéraire Multiple

# 1. Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importer le dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #équiv. à 'Cells(rows, columns)' avec ':' = all
Y = dataset.iloc[:, -1].values #-1 désigne la dernière colonne du dataset

# 3. Gérer les variables catégoriques
# Adapter l'indice de la colonne de variables catégoriques, ici : 3
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #transforme la colonne texte des états
#en valeurs numériques associées : 0, 1 et 2
onehotencoder = OneHotEncoder(categorical_features = [3]) 
X = onehotencoder.fit_transform(X).toarray() # les 3 dummy variables sont positionnées début du tableau
X = X[:, 1:]

# 4. Diviser le dataset entre le Training set (80% du dataset soient 40 obs.) et le Test set (20% soient 10)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

