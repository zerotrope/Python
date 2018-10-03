# I - Data Preprocessing

# 1. Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importer le dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #équiv. à 'Cells(rows, columns)' avec ':' = all
Y = dataset.iloc[:, -1].values #-1 désigne la dernière colonne du dataset

# 3. Gérer les données manquantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3]) #la borne supérieure est toujours exclue dans Python
X[:, 1:3] = imputer.transform(X[:, 1:3]) #On applique aux colonnes des NaN

# 4. Gérer les variables catégoriques
# NOTES:
# -> variables catégoriques >< valeurs numériques continues
# -> Dans notre dataset : les colonnes 'Country' (n.0) & 'Purchased' (n.3)
# -> On va donc enconder ces catégories sous forme de texte en valeurs
# numériques, ie : pour 'Purchased' on pourrait dire YES = 1 et NO = 0
# -> Une fois repérées, les variables catégoriques sont-elles nominales ou ordinales?
# -> Dans notre dataset, 'Country' : nominales, il n'y a pas d'ordres entre pays
# -> On va splitter la colonne 'Country' en n catégories différentes, ici 3 :
# France, German, Spain, et on va afficher des 1 pour la ligne du pays correspondant
# et des 0 pour les pays non associés. Cela s'appelle l'encodage par Dummy variables.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #transforme la colonne texte des pays
#en valeurs numériques associées : 0, 1 et 2
onehotencoder = OneHotEncoder(categorical_features = [0]) 
X = onehotencoder.fit_transform(X).toarray() #transforme la nouvelle colonne
#de valeurs numériques 0, 1 et 2 en trois colonnes avec des 0 et des 1
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y) #Idem avec YES/NO de la variable dépendante

# 5. Diviser le dataset entre le Training set (80% du dataset) et le Test set (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# 6. Feature Scaling
# NOTES:
#Consiste à mettre toutes nos variables sous la même échelle.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
